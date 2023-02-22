#include "tensorlite/tensor_op/tensor_iterator.h"

#include <algorithm>

#include "tensorlite/utils/logging.h"

namespace tl {

void TensorIterator::FixTensors() {
  num_inputs_ = inputs_.size();
  num_outputs_ = outputs_.size();
  auto num_tensors = NumTensors();

  // dangerous section: NumTensors/NumInputs/NumOutputs
  // calling from here may be invalid
  operands_ = std::move(outputs_);
  operands_.reserve(num_tensors);
  outputs_.clear();
  for (const auto &t : inputs_) {
    operands_.push_back(t);
  }
  inputs_.clear();

  // we are safe to call NumTensors/NumInputs/NumOutputs now
  has_tensors_fixed_ = true;
}

void TensorIterator::InitializeShape() {
  CHECK(has_tensors_fixed_);

  size_t max_rank = 0;
  for (const auto &tensor : operands_) {
    max_rank = std::max(tensor.Rank(), max_rank);
  }
  CHECK_NE(max_rank, 0);

  shape_ = std::vector<shape_elem_t>(max_rank, shape_elem_t(1));
  has_shape_initialized_ = true;
}

void TensorIterator::BroadcastShape() {
  CHECK(has_shape_initialized_);
  auto max_rank = shape_.size();

  for (auto &tensor : operands_) {
    auto offset = max_rank - tensor.Rank();
    auto &tensor_shape = tensor.GetShapeWithStride();
    tensor_shape.ResetRank(max_rank);

    // move shape elements to make them align at the least
    // significant axis
    for (auto i = max_rank - 1; i >= offset && i < max_rank; --i) {
      tensor_shape.Shape(i) = tensor_shape.Shape(i - offset);
      tensor_shape.Stride(i) = tensor_shape.Stride(i - offset);

      if (tensor_shape.Shape(i) == 1) {
        tensor_shape.Stride(i) = 0; // for broadcast
      } else {
        if (shape_[i] == 1) {
          shape_[i] = tensor_shape.Shape(i);
        } else {
          CHECK_EQ(shape_[i], tensor_shape.Shape(i));
        }
      }
    }

    // padding shape elements
    for (auto i = 0; i < offset; ++i) {
      tensor_shape.Shape(i) = 1;
      tensor_shape.Stride(i) = 0;
    }
  }

  has_shape_broadcasted_ = true;
}

bool TensorIterator::CanCompress(int dim0, int dim1) const {
  // assume dim1 > dim0, where dim1 is the less significant axis
  if (dim1 <= dim0)
    return false;

  // squeeze
  if (shape_[dim0] == 1 || shape_[dim1] == 1)
    return true;

  for (const auto &tensor : operands_) {
    const auto &shape = tensor.GetShapeWithStride();
    if (shape.Stride(dim1) * shape.Stride(dim0) == 0) { // broadcast case
      return false;
    } else if (shape.Stride(dim0) != (shape.Shape(dim1) * shape.Stride(dim1))) {
      return false;
    }
  }
  return true;
}

void TensorIterator::CompressAxes(int dim0, int dim1) {
  if (dim1 <= dim0)
    return;

  for (auto &tensor : operands_) {
    auto &shape = tensor.GetShapeWithStride();
    shape.Shape(dim1) *= shape.Shape(dim0);
    if (shape.Stride(dim1) == 0) { // 0 when axis_size == 1
      shape.Stride(dim1) = shape.Stride(dim0);
    }

    // eat dim0
    shape.Stride(dim0) = 0;
    shape.Shape(dim0) = 1;
  }
  shape_[dim1] *= shape_[dim0];
  shape_[dim0] = 1;
}

void TensorIterator::CompressShape() {
  // no need for compression
  if (Rank() <= 1) {
    has_shape_compressed_ = true;
    return;
  }

  auto base = Rank() - 1;
  auto forward = base - 1;

  while (forward < base) {
    if (CanCompress(forward, base)) {
      CompressAxes(forward, base);
      --forward;
    } else {
      // if cannot compress, move element at 'forward' to the position
      // next to 'base' to make the compressed shape array compact,
      // then move 'base' position.
      --base;
      if (base != forward) {
        // use CompressAxes to move axis from 'forward' position
        // to 'base' position, this is valid since current 'base' position
        // has been eaten (stride[base] = 0, shape[base] = 1)
        CompressAxes(forward, base);
      }
    }
  }

  if (base == 0) {
    has_shape_compressed_ = true;
    return;
  }

  auto new_rank = Rank() - base;
  for (auto &tensor : operands_) {
    auto &shape = tensor.GetShapeWithStride();
    for (auto i = 0; i < new_rank; ++i) {
      shape.Shape(i) = shape.Shape(i + base);
      shape.Stride(i) = shape.Stride(i + base);
    }
    shape.ResetRank(new_rank);
  }

  for (auto i = 0; i < new_rank; ++i) {
    shape_[i] = shape_[i + base];
  }
  shape_.resize(new_rank);
  has_shape_compressed_ = true;
}

std::vector<size_t> TensorIterator::GetStridesInBytes() const {
  size_t num_tensors = NumTensors();
  size_t rank = Rank();
  std::vector<size_t> strides(num_tensors * rank, 0);

  for (size_t t = 0; t < num_tensors; ++t) {
    size_t dtype_size = operands_[t].GetDataType().Size();
    for (size_t i = 0; i < rank; ++i) {
      strides[i + t * rank] =
          operands_[t].GetShapeWithStride().Stride(i) * dtype_size;
    }
  }

  return std::move(strides);
}

void TensorIterator::ForEach(loop2d_t loop) {
  CHECK(IsValid());

  // init data ptrs
  size_t num_tensors = NumTensors();
  size_t rank = Rank();
  std::vector<char *> base_dptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    base_dptrs[i] = reinterpret_cast<char *>(operands_[i].RawPtr());
  }

  // init stride for loop
  size_t stride_axis = std::max(rank, 2ULL);
  int stride_idx = static_cast<int>(stride_axis - 1);
  std::vector<size_t> loop_stride(num_tensors * stride_axis, 0);
  for (int i = static_cast<int>(rank - 1); i >= 0; --i, --stride_idx) {
    for (size_t t = 0; t < operands_.size(); ++t) {
      loop_stride[stride_idx * num_tensors + t] =
          operands_[t].GetShapeWithStride().Stride(i) *
          operands_[t].GetDataType().Size();
    }
  }
  size_t inner_size = shape_[rank - 1];
  size_t outer_size = (rank > 1) ? shape_[rank - 2] : 1;

  if (rank <= 2) {
    loop(base_dptrs.data(), loop_stride.data(), inner_size, outer_size);
  } else {
    auto counter = IndexCounter(shape_);
    std::vector<char *> dptrs(num_tensors);
    std::vector<size_t> stride_bytes = GetStridesInBytes();
    while (!counter.IsFinish()) {
      GetDataPtrs(dptrs, base_dptrs, counter.Index(), stride_bytes);
      loop(dptrs.data(), loop_stride.data(), inner_size, outer_size);
      counter.Advance(rank - 3);
    }
  }
}

void TensorIterator::GetDataPtrs(
    std::vector<char *> &dptrs, const std::vector<char *> &base,
    const std::vector<shape_elem_t> &index,
    const std::vector<size_t> &stride_bytes) const {
  size_t num_tensors = NumTensors();
  size_t rank = Rank();
  CHECK_EQ(dptrs.size(), num_tensors);
  CHECK_EQ(index.size(), rank);

  for (size_t t = 0; t < num_tensors; ++t) {
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {
      offset += index[i] * stride_bytes[t * rank + i];
    }
    dptrs[t] = base[t] + offset;
  }
}

} // namespace tl
