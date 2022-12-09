#include "tensorlite/tensor_op/tensor_iterator.h"

#include <algorithm>

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

    // move shape elements to make them align at the least
    // significant axis
    for (auto i = max_rank - 1; i >= offset; --i) {
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
  if (shape_[dim0] == 1 || shape_[dim1] == 0)
    return true;

  for (const auto &tensor : operands_) {
    const auto &shape = tensor.GetShapeWithStride();
    if (shape.Shape(dim0) != (shape.Shape(dim1) * shape.Stride(dim1)) &&
        shape.Stride(dim1) != 0) {
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
    if (shape.Stride(dim0) != 0) { // 0 when axis_size == 1
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

} // namespace tl
