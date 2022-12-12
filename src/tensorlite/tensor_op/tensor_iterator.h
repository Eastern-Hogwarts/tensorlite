#ifndef TENSORLITE_TENSOR_OP_TENSOR_ITERATOR_H_
#define TENSORLITE_TENSOR_OP_TENSOR_ITERATOR_H_

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensorlite/tensor.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/tool_functions.h"

namespace tl {

/**
 * \brief A range on one dimension (axis)
 *
 * \tparam IndexTy Type of index
 */
template <typename IndexTy> struct IndexRange {
  IndexTy start;
  IndexTy end;
  IndexRange() = default;
  IndexRange(IndexTy s, IndexTy e) : start(s), end(e) {}
};

/**
 * \brief A helper class for multi-axes looping.
 *
 * You can view this class as an variable index vector with shape.
 */
class IndexCounter {
public:
  IndexCounter() = default;

  /**
   * \brief Construct a new IndexCounter object
   *
   * \param shape The shape of corresponding multi-dimensional loop
   */
  IndexCounter(const std::vector<shape_elem_t> &shape)
      : shape_(shape), index_(std::vector<shape_elem_t>(shape.size(), 0)) {}

  /**
   * \brief Check whether this loop is finish (whether this index reaching to
   * the end)
   */
  bool IsFinish() const {
    if (overflow_)
      return true;
    bool finish = true;
    for (size_t i = 0; i < Rank(); ++i) {
      finish &= (index_[i] >= shape_[i]);
    }
    return finish;
  }

  /**
   * \brief Advance the index along a specific axis with a specific step size
   *
   * \param axis_idx The axis where the index to be advanced
   * \param step Step size, default = 1
   */
  void Advance(size_t axis_idx, shape_elem_t step = 1) {
    CHECK(axis_idx < Rank());
    shape_elem_t carry = step;
    for (size_t i = axis_idx; i <= axis_idx; --i) {
      index_[i] += carry;
      carry = index_[i] / shape_[i];
      index_[i] %= shape_[i];
    }
    if (carry)
      overflow_ = true;
  }

  /**
   * \brief Reset the index to all zeros
   */
  void Reset() {
    memset(index_.data(), 0, sizeof(shape_elem_t) * index_.size());
    overflow_ = false;
  }

  /**
   * \brief Get index vector
   *
   * \return const std::vector<shape_elem_t>&
   */
  const std::vector<shape_elem_t> &Index() const { return index_; }

  /**
   * \brief Get shape vector
   *
   * \return const std::vector<shape_elem_t>&
   */
  const std::vector<shape_elem_t> &Shape() const { return shape_; }

  /**
   * \brief Number of axes of this multi-dimensional loop
   *
   * \return size_t
   */
  size_t Rank() const { return shape_.size(); }

private:
  // The shape of corresponding loop
  std::vector<shape_elem_t> shape_;

  // The index value
  std::vector<shape_elem_t> index_;

  // whether current index value if overflow
  bool overflow_;
};

/**
 * \brief TensorIterator is a helper class for element-wise operations, such as
 * arithmetic, comparisons, and trigonometric functions. It handles
 * broadcasting and type conversions of operands.
 *
 * This is inspired by NumPy's Array Iterator API (NpyIter) and
 * TensorIterator in Pytorch.
 *
 * The files Loops.h provide functions to build kernels that use
 * TensorIterator.
 *
 * \note this class is not thread-safe, each thread should have its own copy
 */
class TensorIterator {
public:
  TensorIterator() = default;

  ~TensorIterator() = default;

  TensorIterator(const TensorIterator &) = default;

  TensorIterator(TensorIterator &&) = default;

  TensorIterator &operator=(const TensorIterator &) = default;

  TensorIterator &operator=(TensorIterator &&) = default;

  void AddInput(const Tensor &in_tensor) {
    CHECK(!has_tensors_fixed_);
    inputs_.push_back(in_tensor);
  }

  void AddOutput(const Tensor &out_tensor) {
    CHECK(!has_tensors_fixed_);
    outputs_.push_back(out_tensor);
  }

  size_t NumInputs() const {
    return has_tensors_fixed_ ? num_inputs_ : inputs_.size();
  }

  size_t NumOutputs() const {
    return has_tensors_fixed_ ? num_outputs_ : outputs_.size();
  }

  size_t NumTensors() const { return NumInputs() + NumOutputs(); }

  /**
   * \brief Get the rank of shape of this iterator
   *
   * \return size_t
   */
  size_t Rank() const {
    CHECK(has_shape_initialized_);
    return shape_.size();
  }

  /**
   * \brief Get the number of elements iterated by this iterators
   *
   * \return shape_elem_t
   */
  shape_elem_t NumElem() const { return ShapeNumElem<shape_elem_t>(shape_); }

  /**
   * \brief Get the shape of this iterator after shape broadcast
   *
   * \return const std::vector<shape_elem_t>&
   */
  const std::vector<shape_elem_t> &Shape() const {
    CHECK(IsValid());
    return shape_;
  }

  /**
   * \brief Get the shape of this iterator after shape broadcast
   *
   * \return std::vector<shape_elem_t>&
   */
  std::vector<shape_elem_t> &Shape() {
    CHECK(IsValid());
    return shape_;
  }

  /**
   * \brief Get tensors of this iterator in vector
   *
   * \return const std::vector<Tensor>&
   */
  const std::vector<Tensor> &Tensors() const {
    CHECK(IsValid());
    return operands_;
  }

  /**
   * \brief Get tensors of this iterator in vector
   *
   * \return std::vector<Tensor>&
   */
  std::vector<Tensor> &Tensors() {
    CHECK(IsValid());
    return operands_;
  }

  /**
   * \brief Check whether this iterator is ready for use.
   *
   * A iterator is valid for use if it has been built.
   */
  bool IsValid() const {
    return HasShapeInitialized() && HasTensorFixed() && HasShapeBroadCasted();
  }

  bool HasShapeInitialized() const { return has_shape_initialized_; }

  bool HasTensorFixed() const { return has_tensors_fixed_; }

  bool HasShapeBroadCasted() const { return has_shape_broadcasted_; }

  bool HasShapeCompressed() const { return has_shape_compressed_; }

  /**
   * \brief Fix input/output tensors and build this iterator
   */
  void Build() {
    FixTensors();
    InitializeShape();
    BroadcastShape();
    CompressShape();
  }

  /**
   * \brief Get strides in bytes of all tensors in this iterator
   *
   * \return std::vector<size_t> A 1-dim vector with size equals to
   *         NumTensor() * Rank() holding all strides in bytes.
   */
  std::vector<size_t> GetStridesInBytes() const;

  using loop2d_t =
      function_reference<void(char **, const size_t *, size_t, size_t)>;

  /**
   * \brief Iterator over a Loop2d object, this specific function is for
   * elementwise tensor op on CPU device.
   *
   * \param loop2d_ref The Loop2d Object
   */
  void ForEach(loop2d_t loop2d_ref);

private:
  /**
   * \brief Fix input/output tensors of this iterator.
   *
   * After this function no more input/ouptut tensor is not allowed to be
   * added.
   */
  void FixTensors();

  /**
   * \brief Initialize shape for this iterator
   *
   * We will choose the maximum number of axes from all tensors
   * belonging to this iterator and initialize the shape of this
   * iterator using this maximum number of axes.
   */
  void InitializeShape();

  /**
   * \brief Broadcast shape between tensors of this iterator.
   *
   * The broadcast rule is the same as numpy's. see
   * https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcasting
   *
   * After this function, all shapes of tensors in this iterator will be aligned
   * at their least significant axis and strides will be set to zero for
   * broadcast if the corresponding axis size is 1.
   *
   *   e.g. assume max_num_axes = 6,
   *        [2, 4, 6] => [1, 1, 1, 2, 4, 6]
   */
  void BroadcastShape();

  /**
   * \brief Compress contiguous axes to make iteration more efficient
   *
   * This will compress adjoining contiguous axes into one axis. Two axes
   * can be compressed if these two axes are contiguous on all tensors in
   * this iterator.
   */
  void CompressShape();

  /**
   * \brief Check two axes can be compressed into one axis.
   *
   * This function is invoked by CompressShape()
   *
   * \param dim0 the first axis
   * \param dim1 the second axis, assume dim1 > dim0
   *
   * \note This function usually be invoked during modification
   *       of tensors' strides and shapes.
   */
  bool CanCompress(int dim0, int dim1) const;

  /**
   * \brief Compress two axes into one axis.
   *
   * This function is invoked by CompressShape(), and the second
   * axis will be used as the new axis.
   *
   * \param dim0 the first axis
   * \param dim1 the second axis, assume dim1 > dim0
   *
   * \note This function usually be invoked during modification
   *       of tensors' strides and shapes.
   */
  void CompressAxes(int dim0, int dim1);

  void GetDataPtrs(std::vector<char *> &dptrs, const std::vector<char *> &base,
                   const std::vector<shape_elem_t> &index,
                   const std::vector<size_t> &stride_bytes) const;

private:
  // when tensors are fixed, operands_ will hold all
  // tensors with output tensors first.
  std::vector<Tensor> operands_;

  // temporary output tensor buffer
  std::vector<Tensor> outputs_;

  // temporary input tensor buffer
  std::vector<Tensor> inputs_;

  // has this iterator's IO tensors fixed?
  bool has_tensors_fixed_ = false;

  // has this iterator's shape initialized?
  bool has_shape_initialized_ = false;

  // have shapes of tensors in this iterator been broadcasted?
  bool has_shape_broadcasted_ = false;

  // have shapes of tensors in this iterator been compressed?
  bool has_shape_compressed_ = false;

  // number of input tensors, this is valid only after all tensors are fixed.
  size_t num_inputs_ = 0;

  // number of output tensors, this is valid only after all tensors are fixed.
  size_t num_outputs_ = 0;

  // the shape of this iterator deduced from its inputs/output tensors.
  std::vector<shape_elem_t> shape_;
};

} // namespace tl

#endif // TENSORLITE_TENSOR_OP_TENSOR_ITERATOR_H_
