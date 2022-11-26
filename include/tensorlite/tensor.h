#ifndef TENSORLITE_TENSOR_H_
#define TENSORLITE_TENSOR_H_

#include <array>
#include <cassert>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "tensorlite/buffer.h"
#include "tensorlite/dtype.h"
#include "tensorlite/macros.h"

// tl for tensorlite
namespace tl {

/**
 * \brief Shape class representing the shape of tensors
 *
 * \note This class should be header-only
 */
class TensorShape {
public:
  // Max rank of tensors
  static constexpr size_t kMaxTensorRank = 16;

  /**
   * \brief Construct a new Tensor Shape object
   */
  TensorShape() {}

  /**
   * \brief Construct a new Tensor Shape object from vector
   *
   * \tparam IndexTy Data type of the given vector object
   * \param shape The given shape
   */
  template <typename IndexTy,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  explicit TensorShape(const std::vector<IndexTy> &shape) {
    assert(shape.size() <= kMaxTensorRank);
    for (auto i = 0; i < shape.size(); ++i) {
      shape_[i] = static_cast<int64_t>(shape[i]);
    }
    rank_ = shape.size();
  }

  TensorShape(const TensorShape &) = default;

  TensorShape(TensorShape &&) = default;

  TensorShape &operator=(const TensorShape &) = default;

  TensorShape &operator=(TensorShape &&) = default;

  /**
   * \brief Dump the shape into a vector of given data type.
   *
   * \tparam IndexTy The data type of the return vector
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy = int64_t>
  std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
  ToVector() const {
    std::vector<IndexTy> ret(rank_);
    for (auto i = 0; i < rank_; ++i) {
      ret[i] = static_cast<IndexTy>(shape_[i]);
    }
    return ret;
  }

  /**
   * \brief Reverse shape array and dump it to a vector with specific dtype
   *
   * \tparam IndexTy The data type of the return vector
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy = int64_t>
  std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
  ReverseShape() const {
    std::vector<IndexTy> ret(rank_);
    for (auto i = 0; i < rank_; ++i) {
      ret[rank_ - i - 1] = shape_[i];
    }
    return ret;
  }

  /**
   * \brief Return a reversed TensorShape object.
   *
   * \return TensorShape
   */
  TensorShape ReverseShape() const {
    TensorShape rshape;
    rshape.rank_ = rank_;
    for (auto i = 0; i < rank_; ++i) {
      rshape.shape_[rank_ - i - 1] = shape_[i];
    }
    return rshape;
  }

  /**
   * \brief Return a pointer pointing to the underlying shape data buffer
   *
   * \return const int64_t*
   */
  const int64_t *RawShapePtr() const { return shape_.data(); }

  /**
   * \brief Return a pointer pointing to the underlying shape data buffer
   *
   * \return int64_t*
   */
  int64_t *RawShapePtr() { return shape_.data(); }

  /**
   * \brief Return the rank of this tensor shape object
   *
   * \return size_t
   */
  size_t Rank() const { return rank_; }

  /**
   * \brief Return total number of elements of this tensor shape object
   *
   * \return int64_t
   */
  int64_t NumElem() const {
    return std::reduce(shape_.begin(), shape_.begin() + rank_, 1LL,
                       [](int64_t s0, int64_t s1) { return s0 * s1; });
  }

  /**
   * \brief Check whether this tensor shape corresponding to a scalar
   *
   * \return true if the shape is corresponding to a scalar
   */
  bool IsScalar() const { return rank_ == 0; }

  /**
   * \brief Return i-th element of this shape object
   *
   * \param idx index value
   * \return const int64_t&
   */
  const int64_t &operator[](int idx) const { return this->operator[](idx); }

  /**
   * \brief Return i-th element of this shape object
   *
   * \param idx index value
   * \return int64_t&
   */
  int64_t &operator[](int idx) {
    assert(idx < rank_);
    return shape_[idx];
  }

  /**
   * \brief Check whether two TensorShape object are equal
   *
   * \param other The other TensorShape object to be compared.
   * \return true if these two TensorShape object are equal,
   * \return false if these two TensorShape object are not equal
   */
  bool operator==(const TensorShape &other) {
    if (rank_ != other.rank_) {
      return false;
    }
    for (auto i = 0; i < rank_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * \brief Check whether two TensorShape object are not equal
   *
   * \param other The other TensorShape object to be compared.
   * \return true if these two TensorShape object are not equal,
   * \return false if these two TensorShape object are equal
   */
  bool operator!=(const TensorShape &other) { return !(*this == other); }

protected:
  // this could have negative elements
  std::array<int64_t, kMaxTensorRank> shape_;

  // a.k.a num_dims, the number of dimensions of this tensor
  size_t rank_ = 0;
};

/**
 * \brief TensorShape with stride
 *
 */
class TensorShapeWithStride : public TensorShape {
public:
  TensorShapeWithStride() {}

  template <typename IndexTy,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  explicit TensorShapeWithStride(const std::vector<IndexTy> &shape,
                                 const std::vector<IndexTy> &stride)
      : TensorShape(shape) {
    assert(shape.size() == stride.size());
    for (auto i = 0; i < stride.size(); ++i) {
      stride_[i] = static_cast<int64_t>(stride[i]);
    }
  }

  TensorShapeWithStride(const TensorShapeWithStride &) = default;

  TensorShapeWithStride(TensorShapeWithStride &&) = default;

  TensorShapeWithStride &operator=(const TensorShapeWithStride &) = default;

  TensorShapeWithStride &operator=(TensorShapeWithStride &&) = default;

  /**
   * \brief Return i-th stride of this shape object
   *
   * \param idx index value
   * \return const int64_t&
   */
  const int64_t &Stride(int idx) const {
    assert(idx < rank_);
    return stride_[idx];
  }

  /**
   * \brief Return i-th stride of this shape object
   *
   * \param idx index value
   * \return int64_t&
   */
  int64_t &Stride(int idx) {
    assert(idx < rank_);
    return stride_[idx];
  }

  /**
   * \brief Reverse stride array and dump it to a vector with specific dtype
   *
   * \tparam IndexTy The data type of the return vector
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy = int64_t>
  std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
  ReverseStride() const {
    std::vector<IndexTy> ret(rank_);
    for (auto i = 0; i < rank_; ++i) {
      ret[rank_ - i - 1] = stride_[i];
    }
    return ret;
  }

  /**
   * \brief Return a reversed TensorShape object.
   *
   * \return TensorShape
   */
  TensorShapeWithStride ReverseShape() const {
    TensorShapeWithStride rshape;
    rshape.rank_ = rank_;
    for (auto i = 0; i < rank_; ++i) {
      rshape.shape_[rank_ - i - 1] = shape_[i];
      rshape.stride_[rank_ - i - 1] = stride_[i];
    }
    return rshape;
  }

  /**
   * \brief Return a pointer pointing to the underlying stride data buffer
   *
   * \return const int64_t*
   */
  const int64_t *RawStridePtr() const { return stride_.data(); }

  /**
   * \brief Return a pointer pointing to the underlying stride data buffer
   *
   * \return int64_t*
   */
  int64_t *RawStridePtr() { return stride_.data(); }

  /**
   * \brief Check whether two TensorShapeWithStride object are equal
   *
   * \param other The other TensorShapeWithStride object to be compared.
   * \return true if these two TensorShapeWithStride object are equal,
   * \return false if these two TensorShapeWithStride object are not equal
   */
  bool operator==(const TensorShapeWithStride &other) {
    if (rank_ != other.rank_) {
      return false;
    }
    for (auto i = 0; i < rank_; ++i) {
      if (shape_[i] != other.shape_[i] || stride_[i] != other.stride_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * \brief Check whether two TensorShapeWithStride object are not equal
   *
   * \param other The other TensorShapeWithStride object to be compared.
   * \return true true if these two TensorShapeWithStride object are not equal,
   * \return false if these two TensorShapeWithStride object are equal
   */
  bool operator!=(const TensorShapeWithStride &other) {
    return !(*this == other);
  }

  /**
   * \brief Check whether this tensor shape is contiguous
   */
  TENSORLITE_DLL bool IsContiguous() const;

  /**
   * \brief Get a contiguous stride from the given shape
   *
   * \tparam IndexTy Index Type
   * \param shape The input tensor shape
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy>
  static std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
  GetContiguousStride(const std::vector<IndexTy> &shape) {
    size_t rank = shape.size();
    std::vector<IndexTy> stride(rank, IndexTy(1));
    for (size_t i = rank - 2; i < rank; --i) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
  }

  /**
   * \brief Get a contiguous tensor shape with stride from a given shape
   *
   * \param shape The input shape
   * \return TensorShapeWithStride
   */
  static TensorShapeWithStride GetContiguousShape(const TensorShape& shape) {
    auto vec_shape = shape.ToVector();
    return TensorShapeWithStride(vec_shape, GetContiguousStride(vec_shape));
  }

protected:
  std::array<int64_t, kMaxTensorRank> stride_;
};

/**
 * \brief Tensor class
 *
 */
class Tensor {
public:
  /**
   * \brief Construct a new Tensor object
   *
   */
  Tensor();

  /**
   * \brief Construct a new Tensor object
   *
   * \param other
   */
  Tensor(const Tensor &other);

  /**
   * \brief Construct a new Tensor object
   *
   * \param other
   */
  Tensor(Tensor &&other);

  /**
   * \brief
   *
   * \param other
   * \return Tensor&
   */
  Tensor &operator=(const Tensor &other);

  /**
   * \brief
   *
   * \param other
   * \return Tensor&
   */
  Tensor &operator=(Tensor &&other);

  /**
   * \brief Destroy the Tensor object
   *
   */
  ~Tensor();

private:
  //
  DataType dtype_;

  //
  TensorShapeWithStride shape_;

  //
  std::shared_ptr<Buffer> buffer_;
};

} // namespace tl

#endif // TENSORLITE_TENSOR_H_
