#ifndef TENSORLITE_TENSOR_H_
#define TENSORLITE_TENSOR_H_

#include <array>
#include <cassert>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#include "tensorlite/buffer.h"
#include "tensorlite/device.h"
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
  // type of element of tensor shapes
  using elem_t = int64_t;

  // Max rank of tensors
  static constexpr size_t kMaxTensorRank = 16;

  /**
   * \brief Construct a new Tensor Shape object from vector
   *
   * \tparam IndexTy Data type of the given vector object
   * \param shape The given shape
   */
  template <typename IndexTy,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  TensorShape(const std::vector<IndexTy> &shape) {
    assert(shape.size() <= kMaxTensorRank);
    for (auto i = 0; i < shape.size(); ++i) {
      shape_[i] = static_cast<elem_t>(shape[i]);
    }
    rank_ = shape.size();
  }

  TensorShape() = default;

  TensorShape(const TensorShape &) = default;

  TensorShape(TensorShape &&) = default;

  TensorShape &operator=(const TensorShape &) = default;

  TensorShape &operator=(TensorShape &&) = default;

  ~TensorShape() = default;

  /**
   * \brief Dump the shape into a vector of given data type.
   *
   * \tparam IndexTy The data type of the return vector
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy = elem_t>
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
  template <typename IndexTy = elem_t>
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
   * \return const elem_t*
   */
  const elem_t *RawShapePtr() const { return shape_.data(); }

  /**
   * \brief Return a pointer pointing to the underlying shape data buffer
   *
   * \return elem_t*
   */
  elem_t *RawShapePtr() { return shape_.data(); }

  /**
   * \brief Return the rank (num_of_axes) of this tensor shape object
   *
   * \return size_t
   */
  size_t Rank() const { return rank_; }

  /**
   * \brief Set a new rank for this shape
   *
   * If the new_rank is less than the original one, the extra part will be
   * dropped
   *
   * \param new_rank the value of new rank
   *
   * \note This function is dangerous, use it carefully
   */
  void ResetRank(size_t new_rank) {
    assert(new_rank >= 0 && new_rank <= kMaxTensorRank);
    rank_ = new_rank;
  }

  /**
   * \brief Return total number of elements of this tensor shape object
   *
   * \return elem_t
   */
  elem_t NumElem() const {
    return std::reduce(shape_.begin(), shape_.begin() + rank_, 1LL,
                       [](elem_t s0, elem_t s1) { return s0 * s1; });
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
   * \return const elem_t&
   */
  const elem_t &operator[](int idx) const { return this->operator[](idx); }

  /**
   * \brief Return i-th element of this shape object
   *
   * \param idx index value
   * \return elem_t&
   */
  elem_t &operator[](int idx) {
    assert(idx < rank_);
    return shape_[idx];
  }

  /**
   * \brief Return i-th element of this shape object
   *
   * \param idx index value
   * \return elem_t&
   */
  elem_t &Shape(int idx) { return this->operator[](idx); }

  /**
   * \brief Return i-th element of this shape object
   *
   * \param idx index value
   * \return const elem_t&
   */
  const elem_t &Shape(int idx) const { return this->operator[](idx); }

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
  std::array<elem_t, kMaxTensorRank> shape_;

  // a.k.a num_dims, the number of dimensions of this tensor
  size_t rank_ = 0;
};

/**
 * \brief TensorShape with stride
 *
 * \note The most significant axis is at the beginning of the underlying array
 * (index 0).
 */
class TensorShapeWithStride : public TensorShape {
public:
  template <typename IndexTy,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  explicit TensorShapeWithStride(const std::vector<IndexTy> &shape,
                                 const std::vector<IndexTy> &stride)
      : TensorShape(shape) {
    assert(shape.size() == stride.size());
    for (auto i = 0; i < stride.size(); ++i) {
      stride_[i] = static_cast<elem_t>(stride[i]);
    }
  }

  TensorShapeWithStride() = default;

  TensorShapeWithStride(const TensorShapeWithStride &) = default;

  TensorShapeWithStride(TensorShapeWithStride &&) = default;

  TensorShapeWithStride &operator=(const TensorShapeWithStride &) = default;

  TensorShapeWithStride &operator=(TensorShapeWithStride &&) = default;

  ~TensorShapeWithStride() = default;

  /**
   * \brief Return i-th stride of this shape object
   *
   * \param idx index value
   * \return const elem_t&
   */
  const elem_t &Stride(int idx) const {
    assert(idx < rank_);
    return stride_[idx];
  }

  /**
   * \brief Return i-th stride of this shape object
   *
   * \param idx index value
   * \return elem_t&
   */
  elem_t &Stride(int idx) {
    assert(idx < rank_);
    return stride_[idx];
  }

  /**
   * \brief Reverse stride array and dump it to a vector with specific dtype
   *
   * \tparam IndexTy The data type of the return vector
   * \return std::enable_if_t<std::is_integral_v<IndexTy>, std::vector<IndexTy>>
   */
  template <typename IndexTy = elem_t>
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
   * \return const elem_t*
   */
  const elem_t *RawStridePtr() const { return stride_.data(); }

  /**
   * \brief Return a pointer pointing to the underlying stride data buffer
   *
   * \return elem_t*
   */
  elem_t *RawStridePtr() { return stride_.data(); }

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
  static TensorShapeWithStride GetContiguousShape(const TensorShape &shape) {
    auto vec_shape = shape.ToVector();
    return TensorShapeWithStride(vec_shape, GetContiguousStride(vec_shape));
  }

protected:
  std::array<elem_t, kMaxTensorRank> stride_;
};

/**
 * \brief Tensor class
 *
 */
class Tensor {
  using BufferPtr = std::shared_ptr<Buffer>;

public:
  Tensor() = delete;

  /**
   * \brief Construct a new Tensor object. DO NOT use this directly.
   *
   * \param buffer The underlying data buffer
   * \param shape The shape of this tensor
   * \param dtype The data type of this tensor
   */
  explicit Tensor(BufferPtr buffer, const TensorShapeWithStride &shape,
                  DataType dtype)
      : buffer_(buffer), shape_(shape), dtype_(dtype) {}

  Tensor(const Tensor &other) = default;

  Tensor(Tensor &&other) = default;

  Tensor &operator=(const Tensor &other) {
    auto tmp(other);
    swap(tmp, *this);
    return *this;
  }

  Tensor &operator=(Tensor &&other) {
    auto tmp(std::move(other));
    swap(tmp, *this);
    return *this;
  }

  ~Tensor() = default;

  /**
   * \brief Swap two tensors
   */
  friend void swap(Tensor &t1, Tensor &t2) {
    using std::swap; // enable ADL
    swap(t1.buffer_, t2.buffer_);
    swap(t1.shape_, t2.shape_);
    swap(t1.dtype_, t2.dtype_);
  }

  /**
   * \brief Get the device object of this tensor
   *
   * \return Device
   */
  Device GetDevice() const { return buffer_->GetDevice(); }

  /**
   * \brief Get the data type of this tensor
   *
   * \return DataType
   */
  DataType GetDataType() const { return dtype_; }

  /**
   * \brief Check whether this tensor is contiguous
   */
  bool IsContiguous() const { return shape_.IsContiguous(); }

  /**
   * \brief Check whether this tensor is a scalar
   */
  bool IsScalar() const { return shape_.IsScalar(); }

  /**
   * \brief Get the alignment of this tensor
   *
   * \return size_t
   */
  size_t GetAlignment() const { return buffer_->GetAlignment(); }

  /**
   * \brief Get the size of underlying buffer in bytes
   *
   * \return size_t
   *
   * \note this may be larger than the tensor's size (to meet the align
   * requirement)
   */
  size_t GetBufferSize() const { return buffer_->GetSize(); }

  /**
   * \brief Get the logically size of this tensor
   *
   * \return TensorShape::elem_t
   */
  TensorShape::elem_t TensorSize() const {
    return GetNumElems() * dtype_.Size();
  }

  /**
   * \brief Get the number of elements in this tensor
   *
   * \return TensorShape::elem_t
   */
  TensorShape::elem_t GetNumElems() const { return shape_.NumElem(); }

  /**
   * \brief Get the rank(num_of_axes) of this tensor.
   *
   * \return size_t
   */
  size_t Rank() const { return shape_.Rank(); }

  /**
   * \brief Return a pointer with specific type pointing to the underlying
   * buffer
   *
   * \tparam DataTy The data type of the return pointer.
   * \return DataTy*
   */
  template <typename DataTy> DataTy *TypedPtr() {
    return buffer_->TypedPtr<DataTy>();
  }

  /**
   * \brief Return a pointer with specific type pointing to the underlying
   * buffer
   *
   * \tparam DataTy The data type of the return pointer.
   * \return const DataTy*
   */
  template <typename DataTy> const DataTy *TypedPtr() const {
    return buffer_->TypedPtr<DataTy>();
  }

  /**
   * \brief Return a pointer pointing to the underlying buffer
   *
   * \return void*
   */
  void *RawPtr() { return buffer_->UntypedData(); }

  /**
   * \brief Return a pointer pointing to the underlying buffer
   *
   * \return const void*
   */
  const void *RawPtr() const { return buffer_->UntypedData(); }

  /**
   * \brief Get tensor shape
   *
   * \return TensorShape&
   */
  TensorShape &GetShape() { return shape_; }

  /**
   * \brief Get tensor shape
   *
   * \return const TensorShape&
   */
  const TensorShape &GetShape() const { return shape_; }

  /**
   * \brief Get tensor shape with strides
   *
   * \return TensorShapeWithStride&
   */
  TensorShapeWithStride &GetShapeWithStride() { return shape_; }

  /**
   * \brief Get tensor shape with strides
   *
   * \return const TensorShapeWithStride&
   */
  const TensorShapeWithStride &GetShapeWithStride() const { return shape_; }

  /**
   * \brief Create an empty tensor with given shape, dtype, alignment and
   * device. If alignment == 0, use the size of data type as its alignment
   *
   * \tparam IndexTy Index type of shape vector.
   * \param shape Shape of this tensor
   * \param dtype Data type of this tensor
   * \param alignment Alignment requirement of this tensor
   * \param device Device that this tensor resides in.
   * \return Tensor
   */
  template <typename IndexTy = TensorShape::elem_t,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  static Tensor Empty(const std::vector<IndexTy> &shape, DataType dtype,
                      size_t alignment = 0,
                      Device device = Device::DefaultDevice()) {
    return Tensor::Empty(TensorShape(shape), dtype, alignment, device);
  }

  /**
   * \brief Create an empty tensor with given shape, dtype, alignment and
   * device. If alignment == 0, use the size of data type as its alignment
   *
   * \param shape Shape of this tensor
   * \param dtype Data type of this tensor
   * \param alignment Alignment requirement of this tensor
   * \param device Device that this tensor resides in.
   * \return Tensor
   */
  TENSORLITE_DLL static Tensor Empty(TensorShape shape, DataType dtype,
                                     size_t alignment = 0,
                                     Device device = Device::DefaultDevice());

  TENSORLITE_DLL static Tensor Ones(TensorShape shape, DataType dtype,
                                    Device device = Device::DefaultDevice());

  TENSORLITE_DLL static Tensor Zeros(TensorShape shape, DataType dtype,
                                     Device device = Device::DefaultDevice());

  // Uniform
  TENSORLITE_DLL static Tensor
  Uniform(TensorShape shape, Scalar low = Scalar(0), Scalar high = Scalar(1),
          DataType dtype = DataType(DataTypeTag::kFloat64),
          Device device = Device::DefaultDevice());

  // Normal
  TENSORLITE_DLL static Tensor
  Normal(TensorShape shape, Scalar mean = Scalar(0), Scalar stddev = Scalar(1),
         DataType dtype = DataType(DataTypeTag::kFloat64),
         Device device = Device::DefaultDevice());

  // SameAs
  TENSORLITE_DLL static Tensor
  SameAs(const Tensor &other, bool contiguous = true,
         std::optional<DataType> dtype = std::nullopt,
         std::optional<Device> device = std::nullopt);

  // Full
  template <typename DataTy,
            std::enable_if_t<support_crt_v<DataTy>> * = nullptr>
  static Tensor Full(TensorShape shape, DataTy val, size_t alignment = 0,
                     Device device = Device::DefaultDevice()) {
    Tensor new_tensor =
        Tensor::Empty(shape, crt_to_dtype_v<DataTy>, alignment, device);
    return new_tensor.Fill(val);
  }

  TENSORLITE_DLL static Tensor Full(TensorShape shape, Scalar val,
                                    size_t alignment = 0,
                                    Device device = Device::DefaultDevice());

  // Contiguous
  TENSORLITE_DLL Tensor Contiguous() const;

  // Copy
  TENSORLITE_DLL Tensor Copy() const;

  // Transpose
  TENSORLITE_DLL Tensor Transpose(size_t i, size_t j) const;
  Tensor Transpose(const std::vector<size_t> &perm) const;
  TENSORLITE_DLL Tensor Transpose_(size_t i, size_t j);
  Tensor Transpose_(const std::vector<size_t> &perm_);

  // Transfer
  TENSORLITE_DLL Tensor Transfer(Device device) const;

  // View
  TENSORLITE_DLL Tensor View(TensorShape view_shape) const;

  // Cast
  TENSORLITE_DLL Tensor Cast(DataType dtype) const;

  // Reshape
  TENSORLITE_DLL Tensor Reshape(TensorShape new_shape) const;

  // Fill
  template <typename DataTy,
            std::enable_if_t<support_crt_v<DataTy>> * = nullptr>
  void Fill(DataTy val) {
    size_t num_bytes = sizeof(DataTy);
    return Tensor::FillInBytes(*this, reinterpret_cast<void *>(&val),
                               num_bytes);
  }

  TENSORLITE_DLL void Fill(Scalar val);

  // FillInBytes
  TENSORLITE_DLL static Tensor FillInBytes(Tensor &t, void *val,
                                           size_t num_bytes);

private:
  //
  DataType dtype_;

  //
  TensorShapeWithStride shape_;

  //
  BufferPtr buffer_;
};

using shape_elem_t = TensorShape::elem_t;

} // namespace tl

#endif // TENSORLITE_TENSOR_H_
