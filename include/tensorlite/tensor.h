#ifndef TENSORLITE_TENSOR_H_
#define TENSORLITE_TENSOR_H_

#include <array>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unordered_set>
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

  /**
   * \brief Construct a new TensorShape object from initializer_list
   *
   * \tparam IndexTy
   * \param init
   */
  template <typename IndexTy,
            std::enable_if_t<std::is_integral_v<IndexTy>> * = nullptr>
  TensorShape(std::initializer_list<IndexTy> init)
      : TensorShape(std::vector<IndexTy>(init)) {}

  TensorShape() = default;

  TensorShape(const TensorShape &) = default;

  TensorShape(TensorShape &&) = default;

  TensorShape &operator=(const TensorShape &) = default;

  TensorShape &operator=(TensorShape &&) = default;

  ~TensorShape() = default;

  /**
   * \brief Deduce a new tensor shape from a view shape (may have -1 as
   * placeholder) and the number of elements.
   *
   * \tparam IndexTy The type of index in view shape.
   * \param view The input view shape.
   * \param num_elems The total number of elements.
   * \return TensorShape
   */
  template <
      typename IndexTy,
      std::enable_if_t<std::is_integral_v<IndexTy> &&
                       std::numeric_limits<IndexTy>::is_signed> * = nullptr>
  static TensorShape DeduceFromView(const std::vector<IndexTy> &view,
                                    elem_t num_elems) {
    bool find_negative = false;
    IndexTy prod = static_cast<IndexTy>(1);
    std::optional<size_t> deduce_pos = std::nullopt;
    for (auto i = 0; i < view.size(); ++i) {
      if (view[i] < 0) {
        assert(!find_negative);
        find_negative = true;
        deduce_pos = i;
      } else {
        prod *= view[i];
      }
    }

    if (find_negative) {
      assert(static_cast<IndexTy>(num_elems) % prod == 0);
      view[deduce_pos.value()] = static_cast<IndexTy>(num_elems) / prod;
    }
    return TensorShape(view);
  }

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
    assert(new_rank <= kMaxTensorRank);
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
  const elem_t &operator[](int idx) const {
    assert(idx < rank_);
    return shape_[idx];
  }

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

  /**
   * \brief Transpose two axes.
   */
  void Transpose(size_t i, size_t j) {
    assert(i < rank_ && j < rank_);
    std::swap(shape_[i], shape_[j]);
  }

  /**
   * \brief Transpose a tensor shape with a given permutation vector.
   *
   * \param perm The input permutation vector.
   */
  void Transpose(const std::vector<size_t> &perm) {
    assert(IsValidPermutation(perm));
    std::array<elem_t, kMaxTensorRank> shape_copy;
    std::copy_n(shape_.begin(), rank_, shape_copy.begin());
    for (auto i = 0; i < rank_; ++i) {
      shape_[i] = shape_copy[perm[i]];
    }
  }

  /**
   * \brief Check whether a permutation vector is valid.
   *
   * \param perm The permutation vector.
   */
  bool IsValidPermutation(const std::vector<size_t> &perm) {
    std::unordered_set<size_t> axis_appear;
    for (const auto &p : perm) {
      if (p >= rank_ || axis_appear.count(p))
        return false;
      axis_appear.insert(p);
    }
    return true;
  }

protected:
  // this could have negative elements
  std::array<elem_t, kMaxTensorRank> shape_;

  // a.k.a num_dims, the number of dimensions of this tensor
  size_t rank_ = 0;
};

using shape_elem_t = TensorShape::elem_t;

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
   * \brief Transpose two axes
   */
  void Transpose(size_t i, size_t j) {
    TensorShape::Transpose(i, j);
    std::swap(stride_[i], stride_[j]);
  }

  /**
   * \brief Transpose a tensor shape with a given permutation vector.
   *
   * \param perm The input permutation vector.
   */
  void Transpose(const std::vector<size_t> &perm) {
    TensorShape::Transpose(perm);
    std::array<elem_t, kMaxTensorRank> stride_copy;
    std::copy_n(stride_.begin(), rank_, stride_copy.begin());
    for (auto i = 0; i < rank_; ++i) {
      stride_[i] = stride_copy[perm[i]];
    }
  }

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
  // TODO: add default empty tensor here.
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
   * \brief Get the i-th shape element.
   *
   * \param idx index value.
   * \return shape_elem_t&
   */
  shape_elem_t &GetShape(size_t idx) { return shape_.Shape(idx); }

  /**
   * \brief Get the i-th shape element.
   *
   * \param idx index value.
   * \return const shape_elem_t&
   */
  const shape_elem_t &GetShape(size_t idx) const { return shape_.Shape(idx); }

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
   * device. If alignment == 0, use the size of data type as its alignment.
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

  /**
   * \brief Create an all-one tensor with the given shape, ddta type and device.
   *
   * \param shape The shape of this tensor.
   * \param dtype The data type of this tensor.
   * \param device The device that this tensor resides.
   * \return Tensor
   */
  TENSORLITE_DLL static Tensor Ones(TensorShape shape, DataType dtype,
                                    Device device = Device::DefaultDevice());

  /**
   * \brief Create an all-zero tensor with the given shape, data type and
   * device.
   *
   * \param shape The shape of this tensor.
   * \param dtype The data type of this tensor.
   * \param device The device that this tensor resides.
   * \return Tensor
   */
  TENSORLITE_DLL static Tensor Zeros(TensorShape shape, DataType dtype,
                                     Device device = Device::DefaultDevice());

  /**
   * \brief Generate a tensor whose elements follow a given uniform distribution.
   *
   * \param shape The shape of the new tensor.
   * \param low The lower bound of the given uniform distribution.
   * \param high The upper bound of the givne unform distribution.
   * \param dtype The data type of the tensor.
   * \param device The device where the tensor resides.
   * \return Tensor
   *
   * \note Currently we only support floating point data type.
   */
  TENSORLITE_DLL static Tensor
  Uniform(TensorShape shape, Scalar low = Scalar(0), Scalar high = Scalar(1),
          DataType dtype = DataType(DataTypeTag::kFloat64),
          Device device = Device::DefaultDevice());

  /**
   * \brief Generate a tensor whose elements follow a given normal distribution.
   *
   * \param shape The shape of the new tensor.
   * \param mean The mean value of the given normal distribution.
   * \param stddev The standard deviation of the given normal distribution.
   * \param dtype The data type of the tensor.
   * \param device The device where the tensor resides.
   * \return Tensor
   *
   * \note Currently we only support floating point data type.
   */
  TENSORLITE_DLL static Tensor
  Normal(TensorShape shape, Scalar mean = Scalar(0), Scalar stddev = Scalar(1),
         DataType dtype = DataType(DataTypeTag::kFloat64),
         Device device = Device::DefaultDevice());

  /**
   * \brief Create a new tensor with the same shape as the other tensor.
   *
   * \param other The other tensor.
   * \param contiguous If true, the new tensor will be contiguous no metter
   *                   whether the other tensor is contiguous or not.
   * \param dtype The data type of the new tensor. If not given, use the data
   *              type of the other tensor.
   * \param device The device of the new tensor. If not given, use the device of
   *               the other tensor.
   * \return Tensor
   */
  TENSORLITE_DLL static Tensor
  SameAs(const Tensor &other, bool contiguous = true,
         std::optional<DataType> dtype = std::nullopt,
         std::optional<Device> device = std::nullopt);

  /**
   * \brief Get a new tensor with all elements equal to a given value.
   *
   * \tparam DataTy The data type of this tensor in c-type.
   * \param shape The shape of this new tensor.
   * \param val The value of all elements of the new tensor.
   * \param alignment The alignment requirement of the new tensor. If 0, use the
   * size of its data type.
   * \param device The device that the new tensor resides in.
   * \return Tensor
   */
  template <typename DataTy,
            std::enable_if_t<support_crt_v<DataTy>> * = nullptr>
  static Tensor Full(TensorShape shape, DataTy val, size_t alignment = 0,
                     Device device = Device::DefaultDevice()) {
    Tensor new_tensor = Tensor::Empty(shape, DataType(crt_to_dtype_v<DataTy>),
                                      alignment, device);
    new_tensor.Fill(val);
    return new_tensor;
  }

  /**
   * \brief Get a new tensor with all elements equal to a given scalar
   *
   * \param shape The shape of this new tensor.
   * \param val The scalar value of all elements of the new tensor.
   * \param alignment The alignment requirement of the new tensor. If 0, use the
   * size of its data type.
   * \param device The device that the new tensor resides
   * in.
   * \return TENSORLITE_DLL
   */
  TENSORLITE_DLL static Tensor Full(TensorShape shape, Scalar val,
                                    size_t alignment = 0,
                                    Device device = Device::DefaultDevice());

  /**
   * \brief Return a new tensor with contiguous layout.
   * The data type and device are the same as the original tensor.
   *
   * \return Tensor
   */
  TENSORLITE_DLL Tensor Contiguous() const;

  /**
   * \brief Make a copy of an existing tensor.
   *
   * The layout, data type and device information will be preserved.
   *
   * \return Tensor
   */
  TENSORLITE_DLL Tensor Copy() const;

  /**
   * \brief Return a new tensor sharing the same data buffer
   * with the original tensor but with two axes tranposed.
   *
   * \param i The first axis.
   * \param j The second axis.
   * \return Tensor
   */
  Tensor Transpose(size_t i, size_t j) const {
    Tensor new_tensor = *this;
    new_tensor.Transpose_(i, j);
    return new_tensor;
  }

  /**
   * \brief Return a new tensor sharing the same data buffer
   * with the original tensor but with all axes transposed.
   *
   * \param perm The permutation vector.
   * \return Tensor
   */
  Tensor Transpose(const std::vector<size_t> &perm) const {
    Tensor new_tensor = *this;
    new_tensor.Transpose_(perm);
    return new_tensor;
  }

  /**
   * \brief Transpose the tensor inplace but with data buffer unchanged.
   *
   * \param i The first axis.
   * \param j The second axis.
   */
  void Transpose_(size_t i, size_t j) { shape_.Transpose(i, j); }

  /**
   * \brief Transpose the tensor inplace but with data buffer unchanged.
   *
   * \param perm The permutation vector.
   * \return Tensor
   */
  void Transpose_(const std::vector<size_t> &perm) { shape_.Transpose(perm); }

  /**
   * \brief Transfer a tensor to another device
   *
   * \param device The target device
   * \return Tensor
   */
  TENSORLITE_DLL Tensor Transfer(Device device) const;

  /**
   * \brief Create a view with another shape of a tensor.
   *
   * The tensor should be contiguous.
   *
   * \param view_shape The given view shape.
   * \return Tensor
   */
  TENSORLITE_DLL Tensor View(TensorShape view_shape) const;

  /**
   * \brief Create a view with another shape of a tensor.
   *
   * The tensor should be contiguous.
   *
   * \tparam IndexTy The type of elements of the input vector.
   * \param view_shape_vec The shape of view in vector form.
   * \return Tensor
   */
  template <
      typename IndexTy,
      std::enable_if_t<std::is_integral_v<IndexTy> &&
                       std::numeric_limits<IndexTy>::is_signed> * = nullptr>
  Tensor View(const std::vector<IndexTy> &view_shape_vec) {
    TensorShape view_shape =
        TensorShape::DeduceFromView(view_shape_vec, GetNumElems());
    return this->View(view_shape);
  }

  /**
   * \brief Create a new tensor with all elements casted to a new data type.
   *
   * \param dtype The target data type.
   * \return Tensor
   */
  TENSORLITE_DLL Tensor Cast(DataType dtype) const;

  /**
   * \brief Create a new tensor with new shape.
   *
   * If the tensor is contiguous, a view will be returned. If not, a
   * contiguous copy of the original tensor will be returned.
   *
   * \param new_shape
   * \return TENSORLITE_DLL
   */
  TENSORLITE_DLL Tensor Reshape(TensorShape new_shape) const;

  /**
   * \brief Create a new tensor with new shape.
   *
   * If the tensor is contiguous, a view will be returned. If not, a
   * contiguous copy of the original tensor will be returned.
   *
   * \tparam IndexTy The type of elements of the input vector.
   * \param new_shape_vec The new shape in vector form.
   * \return Tensor
   */
  template <
      typename IndexTy,
      std::enable_if_t<std::is_integral_v<IndexTy> &&
                       std::numeric_limits<IndexTy>::is_signed> * = nullptr>
  Tensor Reshape(const std::vector<IndexTy> &new_shape_vec) {
    TensorShape new_shape =
        TensorShape::DeduceFromView(new_shape_vec, GetNumElems());
    return this->Reshape(new_shape);
  }

  /**
   * \brief Fill a tensor with given value.
   *
   * \tparam DataTy The data type of this tensor.
   * \param val The value used for filling.
   */
  template <typename DataTy,
            std::enable_if_t<support_crt_v<DataTy>> * = nullptr>
  void Fill(DataTy val) {
    size_t num_bytes = sizeof(DataTy);
    Tensor::FillInBytes(*this, reinterpret_cast<void *>(&val), num_bytes);
  }

  /**
   * \brief Fill a tensor with a given scalar
   *
   * \param val The scalar usef for filling.
   */
  TENSORLITE_DLL void Fill(Scalar val);

  /**
   * \brief Fill a tensor with a section of bytes.
   *
   * \param t The tensor to be filled.
   * \param val The section of bytes.
   * \param num_bytes The size of bytes.
   * \return Tensor
   */
  TENSORLITE_DLL static Tensor FillInBytes(Tensor &t, void *val,
                                           size_t num_bytes);

private:
  // The data type of this tensor
  DataType dtype_;

  // The shape of this tensor
  TensorShapeWithStride shape_;

  // The underlying data buffer
  BufferPtr buffer_;
};

} // namespace tl

#endif // TENSORLITE_TENSOR_H_
