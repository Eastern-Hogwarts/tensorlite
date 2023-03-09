#include "tensorlite/allocator/allocator.h"
#include "tensorlite/allocator/cpu_allocator.h"
#include "tensorlite/device/data_transfer.h"
#include "tensorlite/device/utils.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/cpu_internal_op.h"
#include "tensorlite/tensor_ops.h"

#ifdef ENABLE_CUDA
#include "tensorlite/allocator/cuda_allocator.h"
#include "tensorlite/tensor_op/cuda_internal_op.cuh"
#endif // ENABLE_CUDA

namespace tl {

bool TensorShapeWithStride::IsContiguous() const {
  if (IsScalar())
    return true; // scalar is contiguous
  int64_t s = 1;
  for (size_t i = rank_ - 1; i < rank_; --i) {
    if (stride_[i] == s) {
      s *= shape_[i];
    } else {
      return false;
    }
  }
  return true;
}

Tensor Tensor::Empty(TensorShape shape, DataType dtype, size_t alignment,
                     Device device) {
  size_t buffer_size = static_cast<size_t>(shape.NumElem() * dtype.Size());
  BufferPtr buffer_ptr = nullptr;
  alignment = (alignment == 0) ? dtype.Size() : alignment;
  DEVICE_SWITCH(device.GetType(), device_t, {
    buffer_ptr = NewBuffer<device_t>(device.GetId(), buffer_size, alignment);
  });

  return Tensor(buffer_ptr, TensorShapeWithStride::GetContiguousShape(shape),
                dtype);
}

Tensor Tensor::Ones(TensorShape shape, DataType dtype, Device device) {
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  DTYPE_SWITCH(dtype.GetTag(),
               [&]() { new_tensor.Fill(static_cast<scalar_t>(1)); });
  return new_tensor;
}

Tensor Tensor::Zeros(TensorShape shape, DataType dtype, Device device) {
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  DTYPE_SWITCH(dtype.GetTag(),
               [&]() { new_tensor.Fill(static_cast<scalar_t>(0)); });
  return new_tensor;
}

template <typename T> Tensor FillImpl(Tensor &t, T val) {
  // TODO: change to DEVICE_SWITCH
  switch (t.GetDevice().GetType()) {
  case DeviceType::kCPU:
    cpu::CpuFillKernel(t, val);
    break;
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    cuda::CudaFillKernel(t, val);
    break;
#endif // ENABLE_CUDA
  default:
    LOG_ERROR << "unknown device type\n";
    break;
  }
  return t;
}

Tensor Tensor::FillInBytes(Tensor &t, void *val, size_t num_bytes) {
  switch (num_bytes) {
  case 1:
    FillImpl<uint8_t>(t, *(reinterpret_cast<uint8_t *>(val)));
    break;
  case 2:
    FillImpl<uint16_t>(t, *(reinterpret_cast<uint16_t *>(val)));
    break;
  case 4:
    FillImpl<uint32_t>(t, *(reinterpret_cast<uint32_t *>(val)));
    break;
  case 8:
    FillImpl<uint64_t>(t, *(reinterpret_cast<uint64_t *>(val)));
    break;
  default:
    LOG_ERROR << "Unsupported data type size\n";
    break;
  }
  return t;
}

Tensor Tensor::SameAs(const Tensor &other, bool contiguous,
                      std::optional<DataType> dtype,
                      std::optional<Device> device) {
  TensorShapeWithStride new_shape =
      contiguous ? TensorShapeWithStride::GetContiguousShape(other.GetShape())
                 : other.GetShapeWithStride();
  DataType new_dtype = dtype.value_or(other.GetDataType());
  Device new_device = device.value_or(other.GetDevice());
  return Tensor::Empty(new_shape, new_dtype, other.GetAlignment(), new_device);
}

Tensor Tensor::Full(TensorShape shape, Scalar val, size_t alignment,
                    Device device) {
  // TODO: change to default empty tensor when possible
  std::optional<Tensor> new_tensor;

  // TODO: use std::visit directly? We need a better dispatch method here.
  DTYPE_SWITCH(val.GetDataType().GetTag(), [&]() {
    new_tensor =
        Tensor::Full<scalar_t>(shape, val.To<scalar_t>(), alignment, device);
  });
  return new_tensor.value();
}

void Tensor::Fill(Scalar val) {
  DTYPE_SWITCH(val.GetDataType().GetTag(),
               [&]() { this->Fill<scalar_t>(val.To<scalar_t>()); });
}

template <typename T> void TensorElemCopyImpl(const Tensor &src, Tensor &dst) {
  // TODO: change to DEVICE_SWITCH
  switch (src.GetDevice().GetType()) {
  case DeviceType::kCPU:
    cpu::CpuCopyKernel<T>(src, dst);
    break;
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    cuda::CudaCopyKernel<T>(src, dst);
    break;
#endif // ENABLE_CUDA
  default:
    LOG_ERROR << "unknown device type\n";
    break;
  }
}

void TensorElemCopy(const Tensor &src, Tensor &dst, size_t elem_size) {
  switch (elem_size) {
  case 1:
    TensorElemCopyImpl<uint8_t>(src, dst);
    break;
  case 2:
    TensorElemCopyImpl<uint16_t>(src, dst);
    break;
  case 4:
    TensorElemCopyImpl<uint32_t>(src, dst);
    break;
  case 8:
    TensorElemCopyImpl<uint64_t>(src, dst);
    break;
  default:
    LOG_ERROR << "Unsupported data type size\n";
    break;
  }
}

Tensor Tensor::Contiguous() const {
  if (IsContiguous()) {
    return *this;
  }
  Tensor contiguous_tensor = Tensor::SameAs(*this);
  TensorElemCopy(*this, contiguous_tensor, this->GetDataType().Size());
  return contiguous_tensor;
}

Tensor Tensor::Transfer(Device device) const {
  Tensor new_tensor =
      Tensor::SameAs(*this, IsContiguous(), std::nullopt, device);
  DEVICE_SWITCH(GetDevice().GetType(), src_device_t, {
    DEVICE_SWITCH(device.GetType(), dst_device_t, {
      DataTransfer<src_device_t, dst_device_t>(
          this->RawPtr(), new_tensor.RawPtr(), this->GetBufferSize(),
          this->GetDevice().GetId(), device.GetId());
    });
  });
  return new_tensor;
}

Tensor Tensor::View(TensorShape view_shape) const {
  CHECK(IsContiguous());
  CHECK_EQ(view_shape.NumElem(), GetNumElems());

  Tensor view_tensor = *this;
  view_tensor.shape_ = TensorShapeWithStride::GetContiguousShape(view_shape);
  return view_tensor;
}

Tensor Tensor::Copy() const {
  auto tensor_copy = Tensor::SameAs(*this, IsContiguous());
  DEVICE_SWITCH(GetDevice().GetType(), device_t, {
    DataTransfer<device_t, device_t>(
        this->RawPtr(), tensor_copy.RawPtr(), this->GetBufferSize(),
        this->GetDevice().GetId(), tensor_copy.GetDevice().GetId());
  });
  return tensor_copy;
}

Tensor Tensor::Cast(DataType dtype) const {
  auto cast_tensor = Tensor::SameAs(*this, IsContiguous(), dtype);

  // TODO: use DEVICE_SWITCH here.
  switch (GetDevice().GetType()) {
  case DeviceType::kCPU:
    cpu::CpuCastKernel(*this, cast_tensor);
    break;
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    cuda::CudaCastKernel(*this, cast_tensor);
    break;
#endif // ENABLE_CUDA
  default:
    LOG_ERROR << "unknown device type!\n";
    break;
  }
  return cast_tensor;
}

Tensor Tensor::Reshape(TensorShape new_shape) const {
  if (IsContiguous()) {
    return this->View(new_shape);
  } else {
    return this->Contiguous().View(new_shape);
  }
}

Tensor Tensor::Uniform(TensorShape shape, Scalar low, Scalar high,
                       DataType dtype, Device device) {
  CHECK(dtype.IsFloat());
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  switch (device.GetType()) {
  case DeviceType::kCPU:
    cpu::CpuUniformDistKernel(new_tensor, low, high);
    break;
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    cuda::CudaUniformDistKernel(new_tensor, low, high);
    break;
#endif // ENABLE_CUDA
  default:
    LOG_ERROR << "unknown device type!\n";
    break;
  }
  return new_tensor;
}

Tensor Tensor::Normal(TensorShape shape, Scalar mean, Scalar stddev,
                      DataType dtype, Device device) {
  CHECK(dtype.IsFloat());
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  switch (device.GetType()) {
  case DeviceType::kCPU:
    cpu::CpuNormalDistKernel(new_tensor, mean, stddev);
    break;
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    cuda::CudaNormalDistKernel(new_tensor, mean, stddev);
    break;
#endif // ENABLE_CUDA
  default:
    LOG_ERROR << "unknown device type!\n";
    break;
  }
  return new_tensor;
}

constexpr int kNumberWidth = 10;

template <typename DataTy>
void VectorDisplayImpl(std::ostream &sm, const DataTy *data_ptr, size_t offset,
                       size_t dim_size) {
  sm.setf(std::ios::left, std::ios::adjustfield);

  for (auto i = 0; i < dim_size; ++i) {
    sm.width(kNumberWidth);
    sm << data_ptr[offset + i] << " ";
  }
}

template <typename DataTy>
void TensorDisplayImpl(std::ostream &sm, const Tensor &tensor,
                       std::vector<TensorShape::elem_t> &index, size_t curr_dim,
                       size_t prefix_space_size) {
  sm << "[";
  if (curr_dim == (tensor.Rank() - 1)) {
    VectorDisplayImpl<DataTy>(sm, tensor.TypedPtr<DataTy>(),
                              tensor.GetShapeWithStride().GlobalOffset(index),
                              tensor.GetShape(curr_dim));
    sm << "]";
    return;
  } else {
    size_t curr_dim_size = tensor.GetShape(curr_dim);
    for (auto i = 0; i < curr_dim_size; ++i) {
      size_t curr_prefix_space_size = (i == 0) ? 0 : prefix_space_size + 1;
      sm << std::string(curr_prefix_space_size, ' ');
      index[curr_dim] = i;
      TensorDisplayImpl<DataTy>(sm, tensor, index, curr_dim + 1,
                                prefix_space_size + 1);
      if (i != curr_dim_size - 1) {
        sm << ((curr_dim == tensor.Rank() - 2) ? "\n" : "\n\n");
      }
    }
  }
  index[curr_dim] = 0;
  sm << "]";
}

template <typename DataTy>
void TensorDisplay(std::ostream &sm, const Tensor &tensor) {
  std::vector<TensorShape::elem_t> index(tensor.Rank(), 0);
  TensorDisplayImpl<DataTy>(sm, tensor, index, 0, 0);
  sm << "\n";
}

void Tensor::Display(std::ostream &sm) const {
  sm << *this << "\n";
  Tensor tensor_to_display = *this;
  if (GetDevice().GetType() != DeviceType::kCPU) {
    tensor_to_display = tensor_to_display.Transfer(Device::DefaultDevice());
  }

  DTYPE_SWITCH(this->dtype_.GetTag(),
               [&]() { TensorDisplay<scalar_t>(sm, tensor_to_display); });
}

Tensor Tensor::operator+(const Tensor &other) const {
  return native_ops::Add(*this, other);
}
Tensor Tensor::operator-(const Tensor &other) const {
  return native_ops::Sub(*this, other);
}
Tensor Tensor::operator*(const Tensor &other) const {
  return native_ops::Mul(*this, other);
}
Tensor Tensor::operator/(const Tensor &other) const {
  return native_ops::Div(*this, other);
}
Tensor Tensor::operator-() const { return native_ops::Neg(*this); }

} // namespace tl
