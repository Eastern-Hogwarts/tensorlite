#include "tensorlite/allocator/allocator.h"
#include "tensorlite/allocator/cpu_allocator.h"
#include "tensorlite/device/utils.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/cpu_internal_op.h"

#if 1 // TODO: change to ENABLE_CUDA
#include "tensorlite/allocator/cuda_allocator.h"
#include "tensorlite/tensor_op/cuda_internal_op.cuh"
#endif

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
  DEVICE_SWITCH(device.GetType(), {
    buffer_ptr = NewBuffer<device_v>(device.GetId(), buffer_size, alignment);
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
  switch (t.GetDevice().GetType()) {
  case DeviceType::kCPU:
    cpu::CpuFillKernel(t, val);
    break;
  case DeviceType::kCUDA:
    cuda::CudaFillKernel(t, val);
    break;
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

template <typename T>
void TensorElemCopyImpl(const Tensor& src, Tensor& dst) {
  switch (src.GetDevice().GetType()) {
  case DeviceType::kCPU:
    cpu::CpuCopyKernel<T>(src, dst);
    break;
  case DeviceType::kCUDA:
    cuda::CudaCopyKernel<T>(src, dst);
    break;
  default:
    LOG_ERROR << "unknown device type\n";
    break;
  }
}

void TensorElemCopy(const Tensor& src, Tensor& dst, size_t elem_size) {
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

} // namespace tl
