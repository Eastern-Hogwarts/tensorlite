#include "tensorlite/allocator/allocator.h"
#include "tensorlite/allocator/cpu_allocator.h"
#include "tensorlite/device/utils.h"
#include "tensorlite/tensor.h"

#if 1 // TODO: change to ENABLE_CUDA
#include "tensorlite/allocator/cuda_allocator.h"
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

  return Tensor(buffer_ptr, TensorShapeWithStride::GetContiguousShape(shape), dtype);
}

} // namespace tl
