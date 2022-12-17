#include "tensorlite/allocator/cuda_allocator.h"
#include "tensorlite/allocator/utils.h"
#include "tensorlite/utils/cuda_common.h"
#include "tensorlite/utils/logging.h"
#include <cuda_runtime.h>

namespace tl {

void *CudaMemoryAllocator::Allocate(int device_id, size_t size, size_t align) {
  void *alloc_ptr;
  CUDA_CALL(cudaSetDevice(device_id));
  CHECK_EQ(256 % align, 0U) << "CUDA space is aligned at 256 bytes\n";
  size = utils::ceil_align(size, align);
  CUDA_CALL(cudaMalloc(&alloc_ptr, size));
  return alloc_ptr;
}

void CudaMemoryAllocator::Free(int device_id, void *ptr, size_t align) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaFree(ptr));
}

std::shared_ptr<Buffer> NewCudaBuffer(int device_id, size_t size,
                                      size_t align) {
  size = utils::ceil_align(size, align);
  void *ptr = CudaMemoryAllocator::Allocate(device_id, size, align);
  if (ptr) {
    return std::make_shared<Buffer>(ptr, size, align,
                                    Device(device_id, DeviceType::kCUDA),
                                    &CudaBufferDeleter);
  } else {
    return nullptr;
  }
}

template <>
std::shared_ptr<Buffer> NewBuffer<DeviceType::kCUDA>(int device_id, size_t size,
                                                     size_t align) {
  return NewCudaBuffer(device_id, size, align);
}

} // namespace tl
