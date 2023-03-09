#include "tensorlite/device/data_transfer.h"

namespace tl {

template <>
void DataTransfer<DeviceType::kCPU, DeviceType::kCPU>(
    const void *src_ptr, void *dst_ptr, size_t size, int /* src_id*/,
    int /*dst_id*/, void * /*stream*/) {
  if (src_ptr == dst_ptr || size == 0) {
    return;
  }
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);
  std::memcpy(dst_ptr, src_ptr, size);
}

#ifdef ENABLE_CUDA

} // namespace tl

#include "tensorlite/utils/cuda_common.h"
#include <cuda_runtime.h>

namespace tl {
template <>
void DataTransfer<DeviceType::kCPU, DeviceType::kCUDA>(const void *src_ptr,
                                                       void *dst_ptr,
                                                       size_t size, int src_id,
                                                       int dst_id,
                                                       void * /*stream*/) {
  if (size == 0) {
    return;
  }
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);

  // TODO: need a better mechanism to exploit async copy and page-lock memory
  CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice));
}

template <>
void DataTransfer<DeviceType::kCUDA, DeviceType::kCPU>(const void *src_ptr,
                                                       void *dst_ptr,
                                                       size_t size, int src_id,
                                                       int dst_id,
                                                       void * /*stream*/) {
  if (size == 0) {
    return;
  }
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);

  // TODO: need a better mechanism to exploit async copy and page-lock memory
  CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
}

template <>
void DataTransfer<DeviceType::kCUDA, DeviceType::kCUDA>(const void *src_ptr,
                                                        void *dst_ptr,
                                                        size_t size, int src_id,
                                                        int dst_id,
                                                        void * /*stream*/) {
  if (size == 0) {
    return;
  }
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);

  // TODO: need a better mechanism to exploit async copy and page-lock memory
  CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
}

#endif // ENABLE_CUDA

} // namespace tl
