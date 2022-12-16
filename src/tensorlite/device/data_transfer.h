/**
 * \file data_transfer.h
 * \brief Manage data transfer between devices
 */
#ifndef TENSORLITE_DEVICE_DATA_TRANSFER_H_
#define TENSORLITE_DEVICE_DATA_TRANSFER_H_

#include "tensorlite/device.h"
#include "tensorlite/utils/logging.h"
#include <cstdint>
#include <cstdlib>

namespace tl {

template <DeviceType SrcDev, DeviceType DstDev>
void DataTransfer(const void *src_ptr, void *dst_ptr, size_t size,
                  int src_id = 0, int dst_id = 0, void *stream = nullptr);

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

#if 1 // TODO: change this to ENABLE_CUDA

#include "tensorlite/utils/cuda_common.h"
#include <cuda_runtime.h>

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

#endif

} // namespace tl

#endif // TENSORLITE_DEVICE_DATA_TRANSFER_H_
