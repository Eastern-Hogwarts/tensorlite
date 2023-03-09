#include "tensorlite/device.h"

#ifdef ENABLE_CUDA
#include "tensorlite/utils/cuda_common.h"
#include <cuda_runtime.h>
#endif // ENABLE_CUDA

namespace tl {

void Device::SetCurrentDevice() const {
  switch (type_) {
#ifdef ENABLE_CUDA
  case DeviceType::kCUDA:
    CUDA_CALL(cudaSetDevice(id_));
    break;
#endif // ENABLE_CUDA
  default:
    break;
  }
}

} // namespace tl
