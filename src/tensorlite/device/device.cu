#include "tensorlite/device.h"
#include "tensorlite/utils/cuda_common.h"
#include <cuda_runtime.h>

namespace tl {

void Device::SetCurrentDevice() const {
  switch (type_) {
  case DeviceType::kCUDA:
    CUDA_CALL(cudaSetDevice(id_));
    break;
  /*case DeviceType::KCPU*/
  default:
    break;
  }
}

} // namespace tl
