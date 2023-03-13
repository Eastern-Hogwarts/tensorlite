#include "tensorlite/device.h"
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

#include "tensorlite/utils/logging.h"

#ifdef ENABLE_CUDA
#include "tensorlite/utils/cuda_common.h"
#include <cuda_runtime.h>
#endif // ENABLE_CUDA

namespace tl {

static void
InitNameToDeviceMap(std::unordered_map<std::string, DeviceType>& map) {
  map["cpu"] = DeviceType::kCPU;
  map["cuda"] = DeviceType::kCUDA;
  map["empty"] = DeviceType::kEmpty;
}

DeviceType DeviceTypeFromName(const std::string& name) {
  static std::unordered_map<std::string, DeviceType> map_;
  std::once_flag init_flag;
  std::call_once(init_flag, InitNameToDeviceMap, map_);

  std::string lower_name(name);
  std::transform(
    lower_name.begin(),
    lower_name.end(),
    lower_name.begin(),
    [](auto c) { return std::tolower(c); }
  );

  if (map_.find(lower_name) == map_.end()) {
    LOG_WARNING << "unknown device name: " << name;
    throw std::invalid_argument("unknown device name: " + name);
  } else {
    return map_.at(lower_name);
  }
}

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
