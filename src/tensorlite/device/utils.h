#ifndef TENSORLITE_DEVICE_UTILS_H_
#define TENSORLITE_DEVICE_UTILS_H_

#include "tensorlite/device.h"
#include "tensorlite/macros.h"
#include "tensorlite/utils/logging.h"

namespace tl {

#define DEVICE_SWITCH_CASE(switch_case, device_type_name, ...)                 \
  case switch_case: {                                                          \
    constexpr auto device_type_name = switch_case;                             \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }

#define DEVICE_SWITCH(device, device_type_name, ...)                           \
  {                                                                            \
    ::tl::DeviceType _sv = ::tl::DeviceType(device);                           \
    switch (_sv) {                                                             \
      DEVICE_SWITCH_CASE(::tl::DeviceType::kCPU, device_type_name,             \
                         __VA_ARGS__)                                          \
      CUDA_MACRO_OPT(DEVICE_SWITCH_CASE(::tl::DeviceType::kCUDA,               \
                                        device_type_name, __VA_ARGS__))        \
    default:                                                                   \
      LOG_ERROR << "unknown device type\n";                                    \
      break;                                                                   \
    }                                                                          \
  }

} // namespace tl

#endif // TENSORLITE_DEVICE_UTILS_H_
