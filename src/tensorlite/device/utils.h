#ifndef TENSORLITE_DEVICE_UTILS_H_
#define TENSORLITE_DEVICE_UTILS_H_

#include "tensorlite/device.h"

namespace tl {

#define DEVICE_SWITCH_CASE(switch_case, ...)                                   \
  case switch_case: {                                                          \
    constexpr auto device_v = switch_case;                                     \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }

#define DEVICE_SWITCH(device, ...)                                             \
  {                                                                            \
    ::tl::DeviceType _sv = ::tl::DeviceType(device);                           \
    switch (_sv) {                                                             \
      DEVICE_SWITCH_CASE(::tl::DeviceType::kCPU, __VA_ARGS__)                  \
      DEVICE_SWITCH_CASE(::tl::DeviceType::kCUDA, __VA_ARGS__)                 \
    }                                                                          \
  }

} // namespace tl

#endif // TENSORLITE_DEVICE_UTILS_H_
