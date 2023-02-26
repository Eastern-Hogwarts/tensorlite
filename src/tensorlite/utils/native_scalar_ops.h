#ifndef TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
#define TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda/std/cmath>
#endif

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/macros.h"

namespace tl {
namespace native_ops {

#define FORCE_CAST(val, TO_TYPE) *reinterpret_cast<TO_TYPE*>(&val)

template <DeviceType Device>
struct SqrtOp {
  template <typename DType>
  DType operator()(DType val) {
    using ::std::sqrt;
    return sqrt(val);
  }
};

template <>
struct SqrtOp<DeviceType::kCUDA> {
  template <typename DType>
  TENSOR_DEVICE DType operator()(DType val) {
    return sqrt(val);
  }

  template <>
  TENSOR_DEVICE tl::fp16_t operator()(tl::fp16_t val) {
    return hsqrt(val);
  }
};

} // namespace native_ops
} // namespace tl


#endif  // TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
