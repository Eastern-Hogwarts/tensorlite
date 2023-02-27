#ifndef TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
#define TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda/std/cmath>
#include <cuda_fp16.h>
#endif

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/macros.h"

namespace tl {
namespace native_scalar_ops {

#define FORCE_CAST(val, TO_TYPE) *reinterpret_cast<TO_TYPE *>(&val)

template <DeviceType Device> struct SqrtOp {
  template <typename DType> DType operator()(DType val) {
    using ::std::sqrt;
    return sqrt(val);
  }
};

#ifdef __CUDACC__
template <> struct SqrtOp<DeviceType::kCUDA> {
  template <typename DType> TENSOR_DEVICE DType operator()(DType val) {
    return sqrt(val);
  }

  template <> TENSOR_DEVICE tl::fp16_t operator()(tl::fp16_t val) {
    return hsqrt(val);
  }
};
#endif

template <DeviceType Device> struct AbsOp {
  template <typename DType> DType operator()(DType val) {
    using ::std::abs;
    return abs(val);
  }

  template <> uint64_t operator()(uint64_t val) {
    return val;
  }

  template <> uint32_t operator()(uint32_t val) {
    return val;
  }

  template <> uint8_t operator()(uint8_t val) {
    return val;
  }
};

#ifdef __CUDACC__
template <> struct AbsOp<DeviceType::kCUDA> {
  template <typename DType> TENSOR_DEVICE DType operator()(DType val) {
    return abs(val);
  }

  template <> TENSOR_DEVICE uint64_t operator()(uint64_t val) {
    return val;
  }

  template <> TENSOR_DEVICE uint32_t operator()(uint32_t val) {
    return val;
  }

  template <> TENSOR_DEVICE uint8_t operator()(uint8_t val) {
    return val;
  }

  template <> TENSOR_DEVICE tl::fp16_t operator()(tl::fp16_t val) {
    return __habs(val);
  }
};
#endif

} // namespace native_scalar_ops
} // namespace tl

#endif // TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
