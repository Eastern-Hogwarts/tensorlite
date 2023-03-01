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

template <DeviceType Device> struct AcosOp {
  template <typename DType> DType operator()(DType val) {
    using ::std::acos;
    return acos(val);
  }
};

template <DeviceType Device> struct AcoshOp {
  template <typename DType> DType operator()(DType val) {
    using ::std::acosh;
    return acosh(val);
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

template <> struct AcosOp<DeviceType::kCUDA> {
  template <typename DType> TENSOR_DEVICE DType operator()(DType val) {
    return acos(val);
  }
};

template <> struct AcoshOp<DeviceType::kCUDA> {
  template <typename DType> TENSOR_DEVICE DType operator()(DType val) {
    return acosh(val);
  }

  template <> TENSOR_DEVICE float operator()(float val) {
    return acoshf(val);
  }
};
#endif

} // namespace native_scalar_ops
} // namespace tl

#endif // TENSORLITE_UTILS_NATIVE_SCALAR_OPS_H_
