#include "tensorlite/dispatch/device_dispatch.h"
#include "tensorlite/tensor_ops.h"
#include "tensorlite/utils/logging.h"

namespace tl {
namespace native_ops {

/// Binary ops
OP_DEF(Native_Add);
Tensor Add(const Tensor &t1, const Tensor &t2) {
  return DeviceDispatchCall<Tensor, Tensor, Tensor>(
      "Native_Add", t1.GetDevice().GetType(), t1, t2);
}
OP_DEF(Native_Sub);
Tensor Sub(const Tensor &t1, const Tensor &t2) {
  return DeviceDispatchCall<Tensor, Tensor, Tensor>(
      "Native_Sub", t1.GetDevice().GetType(), t1, t2);
}
OP_DEF(Native_Mul);
Tensor Mul(const Tensor &t1, const Tensor &t2) {
  return DeviceDispatchCall<Tensor, Tensor, Tensor>(
      "Native_Mul", t1.GetDevice().GetType(), t1, t2);
}
OP_DEF(Native_Div);
Tensor Div(const Tensor &t1, const Tensor &t2) {
  return DeviceDispatchCall<Tensor, Tensor, Tensor>(
      "Native_Div", t1.GetDevice().GetType(), t1, t2);
}

/// Unary ops
OP_DEF(Native_Sqrt);
Tensor Sqrt(const Tensor &t) {
  return DeviceDispatchCall<Tensor, Tensor>("Native_Sqrt",
                                            t.GetDevice().GetType(), t);
}

OP_DEF(Native_Neg);
Tensor Neg(const Tensor &t) {
  return DeviceDispatchCall<Tensor, Tensor>("Native_Neg", t.GetDevice().GetType(), t);
}

OP_DEF(Native_Abs);
Tensor Abs(const Tensor &t) {
  return DeviceDispatchCall<Tensor, Tensor>("Native_Abs", t.GetDevice().GetType(), t);
}

} // namespace native_ops
} // namespace tl
