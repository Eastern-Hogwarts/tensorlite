#include "tensorlite/tensor_ops.h"
#include "tensorlite/dispatch/device_dispatch.h"
#include "tensorlite/utils/logging.h"

namespace tl {

OP_DEF(Add);
Tensor Add(const Tensor &t1, const Tensor &t2) {
  return DeviceDispatchCall<Tensor, Tensor, Tensor>("Add", t1.GetDevice().GetType(), t1, t2);
}

} // namespace tl
