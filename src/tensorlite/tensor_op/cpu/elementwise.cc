#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/utils/cpu_tools.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/dispatch/device_dispatch.h"

namespace tl {
namespace {

template <typename DataTy, typename BinaryOp>
void BinaryElementwiseOpKernel(Tensor &out, const Tensor &t1, const Tensor &t2,
                               BinaryOp &&op) {
  TensorIterator iter;
  iter.AddOutput(out).AddInput(t1).AddInput(t2).Build();
  ::tl::cpu::CPUElemwiseKernel(iter, std::forward<BinaryOp>(op));
}

} // namespace

Tensor CpuAdd(const Tensor &t1, const Tensor &t2) {
  CHECK_EQ(t1.GetDevice(), t2.GetDevice())
      << "Operands should reside on the same device";
  CHECK_EQ(t1.GetDataType(), t2.GetDataType())
      << "Operands should have the same data type";
  DataType dtype = t1.GetDataType();
  auto broadcast_shape =
      TensorShape::BroadcastShape(t1.GetShape(), t2.GetShape());
  CHECK(broadcast_shape.has_value())
      << "Cannot broadcast shaped between input tensors, t1 shape: "
      << t1.GetShape() << ", t2 shape: " << t2.GetShape();
  Tensor out = Tensor::Empty(
      TensorShape::BroadcastShape(t1.GetShape(), t2.GetShape()).value(),
      t1.GetDataType(), t1.GetAlignment(), t1.GetDevice());

  DTYPE_SWITCH(dtype.GetTag(), [&]() {
    BinaryElementwiseOpKernel<scalar_t>(
        out, t1, t2, [](scalar_t a, scalar_t b) { return a + b; });
  });
  return out;
}

OP_IMPL(Add, kCPU, CpuAdd);

} // namespace tl