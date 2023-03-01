#include "tensorlite/device.h"
#include "tensorlite/dispatch/device_dispatch.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/utils/cpu_tools.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/native_scalar_ops.h"

namespace tl {
namespace {

template <typename DataTy, typename BinaryOp>
void BinaryElementwiseOpKernel(Tensor &out, const Tensor &t1, const Tensor &t2,
                               BinaryOp &&op) {
  TensorIterator iter;
  iter.AddOutput(out).AddInput(t1).AddInput(t2).Build();
  ::tl::cpu::CPUElemwiseKernel(iter, std::forward<BinaryOp>(op));
}

template <typename DataTy, typename Unary>
void UnaryElementwiseOpKernel(Tensor &out, const Tensor &t, Unary &&op) {
  TensorIterator iter;
  iter.AddOutput(out).AddInput(t).Build();
  ::tl::cpu::CPUElemwiseKernel(iter, std::forward<Unary>(op));
}

} // namespace

namespace native_ops {

#define DEFINE_BINARY_INFIX(name, infix_op)                                    \
  Tensor Cpu##name(const Tensor &t1, const Tensor &t2) {                       \
    CHECK_EQ(t1.GetDevice(), t2.GetDevice())                                   \
        << "Operands should reside on the same device";                        \
    CHECK_EQ(t1.GetDataType(), t2.GetDataType())                               \
        << "Operands should have the same data type";                          \
    DataType dtype = t1.GetDataType();                                         \
    auto broadcast_shape =                                                     \
        TensorShape::BroadcastShape(t1.GetShape(), t2.GetShape());             \
    CHECK(broadcast_shape.has_value())                                         \
        << "Cannot broadcast shaped between input tensors, t1 shape: "         \
        << t1.GetShape() << ", t2 shape: " << t2.GetShape();                   \
    Tensor out = Tensor::Empty(                                                \
        TensorShape::BroadcastShape(t1.GetShape(), t2.GetShape()).value(),     \
        t1.GetDataType(), t1.GetAlignment(), t1.GetDevice());                  \
    DTYPE_SWITCH(dtype.GetTag(), [&]() {                                       \
      BinaryElementwiseOpKernel<scalar_t>(                                     \
          out, t1, t2, [](scalar_t a, scalar_t b) { return a infix_op b; });   \
    });                                                                        \
    return out;                                                                \
  }                                                                            \
  OP_IMPL(Native_##name, kCPU, Cpu##name);

DEFINE_BINARY_INFIX(Add, +);
DEFINE_BINARY_INFIX(Sub, -);
DEFINE_BINARY_INFIX(Mul, *);
DEFINE_BINARY_INFIX(Div, /);

#undef DEFINE_BINARY_INFIX
#define CPU_SCALAR_OP(name) ::tl::native_scalar_ops::name<DeviceType::kCPU>()


#define DEFINE_UNARY(name, op)                                                 \
  Tensor Cpu##name(const Tensor &t) {                                          \
    Tensor out = Tensor::SameAs(t);                                            \
    DTYPE_SWITCH(t.GetDataType().GetTag(), [&]() {                             \
      UnaryElementwiseOpKernel<scalar_t>(out, t,                               \
                                         [](scalar_t x) { return op(x); });    \
    });                                                                        \
    return out;                                                                \
  }                                                                            \
  OP_IMPL(Native_##name, kCPU, Cpu##name);

DEFINE_UNARY(Sqrt, sqrt);
DEFINE_UNARY(Neg, -);
DEFINE_UNARY(Abs, CPU_SCALAR_OP(AbsOp));

#undef DEFINE_UNARY

#define DEFINE_UNARY_ONLY_FLOAT(name, op)                                                 \
  Tensor Cpu##name(const Tensor &t) {                                          \
    Tensor out = Tensor::SameAs(t);                                            \
    DTYPE_SWITCH_FLOAT(t.GetDataType().GetTag(), [&]() {                             \
      UnaryElementwiseOpKernel<scalar_t>(out, t,                               \
                                         [](scalar_t x) { return op(x); });    \
    });                                                                        \
    return out;                                                                \
  }                                                                            \
  OP_IMPL(Native_##name, kCPU, Cpu##name);

DEFINE_UNARY_ONLY_FLOAT(Acos, CPU_SCALAR_OP(AcosOp));
DEFINE_UNARY_ONLY_FLOAT(Acosh, CPU_SCALAR_OP(AcoshOp));

#undef DEFINE_UNARY_ONLY_FLOAT
#undef CPU_SCALAR_OP
} // namespace native_ops
} // namespace tl
