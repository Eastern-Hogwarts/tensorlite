#include "tensorlite/device.h"
#include "tensorlite/dispatch/device_dispatch.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/utils/cuda_tools.h"
#include "tensorlite/utils/logging.h"

namespace tl {
namespace {

template <typename DataTy, typename BinaryOp>
void BinaryElementwiseOpKernel(Tensor &out, const Tensor &t1, const Tensor &t2,
                               BinaryOp &&op) {
  TensorIterator iter;
  iter.AddOutput(out).AddInput(t1).AddInput(t2).Build();
  ::tl::cuda::CudaElemwiseKernel<BinaryOp>(iter,
                                                 std::forward<BinaryOp>(op));
}

} // namespace

#define DEFINE_BINARY_INFIX(name, infix_op)                                    \
  Tensor Cuda##name(const Tensor &t1, const Tensor &t2) {                      \
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
    DTYPE_SWITCH(dtype.GetTag(), [&]() {                             \
      BinaryElementwiseOpKernel<scalar_t>(                                     \
          out, t1, t2, [] CUDA_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {     \
            return a infix_op b;                                               \
          });                                                                  \
    });                                                                        \
    return out;                                                                \
  }                                                                            \
  OP_IMPL(Native_##name, kCUDA, Cuda##name);

DEFINE_BINARY_INFIX(Add, +);
DEFINE_BINARY_INFIX(Sub, -);
DEFINE_BINARY_INFIX(Mul, *);
DEFINE_BINARY_INFIX(Div, /);

#undef DEFINE_BINARY_INFIX

} // namespace tl
