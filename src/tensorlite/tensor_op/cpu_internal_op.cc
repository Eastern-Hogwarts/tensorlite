#include "tensorlite/tensor_op/cpu_internal_op.h"

namespace tl {
namespace cpu {

template <typename DataTy> void CpuCopyKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();
  CPUElemwiseKernel(iter, [](DataTy val) { return val; });
}

} // namespace cpu
} // namespace tl
