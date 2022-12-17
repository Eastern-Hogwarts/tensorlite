#include "tensorlite/tensor_op/cpu_internal_op.h"

namespace tl {
namespace cpu {

template <typename DataTy> void CpuFillKernel(Tensor &tensor, DataTy val) {
  DataTy *data = tensor.TypedPtr<DataTy>();
  TensorIterator iter;
  iter.AddOutput(tensor);
  iter.Build();
  CPUContiguousKernel(iter, [=]() -> DataTy { return val; });
}

template void CpuFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CpuFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CpuFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CpuFillKernel<uint64_t>(Tensor &t, uint64_t val);

template <typename DataTy> void CpuCopyKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();
  CPUElemwiseKernel(iter, [](DataTy val) { return val; });
}

template void CpuCopyKernel<uint8_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint16_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint32_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint64_t>(const Tensor &src, Tensor &dst);

void CpuCastKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();

  DTYPE_SWITCH(src.GetDataType().GetTag(), [&]() {
    using src_dtype_t = scalar_t;
    DTYPE_SWITCH(dst.GetDataType().GetTag(), [&]() {
      using dst_dtype_t = scalar_t;
      CPUContiguousKernel(iter, [](src_dtype_t val) {
        return dtype_cast<src_dtype_t, dst_dtype_t, DeviceType::kCPU>::cast(
            val);
      });
    });
  });
}

} // namespace cpu
} // namespace tl
