#include "tensorlite/tensor_op/cuda_internal_op.cuh"

#include "tensorlite/device/data_transfer.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/cuda_tools.h"

namespace tl {
namespace cuda {

template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val) {
  tensor.GetDevice().SetCurrentDevice();
  TensorIterator iter;
  iter.AddOutput(tensor).Build();
  CudaContiguousKernel(iter, [=] CUDA_LAMBDA() { return val; });
}

// switch on data type size
template void CudaFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CudaFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CudaFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CudaFillKernel<uint64_t>(Tensor &t, uint64_t val);

template <typename DataTy> void CudaCopyKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  src.GetDevice().SetCurrentDevice();
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();
  CudaElemwiseKernel(iter, [] CUDA_LAMBDA(DataTy val) -> DataTy { return val; });
}

template void CudaCopyKernel<uint8_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint16_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint32_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint64_t>(const Tensor &src, Tensor &dst);

} // namespace cuda
} // namespace tl
