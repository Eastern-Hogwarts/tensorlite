#include "tensorlite/tensor_op/cuda_internal_op.cuh"

namespace tl {
namespace cuda {

template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val) {
  tensor.GetDevice().SetCurrentDevice();
  DataTy *data = tensor.TypedPtr<DataTy>();
  TensorIterator iter;
  iter.AddOutput(tensor);
  iter.Build();
  CUDAContiguousKernel(iter, [=] CUDA_LAMBDA() { return val; });
}

// switch on data type size
template void CudaFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CudaFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CudaFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CudaFillKernel<uint64_t>(Tensor &t, uint64_t val);

} // namespace cuda
} // namespace tl
