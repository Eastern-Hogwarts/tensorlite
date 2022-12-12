/**
 * \file cuda_internal_op.cuh
 * \brief Tensor constructors and member functions
 */

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/cuda_common.h"
#include "tensorlite/utils/cuda_tools.cuh"
#include "tensorlite/utils/logging.h"
#include <type_traits>

namespace tl {
namespace cuda {

template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val) {
  tensor.GetDevice().SetCurrentDevice();
  DataTy *data = tensor.TypedPtr<DataTy>();
  constexpr size_t unroll = sizeof(DataTy) >= 4 ? 2 : 4;
  TensorIterator iter;
  iter.AddOutput(tensor);
  iter.Build();
  CUDAContiguousKernel(iter, []() -> DataTy { return val; });
}

// switch on data type size
template void CudaFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CudaFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CudaFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CudaFillKernel<uint64_t>(Tensor &t, uint64_t val);

void CopyKernel(const Tensor &src, Tensor &dst);

} // namespace cuda
} // namespace tl
