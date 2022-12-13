#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/cpu_tools.h"
#include "tensorlite/utils/logging.h"

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

} // namespace cpu
} // namespace tl
