#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/cpu_tools.h"
#include "tensorlite/utils/logging.h"

namespace tl {
namespace cpu {

/**
 * \brief Fill a tensor with the give value.
 *
 * \tparam DataTy The data type of the given tensor.
 * \param tensor The given tensor to be filled.
 * \param val The value used for filling.
 */
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

/**
 * \brief This is an elementwise copy kernel where the source and dstination
 * tensors should have the same device and data type. However, these two tensors
 * may have different layouts (e.g. one is contiguous and the other is not
 * contiguous).
 *
 * \tparam DataTy The data type of tensors.
 * \param src Source tensor.
 * \param dst Dstination tensor.
 *
 * \note DO NOT use this kernel directly, this kernel usuall is used to
 * implement other tensor functions like contiguous.
 */
template <typename DataTy> void CpuCopyKernel(const Tensor &src, Tensor &dst);

template void CpuCopyKernel<uint8_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint16_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint32_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint64_t>(const Tensor &src, Tensor &dst);

} // namespace cpu
} // namespace tl
