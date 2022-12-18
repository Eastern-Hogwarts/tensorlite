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
template <typename DataTy> void CpuFillKernel(Tensor &tensor, DataTy val);

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

/**
 * \brief Perform data type cast between two tensors.
 *
 * \param src The source tensor.
 * \param dst The destination tensor.
 * \note: TODO: change this to template with device_type as template args.
 */
void CpuCastKernel(const Tensor &src, Tensor &dst);

/**
 * \brief Generate tensor elements with an uniform distribution given its lower
 * and upper boundary.
 *
 * \param tensor The target tensor.
 * \param low The lower boundary.
 * \param high The upper boundary.
 */
void CpuUniformDistKernel(Tensor &tensor, Scalar low, Scalar high);

/**
 * \brief Generate tensor elements with a normal distribution given its mean and
 * stddev value.
 *
 * \param tensor The target tensor.
 * \param mean The mean value.
 * \param stddev The stddev value.
 */
void CpuNormalDistKernel(Tensor &tensor, Scalar mean, Scalar stddev);

} // namespace cpu
} // namespace tl
