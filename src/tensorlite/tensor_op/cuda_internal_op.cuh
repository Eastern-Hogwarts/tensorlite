/**
 * \file cuda_internal_op.cuh
 * \brief Tensor constructors and member functions
 *
 * \note all functions in this header should only have declaration, because
 *       source files include this header may not be compiled by nvcc
 */

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/macros.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/cuda_common.h"
#include "tensorlite/utils/cuda_tools.h"
#include "tensorlite/utils/logging.h"
#include <type_traits>

namespace tl {
namespace cuda {

/**
 * \brief Fill a tensor with the give value.
 *
 * \tparam DataTy The data type of the given tensor.
 * \param tensor The given tensor to be filled.
 * \param val The value used for filling.
 */
template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val);

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
template <typename DataTy> void CudaCopyKernel(const Tensor &src, Tensor &dst);

} // namespace cuda
} // namespace tl
