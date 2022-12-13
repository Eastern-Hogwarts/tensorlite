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

template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val);

void CopyKernel(const Tensor &src, Tensor &dst);

} // namespace cuda
} // namespace tl
