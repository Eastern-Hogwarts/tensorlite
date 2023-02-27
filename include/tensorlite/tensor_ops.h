#ifndef TENSORLITE_TENSOR_OPS_H_
#define TENSORLITE_TENSOR_OPS_H_

#include "tensorlite/macros.h"
#include "tensorlite/tensor.h"

namespace tl {
namespace native_ops {


/// binary ops
/**
 * \brief Perform elementwise addition
 *
 * \param t1
 * \param t2
 * \return Tensor
 */
TENSORLITE_DLL Tensor Add(const Tensor &t1, const Tensor &t2);

TENSORLITE_DLL Tensor Sub(const Tensor &t1, const Tensor &t2);

TENSORLITE_DLL Tensor Mul(const Tensor &t1, const Tensor &t2);

TENSORLITE_DLL Tensor Div(const Tensor &t1, const Tensor &t2);


/// unary ops

TENSORLITE_DLL Tensor Sqrt(const Tensor &t);

TENSORLITE_DLL Tensor Neg(const Tensor &t);

TENSORLITE_DLL Tensor Abs(const Tensor &t);

TENSORLITE_DLL Tensor Acos(const Tensor &t);

} // namespace native_ops
} // namespace tl

#endif // TENSORLITE_TENSOR_OPS_H_
