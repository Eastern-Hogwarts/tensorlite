#ifndef TENSORLITE_PYTHON_API_TENSORCREATION_H_
#define TENSORLITE_PYTHON_API_TENSORCREATION_H_

#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include <vector>
#include "utils.h"

namespace tl {
namespace pyapi {

tl::Tensor empty(const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
  return tl::Tensor::Empty(tl::pyapi::pyargs_to_vector(shape_args), dtype, 0, device);
}

tl::Tensor ones(const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
  return tl::Tensor::Ones(tl::pyapi::pyargs_to_vector(shape_args), dtype, device);
}

tl::Tensor zeros(const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
  return tl::Tensor::Zeros(tl::pyapi::pyargs_to_vector(shape_args), dtype, device);
}

} // namespace pyapi
} // namespace tl


#endif  // TENSORLITE_PYTHON_API_TENSORCREATION_H_
