#ifndef TENSORLITE_PYTHON_API_UTILS_H_
#define TENSORLITE_PYTHON_API_UTILS_H_

#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include <string>
#include <vector>

namespace tl {
namespace pyapi {

std::vector<tl::shape_elem_t> pyargs_to_vector(const py::args& shape_args) {
  std::vector<tl::shape_elem_t> shape(shape_args.size());
  size_t idx = 0;
  for (const auto& elem : shape_args) {
    shape[idx++] = static_cast<tl::shape_elem_t>(elem.cast<py::int_>());
  }
  return shape;
}

} // namespace pyapi
} // namespace tl


#endif  // TENSORLITE_PYTHON_API_UTILS_H_
