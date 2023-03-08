#ifndef TENSORLITE_PYTHON_API_SHAPE_H_
#define TENSORLITE_PYTHON_API_SHAPE_H_

#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace tl {
namespace pyapi {

class PyTensorShape final {
public:
  PyTensorShape() = default;

  explicit PyTensorShape(const py::tuple& shape_tuple) : contents_(shape_tuple.size()) {
    size_t idx = 0;
    for (const auto& elem : shape_tuple) {
      contents_[idx++] = (
        static_cast<tl::shape_elem_t>(elem.cast<py::int_>())
      );
    }
  }

  explicit PyTensorShape(const py::list& shape_list)
    : PyTensorShape(static_cast<py::tuple>(shape_list)) {}

  size_t Rank() const { return contents_.size(); }

  tl::shape_elem_t operator[](int idx) const {
    if (idx < 0) {
      idx = Rank() + idx;
    }
    if (idx < 0 || idx >= Rank()) {
      throw std::out_of_range("Index out of range");
    }
    return contents_[idx];
  }

  tl::shape_elem_t& operator[](int idx) {
    if (idx < 0) {
      idx = Rank() + idx;
    }
    if (idx < 0 || idx >= Rank()) {
      throw std::out_of_range("Index out of range");
    }
    return contents_[idx];
  }

  const std::vector<tl::shape_elem_t>& to_vector() const {
    return contents_;
  }

  std::vector<tl::shape_elem_t>& to_vector() {
    return contents_;
  }

  std::vector<tl::shape_elem_t>::iterator begin() {
    return contents_.begin();
  }

  std::vector<tl::shape_elem_t>::iterator end() {
    return contents_.end();
  }

  std::string to_string() const {
    std::ostringstream sm;
    sm << "[";
    for (auto i = 0; i < Rank() - 1; ++i) {
      sm << contents_[i] << ", ";
    }
    sm << contents_[Rank() - 1] << "]";
    return sm.str();
  }

private:
  std::vector<tl::shape_elem_t> contents_;
};

} // namespace pyapi
} // namespace tl

// PYBIND11_MAKE_OPAQUE(tl::pyapi::PyTensorShape);

#endif  // TENSORLITE_PYTHON_API_SHAPE_H_
