#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(pytensorlite, m) {
  m.doc() = R"pbdoc(
    Tensorlite: a light-weight tensor library
    -----------------------------------------

    .. currentmodule:: tensorlite

    .. autosummary::
      :toctree: __generate
  )pbdoc";

  py::class_<tl::DataType>(m, "dtype")
      .def(py::init<const std::string &>())
      .def("__repr__", [](const tl::DataType &dtype) {
        return "<tensorlite.dtype " + static_cast<std::string>(dtype.Name()) +
               ">";
      });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else  // VERSION_INFO
  m.attr("__version__") = "dev";
#endif // VERSION_INFO
}
