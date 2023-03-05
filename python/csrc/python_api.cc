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

  ///
  /// Data types
  /// Python manages init return value
  ///
  py::class_<tl::DataType> dtype_class(m, "DataType");

  py::enum_<tl::DataTypeTag>(dtype_class, "DataTypeTag")
      .value("int8", tl::DataTypeTag::kInt8)
      .value("int32", tl::DataTypeTag::kInt32)
      .value("int64", tl::DataTypeTag::kInt64)
      .value("uint8", tl::DataTypeTag::kUInt8)
      .value("uint32", tl::DataTypeTag::kUInt32)
      .value("uint64", tl::DataTypeTag::kUInt64)
      .value("float16", tl::DataTypeTag::kFloat16)
      .value("float32", tl::DataTypeTag::kFloat32)
      .value("float64", tl::DataTypeTag::kFloat64)
      .value("bool", tl::DataTypeTag::kBool)
      .value("invalid", tl::DataTypeTag::kInvalid)
      .export_values();

  dtype_class.def(py::init<const std::string &>(), py::arg("dtype_name"))
      .def(py::init<tl::DataTypeTag>(), py::arg("dtype_tag"))
      .def("__repr__",
           [](const tl::DataType &dtype) {
             return "<tensorlite.DataType: " +
                    static_cast<std::string>(dtype.Name()) + ">";
           })
      .def("__eq__",
           [](const tl::DataType &dtype1, const tl::DataType &dtype2) {
             return dtype1 == dtype2;
           })
      .def_property_readonly("size", &tl::DataType::Size)
      .def_property_readonly("alignment", &tl::DataType::Alignment)
      .def_property_readonly("name", &tl::DataType::Name)
      .def_property_readonly("tag", &tl::DataType::GetTag)
      .def("is_float", &tl::DataType::IsFloat)
      .def("is_integral", &tl::DataType::IsIntegral);

  ///
  /// TODO: Scalar?
  ///

  ///
  /// Device
  /// Python manages init return value
  ///
  py::class_<tl::Device> device_class(m, "Device");

  py::enum_<tl::DeviceType>(device_class, "DeviceType")
      .value("cpy", tl::DeviceType::kCPU)
      .value("cuda", tl::DeviceType::kCUDA)
      .value("empty", tl::DeviceType::kEmpty)
      .export_values();

  device_class.def(py::init<int, tl::DeviceType>(), py::arg("id"), py::arg("device_type"))
    .def_static("cpu_device", &tl::Device::CpuDevice, py::arg("id"))
    .def_static("cuda_device", &tl::Device::CudaDevice, py::arg("id"))
    .def_static("default_device", &tl::Device::DefaultDevice)
    .def_static("static_device", &tl::Device::EmptyDevice)
    .def("__repr__", [](const tl::Device &device) {
      return "<tensorlite.Device: " + device.Name() + ">";
    })
    .def("__eq__", [](const tl::Device &d1, const tl::Device &d2) {
      return d1 == d2;
    })
    .def_property_readonly("name", &tl::Device::Name)
    .def_property_readonly("id", &tl::Device::GetId)
    .def_property_readonly("type", &tl::Device::GetType)
    .def("is_empty", &tl::Device::IsEmpty)
    .def("as_current_device", &tl::Device::SetCurrentDevice);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else  // VERSION_INFO
  m.attr("__version__") = "dev";
#endif // VERSION_INFO
}
