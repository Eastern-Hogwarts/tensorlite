#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include <string>
#include <vector>
#include <sstream>

#include "shape.h"
#include "tensor_creation.h"

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(pytensorlite_capi, m) {
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
      .value("cpu", tl::DeviceType::kCPU)
      .value("cuda", tl::DeviceType::kCUDA)
      .value("empty", tl::DeviceType::kEmpty)
      .export_values();

  device_class.def(py::init<int, tl::DeviceType>(), py::arg("id"), py::arg("device_type"))
    .def(py::init<const std::string&>(), py::arg("device_name"))
    .def_static("cpu_device", &tl::Device::CpuDevice, py::arg("id"))
    .def_static("cuda_device", &tl::Device::CudaDevice, py::arg("id"))
    .def_static("default_device", &tl::Device::DefaultDevice)
    .def_static("empty_device", &tl::Device::EmptyDevice)
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


    ///
    /// TensorShape
    ///
    py::class_<tl::pyapi::PyTensorShape> shape_class(m, "Shape");
    shape_class.def(py::init<const py::tuple&>())
      .def(py::init<const py::list&>())
      .def("__len__", [](const tl::pyapi::PyTensorShape& shape) { return shape.Rank(); })
      .def("__getitem__", static_cast<tl::shape_elem_t&(tl::pyapi::PyTensorShape::*)(int)>(&tl::pyapi::PyTensorShape::operator[]))
      .def("__setitem__", [](tl::pyapi::PyTensorShape& shape, int idx, tl::shape_elem_t newvalue) {
        shape[idx] = newvalue;
      })
      .def("__repr__", [](const tl::pyapi::PyTensorShape& shape) {
        return "<tensorlite.Shape: " + shape.to_string() + ">";
      })
      .def("__iter__", [](tl::pyapi::PyTensorShape& shape) {
        return py::make_iterator(shape.begin(), shape.end());
      }, py::keep_alive<0, 1>()); /* Keep this alive while iterator is used */


    ///
    /// Tensor
    ///
    py::class_<tl::Tensor> tensor_class(m, "Tensor");
    tensor_class.def_property_readonly("dtype", &tl::Tensor::GetDataType)
    .def_property_readonly("device", &tl::Tensor::GetDevice)
    .def_property_readonly("shape", [](const tl::Tensor& tensor) { return tl::pyapi::PyTensorShape(tensor.GetShape().ToVector()); })
    .def("__repr__", [](const tl::Tensor& tensor) {
      std::ostringstream sm;
      sm << tensor;
      return sm.str();
    })
    .def("__str__", [](const tl::Tensor& tensor) {
      std::ostringstream sm;
      tensor.Display(sm);
      return sm.str();
    });

    m.def("empty", [](const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::empty<tl::DataType, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("empty", [](const py::args& shape_args, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::empty<std::string, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("empty", [](const py::args& shape_args, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::empty<tl::DataType, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("empty", [](const py::args& shape_args, const std::string& dtype, const std::string& device) {
      return tl::pyapi::empty<std::string, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());

    m.def("ones", [](const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::ones<tl::DataType, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("ones", [](const py::args& shape_args, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::ones<std::string, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("ones", [](const py::args& shape_args, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::ones<tl::DataType, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("ones", [](const py::args& shape_args, const std::string& dtype, const std::string& device) {
      return tl::pyapi::ones<std::string, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());

    m.def("zeros", [](const py::args& shape_args, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::zeros<tl::DataType, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("zeros", [](const py::args& shape_args, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::zeros<std::string, tl::Device>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("zeros", [](const py::args& shape_args, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::zeros<tl::DataType, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("zeros", [](const py::args& shape_args, const std::string& dtype, const std::string& device) {
      return tl::pyapi::zeros<std::string, std::string>(shape_args, dtype, device);
    }, py::kw_only(), py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());

    m.def("uniform", [](const py::args& shape_args, double low, double high, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::uniform<tl::DataType, tl::Device>(shape_args, low, high, dtype, device);
    }, py::kw_only(), py::arg("low") = 0., py::arg("high") = 1., py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("uniform", [](const py::args& shape_args, double low, double high, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::uniform<std::string, tl::Device>(shape_args, low, high, dtype, device);
    }, py::kw_only(), py::arg("low") = 0., py::arg("high") = 1., py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("uniform", [](const py::args& shape_args, double low, double high, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::uniform<tl::DataType, std::string>(shape_args, low, high, dtype, device);
    }, py::kw_only(), py::arg("low") = 0., py::arg("high") = 1., py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("uniform", [](const py::args& shape_args, double low, double high, const std::string& dtype, const std::string& device) {
      return tl::pyapi::uniform<std::string, std::string>(shape_args, low, high, dtype, device);
    }, py::kw_only(), py::arg("low") = 0., py::arg("high") = 1., py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());

    m.def("normal", [](const py::args& shape_args, double mean, double stddev, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::normal<tl::DataType, tl::Device>(shape_args, mean, stddev, dtype, device);
    }, py::kw_only(), py::arg("mean") = 0., py::arg("stddev") = 1., py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("normal", [](const py::args& shape_args, double mean, double stddev, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::normal<std::string, tl::Device>(shape_args, mean, stddev, dtype, device);
    }, py::kw_only(), py::arg("mean") = 0., py::arg("stddev") = 1., py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("normal", [](const py::args& shape_args, double mean, double stddev, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::normal<tl::DataType, std::string>(shape_args, mean, stddev, dtype, device);
    }, py::kw_only(), py::arg("mean") = 0., py::arg("stddev") = 1., py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("normal", [](const py::args& shape_args, double mean, double stddev, const std::string& dtype, const std::string& device) {
      return tl::pyapi::normal<std::string, std::string>(shape_args, mean, stddev, dtype, device);
    }, py::kw_only(), py::arg("mean") = 0., py::arg("stddev") = 1., py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());


    m.def("same_as", [](const tl::Tensor& other, const tl::DataType* dtype, const tl::Device* device) {
      return tl::pyapi::same_as<tl::DataType, tl::Device>(other, dtype, device);
    }, py::arg("other"), py::arg("dtype").none(true) = nullptr, py::arg("device").none(true) = nullptr);
    m.def("same_as", [](const tl::Tensor& other, const std::string* dtype, const tl::Device* device) {
      return tl::pyapi::same_as<std::string, tl::Device>(other, dtype, device);
    }, py::arg("other"), py::arg("dtype").none(true) = nullptr, py::arg("device").none(true) = nullptr);
    m.def("same_as", [](const tl::Tensor& other, const tl::DataType* dtype, const std::string* device) {
      return tl::pyapi::same_as<tl::DataType, std::string>(other, dtype, device);
    }, py::arg("other"), py::arg("dtype").none(true) = nullptr, py::arg("device").none(true) = nullptr);
    m.def("same_as", [](const tl::Tensor& other, const std::string* dtype, const std::string* device) {
      return tl::pyapi::same_as<std::string, std::string>(other, dtype, device);
    }, py::arg("other"), py::arg("dtype").none(true) = nullptr, py::arg("device").none(true) = nullptr);

    m.def("full", [](const py::args& shape_args, double val, const tl::DataType& dtype, const tl::Device& device) {
      return tl::pyapi::full<tl::DataType, tl::Device>(shape_args, val, dtype, device);
    }, py::kw_only(), py::arg("val") = 0, py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice());
    m.def("full", [](const py::args& shape_args, double val, const std::string& dtype, const tl::Device& device) {
      return tl::pyapi::full<std::string, tl::Device>(shape_args, val, dtype, device);
    }, py::kw_only(), py::arg("val") = 0, py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice());
    m.def("full", [](const py::args& shape_args, double val, const tl::DataType& dtype, const std::string& device) {
      return tl::pyapi::full<tl::DataType, std::string>(shape_args, val, dtype, device);
    }, py::kw_only(), py::arg("val") = 0, py::arg("dtype") = tl::DataType("double"), py::arg("device") = tl::Device::DefaultDevice().Name());
    m.def("full", [](const py::args& shape_args, double val, const std::string& dtype, const std::string& device) {
      return tl::pyapi::full<std::string, std::string>(shape_args, val, dtype, device);
    }, py::kw_only(), py::arg("val") = 0, py::arg("dtype") = "double", py::arg("device") = tl::Device::DefaultDevice().Name());


#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else  // VERSION_INFO
  m.attr("__version__") = "dev";
#endif // VERSION_INFO
}
