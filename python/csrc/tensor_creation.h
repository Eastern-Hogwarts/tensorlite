#ifndef TENSORLITE_PYTHON_API_TENSORCREATION_H_
#define TENSORLITE_PYTHON_API_TENSORCREATION_H_

#include <vector>
#include <type_traits>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>

#include "pybind11/pybind11.h"
#include "tensorlite/tensor.h"
#include "tensorlite/dtype.h"
#include "utils.h"

namespace tl {
namespace pyapi {

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor empty(
  const py::args& shape_args,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  return tl::Tensor::Empty(tl::pyapi::pyargs_to_vector(shape_args), tl::DataType(dtype), 0, tl::Device(device));
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor ones(
  const py::args& shape_args,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  return tl::Tensor::Ones(tl::pyapi::pyargs_to_vector(shape_args), tl::DataType(dtype), tl::Device(device));
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor zeros(
  const py::args& shape_args,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  return tl::Tensor::Zeros(tl::pyapi::pyargs_to_vector(shape_args), tl::DataType(dtype), tl::Device(device));
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor uniform(
  const py::args& shape_args,
  double low,
  double high,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  return tl::Tensor::Uniform(tl::pyapi::pyargs_to_vector(shape_args), low, high, tl::DataType(dtype), tl::Device(device));
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor normal(
  const py::args& shape_args,
  double mean,
  double stddev,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  return tl::Tensor::Normal(tl::pyapi::pyargs_to_vector(shape_args), mean, stddev, tl::DataType(dtype), tl::Device(device));
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor same_as(
  const tl::Tensor& other,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>* dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>* device
) {
  std::optional<tl::DataType> inner_dtype = std::nullopt;
  std::optional<tl::Device> inner_device = std::nullopt;
  if (dtype) {
    inner_dtype = tl::DataType(*dtype);
  }
  if (device) {
    inner_device = tl::Device(*device);
  }
  return tl::Tensor::SameAs(other, true, inner_dtype, inner_device);
}

template <typename DtypeArgType, typename DeviceArgType>
tl::Tensor full(
  const py::args& shape_args,
  double value,
  const std::enable_if_t<std::is_constructible_v<tl::DataType, DtypeArgType>, DtypeArgType>& dtype,
  const std::enable_if_t<std::is_constructible_v<tl::Device, DeviceArgType>, DeviceArgType>& device
) {
  tl::DataType inner_dtype(dtype);
  DTYPE_SWITCH(inner_dtype.GetTag(), [&](){
    return tl::Tensor::Full(tl::pyapi::pyargs_to_vector(shape_args), static_cast<scalar_t>(value), 0, tl::Device(device));
  });

  std::ostringstream sm;
  sm << "Invalid data type: " << inner_dtype.Name();
  throw std::invalid_argument(sm.str());
}

} // namespace pyapi
} // namespace tl


#endif  // TENSORLITE_PYTHON_API_TENSORCREATION_H_
