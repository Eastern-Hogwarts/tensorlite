#include "tensorlite/tensor_op/cuda_internal_op.cuh"

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/utils/cuda_tools.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/device/data_transfer.h"

namespace tl {
namespace cuda {

template <typename DataTy> void CudaFillKernel(Tensor &tensor, DataTy val) {
  tensor.GetDevice().SetCurrentDevice();
  TensorIterator iter;
  iter.AddOutput(tensor).Build();
  CudaContiguousKernel(iter, [=] CUDA_LAMBDA() { return val; });
}

// switch on data type size
template void CudaFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CudaFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CudaFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CudaFillKernel<uint64_t>(Tensor &t, uint64_t val);

template <typename DataTy> void CudaCopyKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  src.GetDevice().SetCurrentDevice();
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();
  CudaElemwiseKernel(iter,
                     [] CUDA_LAMBDA(DataTy val) -> DataTy { return val; });
}

template void CudaCopyKernel<uint8_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint16_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint32_t>(const Tensor &src, Tensor &dst);
template void CudaCopyKernel<uint64_t>(const Tensor &src, Tensor &dst);

void CudaCastKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();

  DTYPE_SWITCH(src.GetDataType().GetTag(), [&]() {
    using src_dtype_t = scalar_t;
    DTYPE_SWITCH(dst.GetDataType().GetTag(), [&]() {
      using dst_dtype_t = scalar_t;
      CudaContiguousKernel(iter, [] CUDA_LAMBDA(src_dtype_t val) {
        return dtype_cast<src_dtype_t, dst_dtype_t, DeviceType::kCUDA>::cast(
            val);
      });
    });
  });
}

template <typename DataTy>
void CudaUniformDistKernelImpl(Tensor &tensor, Scalar low, Scalar high) {
  static_assert(std::is_same_v<DataTy, float>);
  DataTy *data = tensor.TypedPtr<DataTy>();
  size_t num_elem = tensor.GetNumElems();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;

  if (num_elem % 2) { // curand host API only accept even number of outputs
    tl::Tensor even_tensor = Tensor::Empty({num_elem + 1}, tensor.GetDataType(), tensor.GetAlignment(), tensor.GetDevice());
    DataTy *even_data = even_tensor.TypedPtr<DataTy>();
    CURAND_CALL(curandGenerateUniform(gen, even_data, num_elem + 1));

    // TODO: add stream support
    DataTransfer<DeviceType::kCUDA, DeviceType::kCUDA>(even_data, data, num_elem * sizeof(DataTy), tensor.GetDevice().GetId(), even_tensor.GetDevice().GetId());
  } else {
    CURAND_CALL(curandGenerateUniform(gen, data, num_elem));
  }

  DataTy scale = high.To<DataTy>() - low.To<DataTy>();
  DataTy bias = low.To<DataTy>();

  TensorIterator iter;
  iter.AddInput(tensor).AddOutput(tensor).Build();

  CudaContiguousKernel(
      iter, [=] CUDA_LAMBDA(DataTy elem) { return scale * elem + bias; });
}

template <>
void CudaUniformDistKernelImpl<double>(Tensor &tensor, Scalar low,
                                       Scalar high) {
  double *data = tensor.TypedPtr<double>();
  size_t num_elem = tensor.GetNumElems();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;

  if (num_elem % 2) { // curand host API only accept even number of outputs
    tl::Tensor even_tensor = Tensor::Empty({num_elem + 1}, tensor.GetDataType(), tensor.GetAlignment(), tensor.GetDevice());
    double *even_data = even_tensor.TypedPtr<double>();
    CURAND_CALL(curandGenerateUniformDouble(gen, even_data, num_elem + 1));

    // TODO: add stream support
    DataTransfer<DeviceType::kCUDA, DeviceType::kCUDA>(even_data, data, num_elem * sizeof(double), tensor.GetDevice().GetId(), even_tensor.GetDevice().GetId());
  } else {
    CURAND_CALL(curandGenerateUniformDouble(gen, data, num_elem));
  }

  double scale = high.To<double>() - low.To<double>();
  double bias = low.To<double>();

  TensorIterator iter;
  iter.AddInput(tensor).AddOutput(tensor).Build();

  CudaContiguousKernel(
      iter, [=] CUDA_LAMBDA(double elem) { return scale * elem + bias; });
}

void CudaUniformDistKernel(Tensor &tensor, Scalar low, Scalar high) {
  tensor.GetDevice().SetCurrentDevice();

  if (tensor.GetDataType().GetTag() == DataTypeTag::kFloat16) {
    Tensor single_tensor = Tensor::SameAs(
        tensor, true, DataType(DataTypeTag::kFloat32), tensor.GetDevice());
    CudaUniformDistKernelImpl<float>(single_tensor, low, high);
    CudaCastKernel(single_tensor, tensor);
  } else {
    DTYPE_SWITCH_FLOAT_WITHOUT_HALF(tensor.GetDataType().GetTag(), [&]() {
      CudaUniformDistKernelImpl<scalar_t>(tensor, low, high);
    });
  }
}

template <typename DataTy>
void CudaNormalDistKernelImpl(Tensor &tensor, Scalar mean, Scalar stddev) {
  static_assert(std::is_same_v<DataTy, float>);
  DataTy *data = tensor.TypedPtr<DataTy>();
  size_t num_elem = tensor.GetNumElems();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;

  if (num_elem % 2) { // curand host API only accept even number of outputs
    tl::Tensor even_tensor = Tensor::Empty({num_elem + 1}, tensor.GetDataType(), tensor.GetAlignment(), tensor.GetDevice());
    DataTy *even_data = even_tensor.TypedPtr<DataTy>();
    CURAND_CALL(curandGenerateNormal(gen, even_data, num_elem + 1, mean.To<DataTy>(),
                       stddev.To<DataTy>()));

    // TODO: add stream support
    DataTransfer<DeviceType::kCUDA, DeviceType::kCUDA>(even_data, data, num_elem * sizeof(DataTy), tensor.GetDevice().GetId(), even_tensor.GetDevice().GetId());
  } else {
    CURAND_CALL(curandGenerateNormal(gen, data, num_elem, mean.To<DataTy>(),
                       stddev.To<DataTy>()));
  }
}

template <>
void CudaNormalDistKernelImpl<double>(Tensor &tensor, Scalar mean,
                                      Scalar stddev) {
  double *data = tensor.TypedPtr<double>();
  size_t num_elem = tensor.GetNumElems();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;

  if (num_elem % 2) { // curand host API only accept even number of outputs
    tl::Tensor even_tensor = Tensor::Empty({num_elem + 1}, tensor.GetDataType(), tensor.GetAlignment(), tensor.GetDevice());
    double *even_data = even_tensor.TypedPtr<double>();
    CURAND_CALL(curandGenerateNormalDouble(gen, even_data, num_elem + 1, mean.To<double>(),
                       stddev.To<double>()));

    // TODO: add stream support
    DataTransfer<DeviceType::kCUDA, DeviceType::kCUDA>(even_data, data, num_elem * sizeof(double), tensor.GetDevice().GetId(), even_tensor.GetDevice().GetId());
  } else {
    CURAND_CALL(curandGenerateNormalDouble(gen, data, num_elem, mean.To<double>(),
                       stddev.To<double>()));
  }
}

void CudaNormalDistKernel(Tensor &tensor, Scalar mean, Scalar stddev) {
  tensor.GetDevice().SetCurrentDevice();

  if (tensor.GetDataType().GetTag() == DataTypeTag::kFloat16) {
    Tensor single_tensor = Tensor::SameAs(
        tensor, true, DataType(DataTypeTag::kFloat32), tensor.GetDevice());
    CudaNormalDistKernelImpl<float>(single_tensor, mean, stddev);
    CudaCastKernel(single_tensor, tensor);
  } else {
    DTYPE_SWITCH_FLOAT_WITHOUT_HALF(tensor.GetDataType().GetTag(), [&]() {
      CudaNormalDistKernelImpl<scalar_t>(tensor, mean, stddev);
    });
  }
}

} // namespace cuda
} // namespace tl
