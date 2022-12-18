#include "tensorlite/tensor_op/cpu_internal_op.h"
#include "tensorlite/utils/random.h"
namespace tl {
namespace cpu {

template <typename DataTy> void CpuFillKernel(Tensor &tensor, DataTy val) {
  DataTy *data = tensor.TypedPtr<DataTy>();
  TensorIterator iter;
  iter.AddOutput(tensor);
  iter.Build();
  CPUContiguousKernel(iter, [=]() -> DataTy { return val; });
}

template void CpuFillKernel<uint8_t>(Tensor &t, uint8_t val);
template void CpuFillKernel<uint16_t>(Tensor &t, uint16_t val);
template void CpuFillKernel<uint32_t>(Tensor &t, uint32_t val);
template void CpuFillKernel<uint64_t>(Tensor &t, uint64_t val);

template <typename DataTy> void CpuCopyKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();
  CPUElemwiseKernel(iter, [](DataTy val) { return val; });
}

template void CpuCopyKernel<uint8_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint16_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint32_t>(const Tensor &src, Tensor &dst);
template void CpuCopyKernel<uint64_t>(const Tensor &src, Tensor &dst);

void CpuCastKernel(const Tensor &src, Tensor &dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  TensorIterator iter;
  iter.AddOutput(dst).AddInput(src).Build();

  DTYPE_SWITCH(src.GetDataType().GetTag(), [&]() {
    using src_dtype_t = scalar_t;
    DTYPE_SWITCH(dst.GetDataType().GetTag(), [&]() {
      using dst_dtype_t = scalar_t;
      CPUContiguousKernel(iter, [](src_dtype_t val) {
        return dtype_cast<src_dtype_t, dst_dtype_t, DeviceType::kCPU>::cast(
            val);
      });
    });
  });
}

template <typename DataTy>
void UniformDistKernelImpl(Tensor &tensor, Scalar low, Scalar high) {
  DataTy low_val = low.To<DataTy>(), high_val = high.To<DataTy>();
  auto &rand_gen = RandomEngine::ThreadLocal();

  TensorIterator iter;
  iter.AddOutput(tensor).Build();

  CPUContiguousKernel(iter,
                      [&]() { return rand_gen.Uniform(low_val, high_val); });
}

void CpuUniformDistKernel(Tensor &tensor, Scalar low, Scalar high) {
  DTYPE_SWITCH_FLOAT(tensor.GetDataType().GetTag(), [&]() {
    UniformDistKernelImpl<scalar_t>(tensor, low, high);
  });
}

template <typename DataTy>
void NormalDistKernelImpl(Tensor &tensor, Scalar mean, Scalar stddev) {
  DataTy mean_val = mean.To<DataTy>(), std_val = stddev.To<DataTy>();
  auto &rand_gen = RandomEngine::ThreadLocal();

  TensorIterator iter;
  iter.AddOutput(tensor).Build();

  CPUContiguousKernel(iter,
                      [&]() { return rand_gen.Normal(mean_val, std_val); });
}

void CpuNormalDistKernel(Tensor &tensor, Scalar mean, Scalar stddev) {
  DTYPE_SWITCH_FLOAT(tensor.GetDataType().GetTag(), [&]() {
    NormalDistKernelImpl<scalar_t>(tensor, mean, stddev);
  });
}

} // namespace cpu
} // namespace tl
