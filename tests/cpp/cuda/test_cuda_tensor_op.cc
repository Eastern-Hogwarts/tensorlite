#include "tensorlite/tensor.h"
#include "tensorlite/tensor_ops.h"
#include "gtest/gtest.h"
#include <cmath>

template <typename CompareTy, typename T1, typename T2>
constexpr static CompareTy ScalarAbsDiff(const T1 &val1, const T2 &val2) {
  return std::abs(static_cast<CompareTy>(val1) - static_cast<CompareTy>(val2));
}

TEST(TestCudaTensor, TestCudaTensorAdd) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"),
                                   tl::Device::CudaDevice(0));
  tl::Tensor t2 =
      tl::Tensor::Zeros({4}, tl::DataType("double"), tl::Device::CudaDevice(0));
  auto t3 = tl::native_ops::Add(t1, t2).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], 1.);
  }
}

TEST(TestCudaTensor, TestCudaTensorSub) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"),
                                   tl::Device::CudaDevice(0));
  tl::Tensor t2 =
      tl::Tensor::Zeros({4}, tl::DataType("double"), tl::Device::CudaDevice(0));
  t2.Fill(0.1);
  auto t3 = tl::native_ops::Sub(t1, t2).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 0.9), 1e-3);
  }
}

TEST(TestCudaTensor, TestCudaTensorMul) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"),
                                   tl::Device::CudaDevice(0));
  tl::Tensor t2 =
      tl::Tensor::Full<double>({4}, 2.5, 0, tl::Device::CudaDevice(0));

  auto t3 = tl::native_ops::Mul(t1, t2).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 2.5), 1e-3);
  }
}

TEST(TestCudaTensor, TestCudaTensorDiv) {
  tl::Tensor t1 =
      tl::Tensor::Full<double>({3, 4}, 4.0, 0, tl::Device::CudaDevice(0));
  tl::Tensor t2 =
      tl::Tensor::Full<double>({4}, 2.0, 0, tl::Device::CudaDevice(0));
  auto t3 = tl::native_ops::Div(t1, t2).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 2.), 1e-3);
  }
}

TEST(TestCudaTensor, TestCudaTensorNeg) {
  tl::Tensor t =
      tl::Tensor::Full<double>({3, 4}, 4.0, 0, tl::Device::CudaDevice(0));
  auto neg = tl::native_ops::Neg(t).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(neg.GetNumElems(), 12);
  auto *ptr = neg.TypedPtr<double>();
  for (auto i = 0; i < neg.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], -4.0), 1e-3);
  }
}

TEST(TestCudaTensor, TestCudaTensorSqrt) {
  tl::Tensor t =
      tl::Tensor::Full({3, 4}, tl::fp16_t(4.), 0, tl::Device::CudaDevice(0));
  auto o = tl::native_ops::Sqrt(t).Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *ptr = o.TypedPtr<tl::fp16_t>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<float>(ptr[i], 2.f), 1e-3f);
  }
}

TEST(TestCudaTensor, TestCudaTensorAcos) {
  tl::Tensor t = tl::Tensor::Uniform({12}, 0, 1, tl::DataType("float"),
                                     tl::Device::CudaDevice(0));
  auto o = tl::native_ops::Acos(t);

  auto t_cpu = t.Transfer(tl::Device::DefaultDevice());
  auto o_cpu = o.Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o_cpu.TypedPtr<float>();
  auto *tptr = t_cpu.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<float>(optr[i], acosf(tptr[i])), 1e-2f);
  }
}

TEST(TestCudaTensor, TestCudaTensorAcosh) {
  tl::Tensor t = tl::Tensor::Uniform({12}, 1, 3, tl::DataType("float"),
                                     tl::Device::CudaDevice(0));
  auto o = tl::native_ops::Acosh(t);

  auto t_cpu = t.Transfer(tl::Device::DefaultDevice());
  auto o_cpu = o.Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o_cpu.TypedPtr<float>();
  auto *tptr = t_cpu.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<float>(optr[i], acoshf(tptr[i])), 1e-3f);
  }
}

TEST(TestCudaTensor, TestCudaTensorAbs) {
  tl::Tensor t = tl::Tensor::Normal({12}, 0, 1, tl::DataType("float"),
                                    tl::Device::CudaDevice(0));
  auto o = tl::native_ops::Abs(t);

  auto t_cpu = t.Transfer(tl::Device::DefaultDevice());
  auto o_cpu = o.Transfer(tl::Device::DefaultDevice());

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o_cpu.TypedPtr<float>();
  auto *tptr = t_cpu.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_FLOAT_EQ(optr[i], abs(tptr[i]));
  }
}
