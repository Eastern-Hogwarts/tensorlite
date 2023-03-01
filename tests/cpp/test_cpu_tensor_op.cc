#include "tensorlite/tensor.h"
#include "tensorlite/tensor_ops.h"
#include "gtest/gtest.h"
#include <cmath>

template <typename CompareTy, typename T1, typename T2>
constexpr static CompareTy ScalarAbsDiff(const T1 &val1, const T2 &val2) {
  return std::abs(static_cast<CompareTy>(val1) - static_cast<CompareTy>(val2));
}

TEST(TestCpuTensor, TestCpuTensorAdd) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"));
  tl::Tensor t2 = tl::Tensor::Zeros({4}, tl::DataType("double"));
  auto t3 = tl::native_ops::Add(t1, t2);

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], 1.);
  }
}

TEST(TestCpuTensor, TestCpuTensorSub) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"));
  tl::Tensor t2 = tl::Tensor::Zeros({4}, tl::DataType("double"));
  t2.Fill(0.1);
  auto t3 = tl::native_ops::Sub(t1, t2);

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 0.9), 1e-3);
  }
}

TEST(TestCpuTensor, TestCpuTensorMul) {
  tl::Tensor t1 = tl::Tensor::Ones({3, 4}, tl::DataType("double"));
  tl::Tensor t2 = tl::Tensor::Full<double>({4}, 2.5, 0);
  auto t3 = tl::native_ops::Mul(t1, t2);

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 2.5), 1e-3);
  }
}

TEST(TestCpuTensor, TestCpuTensorDiv) {
  tl::Tensor t1 = tl::Tensor::Full<double>({3, 4}, 4.0, 0);
  tl::Tensor t2 = tl::Tensor::Full<double>({4}, 2.0, 0);
  auto t3 = tl::native_ops::Div(t1, t2);

  EXPECT_EQ(t3.GetNumElems(), 12);
  auto *ptr = t3.TypedPtr<double>();
  for (auto i = 0; i < t3.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], 2.), 1e-3);
  }
}

TEST(TestCpuTensor, TestCpuTensorNeg) {
  tl::Tensor t = tl::Tensor::Full<double>({3, 4}, 4.0, 0);
  auto neg = tl::native_ops::Neg(t);

  EXPECT_EQ(neg.GetNumElems(), 12);
  auto *ptr = neg.TypedPtr<double>();
  for (auto i = 0; i < neg.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<double>(ptr[i], -4.0), 1e-3);
  }
}

TEST(TestCpuTensor, TestCpuTensorSqrt) {
  tl::Tensor t = tl::Tensor::Empty({4}, tl::DataType("half"));
  t.Fill<tl::fp16_t>(tl::fp16_t(4.));
  auto o = tl::native_ops::Sqrt(t);

  EXPECT_EQ(o.GetNumElems(), 4);
  auto *ptr = o.TypedPtr<tl::fp16_t>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_LT(ScalarAbsDiff<float>(ptr[i], 2.f), 1e-3f);
  }
}

TEST(TestCpuTensor, TestCpuTensorAcos) {
  tl::Tensor t = tl::Tensor::Uniform({12}, 0, 1, tl::DataType("float"));
  auto o = tl::native_ops::Acos(t);

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o.TypedPtr<float>();
  auto *tptr = t.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_FLOAT_EQ(optr[i], acosf(tptr[i]));
  }
}

TEST(TestCpuTensor, TestCpuTensorAcosh) {
  tl::Tensor t = tl::Tensor::Uniform({12}, 1, 3, tl::DataType("float"));
  auto o = tl::native_ops::Acosh(t);

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o.TypedPtr<float>();
  auto *tptr = t.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_FLOAT_EQ(optr[i], acoshf(tptr[i]));
  }
}

TEST(TestCpuTensor, TestCpuTensorAbs) {
  tl::Tensor t = tl::Tensor::Normal({12}, 0, 1, tl::DataType("float"));
  auto o = tl::native_ops::Abs(t);

  EXPECT_EQ(o.GetNumElems(), 12);
  auto *optr = o.TypedPtr<float>();
  auto *tptr = t.TypedPtr<float>();
  for (auto i = 0; i < o.GetNumElems(); ++i) {
    EXPECT_FLOAT_EQ(optr[i], abs(tptr[i]));
  }
}
