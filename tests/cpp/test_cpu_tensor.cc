#include "tensorlite/tensor.h"
#include "gtest/gtest.h"

TEST(TestCpuTensor, TestCpuTensorEmpty) {
  tl::Tensor t1 = tl::Tensor::Empty({2, 3, 4}, tl::DataType("float"));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cpu_0");
  EXPECT_TRUE(t1.IsContiguous());
}

TEST(TestCpuTensor, TestCpuTensorOnes) {
  tl::Tensor t1 = tl::Tensor::Ones({2, 3, 4}, tl::DataType("float"));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cpu_0");
  EXPECT_TRUE(t1.IsContiguous());

  auto ptr = t1.TypedPtr<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ((ptr[i]), 1.f);
  }
}

TEST(TestCpuTensor, TestCpuTensorZeros) {
  tl::Tensor t1 = tl::Tensor::Zeros({2, 3, 4}, tl::DataType("double"));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float64");
  EXPECT_EQ(t1.GetDevice().Name(), "cpu_0");
  EXPECT_TRUE(t1.IsContiguous());

  auto ptr = t1.TypedPtr<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ((ptr[i]), 0.0);
  }
}

TEST(TestCpuTensor, TensorCpuTensorUniform) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Uniform(shape, -1, 1, tl::DataType("float"));

  auto *t1_ptr = t1.TypedPtr<float>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_LE(t1_ptr[i], 1);
    EXPECT_GE(t1_ptr[i], -1);
  }
}

TEST(TestCpuTensor, TensorCpuTensorNormal) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Normal(shape, -1, 1, tl::DataType("float"));
}

TEST(TestCpuTensor, TestCpuTensorSameAs) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Zeros(shape, tl::DataType("double"));
  tl::Tensor t2 = tl::Tensor::SameAs(t1, true, tl::DataType("int"));

  EXPECT_EQ(t2.GetDataType().Name(), "int32");
  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t2.GetShape(i));
  }
  EXPECT_EQ(t2.GetDevice().Name(), "cpu_0");
}

TEST(TestCpuTensor, TestCpuTensorFull) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  int val = 10;
  tl::Tensor t1 = tl::Tensor::Full(shape, tl::Scalar(val));
  EXPECT_EQ(t1.GetDataType().Name(), "int32");
  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t1.GetShape(i));
  }

  auto *ptr = t1.TypedPtr<int>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], val);
  }

  tl::Tensor t2 = tl::Tensor::Full<decltype(val)>(shape, val);
  EXPECT_EQ(t2.GetDataType().Name(), "int32");
  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t2.GetShape(i));
  }

  ptr = t2.TypedPtr<int>();
  for (auto i = 0; i < t2.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], val);
  }
}

TEST(TestCpuTensor, TestCpuTensorContiguous) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};

  tl::Tensor t1 = tl::Tensor::Uniform(shape, 0, 1, tl::DataType("double"));
  t1.Transpose_({2, 1, 0});
  EXPECT_FALSE(t1.IsContiguous());

  auto t2 = t1.Contiguous();
  EXPECT_TRUE(t2.IsContiguous());

  // TODO: check values are all equal
}

TEST(TestCpuTensor, TestCpuTensorTransfer) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};

  tl::Tensor t1 = tl::Tensor::Uniform(shape, 0, 1, tl::DataType("double"));

  // to cpu
  auto t2 = t1.Transfer(tl::Device::CpuDevice());
  EXPECT_EQ(t2.GetDevice().Name(), "cpu_0");

  auto *t1_ptr = t1.TypedPtr<double>();
  auto *t2_ptr = t2.TypedPtr<double>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(t1_ptr[i], t2_ptr[i]);
  }

  // to cuda
  auto t3 = t1.Transfer(tl::Device::CudaDevice(0));
  EXPECT_EQ(t3.GetDevice().Name(), "cuda_0");

  auto t4 = t3.Transfer(tl::Device::CpuDevice());
  EXPECT_EQ(t4.GetDevice().Name(), "cpu_0");

  auto *t4_ptr = t4.TypedPtr<double>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(t1_ptr[i], t4_ptr[i]);
  }
}

TEST(TestCpuTensor, TestCpuTensorCast) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Ones(shape, tl::DataType("double"));

  auto t2 = t1.Cast(tl::DataType("float"));
  EXPECT_EQ(t2.GetDataType().Name(), "float32");

  auto *t1_ptr = t1.TypedPtr<double>();
  auto *t2_ptr = t2.TypedPtr<float>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(t1_ptr[i], static_cast<double>(t2_ptr[i]));
  }
}
