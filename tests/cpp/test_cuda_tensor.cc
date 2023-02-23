#include "tensorlite/tensor.h"
#include "tensorlite/tensor_ops.h"
#include "gtest/gtest.h"

TEST(TestCudaTensor, TestCudaTensorEmpty) {
  tl::Tensor t1 = tl::Tensor::Empty({2, 3, 4}, tl::DataType("float"), 0,
                                    tl::Device::CudaDevice(0));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cuda_0");
  EXPECT_TRUE(t1.IsContiguous());
}

TEST(TestCudaTensor, TestCudaTensorOnes) {
  tl::Tensor t1 = tl::Tensor::Ones({2, 3, 4}, tl::DataType("float"),
                                   tl::Device::CudaDevice(0));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cuda_0");
  EXPECT_TRUE(t1.IsContiguous());

  tl::Tensor t2 = t1.Transfer(tl::Device::CpuDevice(0));
  auto ptr = t2.TypedPtr<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ((ptr[i]), 1.f);
  }
}

TEST(TestCudaTensor, TestCudaTensorZeros) {
  tl::Tensor t1 = tl::Tensor::Zeros({2, 3, 4}, tl::DataType("float"),
                                    tl::Device::CudaDevice(0));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cuda_0");
  EXPECT_TRUE(t1.IsContiguous());

  tl::Tensor t2 = t1.Transfer(tl::Device::CpuDevice(0));
  auto ptr = t2.TypedPtr<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ((ptr[i]), 0.f);
  }
}

TEST(TestCudaTensor, TensorCudaTensorUniform) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Uniform(shape, -1, 1, tl::DataType("float"),
                                      tl::Device::CudaDevice(0));

  tl::Tensor t2 = t1.Transfer(tl::Device::CpuDevice(0));

  auto *t2_ptr = t2.TypedPtr<float>();
  for (auto i = 0; i < t2.GetNumElems(); ++i) {
    EXPECT_LE(t2_ptr[i], 1);
    EXPECT_GE(t2_ptr[i], -1);
  }

  tl::Tensor t3 = tl::Tensor::Uniform(shape, -1, 1, tl::DataType("half"),
                                      tl::Device::CudaDevice(0));
}

TEST(TestCudaTensor, TensorCudaTensorNormal) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Normal(shape, -1, 1, tl::DataType("float"),
                                     tl::Device::CudaDevice(0));
}

TEST(TestCudaTensor, TestCudaTensorSameAs) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Zeros(shape, tl::DataType("double"),
                                    tl::Device::CudaDevice(0));
  tl::Tensor t2 = tl::Tensor::SameAs(t1, true, tl::DataType("int"));

  EXPECT_EQ(t2.GetDataType().Name(), "int32");

  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t2.GetShape(i));
  }
  EXPECT_EQ(t2.GetDevice().Name(), "cuda_0");
}

TEST(TestCudaTensor, TestCudaTensorFull) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  int val = 10;
  tl::Tensor t1 =
      tl::Tensor::Full(shape, tl::Scalar(val), 0, tl::Device::CudaDevice(0));
  EXPECT_EQ(t1.GetDataType().Name(), "int32");
  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t1.GetShape(i));
  }

  auto t1_cpu = t1.Transfer(tl::Device::CpuDevice(0));
  auto *ptr = t1_cpu.TypedPtr<int>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], val);
  }

  tl::Tensor t2 =
      tl::Tensor::Full<decltype(val)>(shape, val, 0, tl::Device::CudaDevice(0));
  EXPECT_EQ(t2.GetDataType().Name(), "int32");
  for (auto i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], t2.GetShape(i));
  }

  auto t2_cpu = t2.Transfer(tl::Device::CpuDevice(0));
  ptr = t2_cpu.TypedPtr<int>();
  for (auto i = 0; i < t2.GetNumElems(); ++i) {
    EXPECT_EQ(ptr[i], val);
  }
}

TEST(TestCudaTensor, TestCudaTensorContiguous) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};

  tl::Tensor t1 = tl::Tensor::Uniform(shape, 0, 1, tl::DataType("double"),
                                      tl::Device::CudaDevice(0));
  t1.Transpose_({2, 1, 0});
  EXPECT_FALSE(t1.IsContiguous());

  auto t2 = t1.Contiguous();
  EXPECT_TRUE(t2.IsContiguous());

  // TODO: check values are all equal
}

TEST(TestCudaTensor, TestCudaTensorCast) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4};
  tl::Tensor t1 = tl::Tensor::Ones(shape, tl::DataType("double"),
                                   tl::Device::CudaDevice(0));

  auto t2 = t1.Cast(tl::DataType("float16"));
  EXPECT_EQ(t2.GetDataType().Name(), "float16");

  auto t1_cpu = t1.Transfer(tl::Device::CpuDevice(0));
  auto t2_cpu = t2.Transfer(tl::Device::CpuDevice(0));

  auto *t1_ptr = t1_cpu.TypedPtr<double>();
  auto *t2_ptr = t2_cpu.TypedPtr<tl::fp16_t>();
  for (auto i = 0; i < t1.GetNumElems(); ++i) {
    EXPECT_EQ(t1_ptr[i], static_cast<double>(t2_ptr[i]));
  }
}

TEST(TestCudaTensor, TestCudaTensorDisplay) {
  std::vector<tl::shape_elem_t> shape{2, 3, 4, 5};
  tl::Tensor t1 = tl::Tensor::Uniform(shape, 0, 1, tl::DataType("double"),
                                      tl::Device::CudaDevice(0));
  t1.Display();
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
