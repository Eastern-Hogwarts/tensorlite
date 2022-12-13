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
