#include "gtest/gtest.h"
#include "tensorlite/tensor.h"

TEST(TestTensor, TestCpuTensorEmpty) {
  tl::Tensor t1 = tl::Tensor::Empty({2, 3, 4}, tl::DataType("float"));
  EXPECT_EQ(t1.GetNumElems(), (2 * 3 * 4));
  EXPECT_EQ(t1.GetDataType().Name(), "float32");
  EXPECT_EQ(t1.GetDevice().Name(), "cpu_0");
}
