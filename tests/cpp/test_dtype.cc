#include "tensorlite/dtype.h"
#include "gtest/gtest.h"
#include <iostream>

TEST(TestDtype, TestScalar) {
  tl::Scalar s(1.f);
  tl::DataType dtype = s.GetDataType();
  EXPECT_EQ(dtype.Name(), "float32");
}
