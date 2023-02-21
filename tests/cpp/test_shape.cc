#include "tensorlite/tensor.h"
#include "gtest/gtest.h"

TEST(TestTensorShape, TestBroadcast) {
  using tl::TensorShape;
  auto shape1 = TensorShape({3, 4, 5});
  auto shape2 = TensorShape({1, 5});
  auto shape3 = std::vector<size_t>{4, 1};

  auto broadcast_shape_opt =
      TensorShape::BroadcastShape(shape1, shape2, shape3);
  EXPECT_TRUE(broadcast_shape_opt.has_value());
  auto broadcast_shape = broadcast_shape_opt.value();
  for (auto i = 0; i < broadcast_shape.Rank(); ++i) {
    EXPECT_EQ(shape1[i], broadcast_shape[i]);
  }
}
