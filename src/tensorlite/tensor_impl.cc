#include "tensorlite/tensor.h"

namespace tl {

bool TensorShapeWithStride::IsContiguous() const {
  if (IsScalar())
    return true; // scalar is contiguous
  int64_t s = 1;
  for (size_t i = rank_ - 1; i < rank_; --i) {
    if (stride_[i] == s) {
      s *= shape_[i];
    } else {
      return false;
    }
  }
  return true;
}

} // namespace tl
