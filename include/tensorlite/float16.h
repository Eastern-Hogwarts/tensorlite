/*!
 * \file half.h
 * \brief A CPU implementation of IEEE-754 Half-Precision Floating-Point
 */
#ifndef TENSORLITE_FP16_H_
#define TENSORLITE_FP16_H_

#include "fp16.h"

namespace tl {

struct Float16 {
  uint16_t bits;

  Float16() = default;
  Float16(float f) : bits(fp16_ieee_from_fp32_value(f)) {}
  operator float() const { return fp16_ieee_to_fp32_value(bits); }

#define DEFINE_BINARY_OP(op)                                                   \
  Float16 operator op(Float16 other) const {                                   \
    return static_cast<float>(*this) op static_cast<float>(other);             \
  }

  DEFINE_BINARY_OP(+)
  DEFINE_BINARY_OP(-)
  DEFINE_BINARY_OP(*)
  DEFINE_BINARY_OP(/)

  static Float16 FromHex(uint16_t b) {
    Float16 h;
    h.bits = b;
    return h;
  }
};

using fp16_t = Float16;

} // namespace tl
#endif // TENSORLITE_FP16_H_
