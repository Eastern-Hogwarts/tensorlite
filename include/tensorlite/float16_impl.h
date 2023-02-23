#ifndef TENSORLITE_FP16_IMPL_CUH_
#define TENSORLITE_FP16_IMPL_CUH_
#include <cstring>
#include <limits>

#include "tensorlite/macros.h"

namespace tl {

#if defined(__CUDACC__)
inline TENSOR_HOST_DEVICE Float16::Float16(float f) {
  __half tmp = __float2half(f);
  this->bits = *reinterpret_cast<uint16_t*>(&tmp);
}
inline TENSOR_HOST_DEVICE Float16::Float16(const __half& nv_half)
  : bits(*reinterpret_cast<const uint16_t *>(&nv_half)) {}
inline TENSOR_HOST_DEVICE Float16::operator float() const { return __half2float(this->operator __half()); }
inline TENSOR_HOST_DEVICE Float16::operator __half() const {
  auto tmp = bits;
  return *reinterpret_cast<__half*>(&tmp);
}
#else
#include "fp16.h"
inline TENSOR_HOST_DEVICE Float16::Float16(float f) : bits(fp16_ieee_from_fp32_value(f)) {}
inline TENSOR_HOST_DEVICE Float16::operator float() const { return fp16_ieee_to_fp32_value(bits); }
#endif


inline TENSOR_HOST_DEVICE Float16 operator+(const Float16& a, const Float16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator-(const Float16& a, const Float16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator*(const Float16& a, const Float16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator/(const Float16& a, const Float16& b)
__ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator-(const Float16& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

inline TENSOR_HOST_DEVICE Float16& operator+=(Float16& a, const Float16& b) {
  a = a + b;
  return a;
}

inline TENSOR_HOST_DEVICE Float16& operator-=(Float16& a, const Float16& b) {
  a = a - b;
  return a;
}

inline TENSOR_HOST_DEVICE Float16& operator*=(Float16& a, const Float16& b) {
  a = a * b;
  return a;
}

inline TENSOR_HOST_DEVICE Float16& operator/=(Float16& a, const Float16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline TENSOR_HOST_DEVICE float operator+(Float16 a, float b) {
  return static_cast<float>(a) + b;
}
inline TENSOR_HOST_DEVICE float operator-(Float16 a, float b) {
  return static_cast<float>(a) - b;
}
inline TENSOR_HOST_DEVICE float operator*(Float16 a, float b) {
  return static_cast<float>(a) * b;
}
inline TENSOR_HOST_DEVICE float operator/(Float16 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline TENSOR_HOST_DEVICE float operator+(float a, Float16 b) {
  return a + static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float operator-(float a, Float16 b) {
  return a - static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float operator*(float a, Float16 b) {
  return a * static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float operator/(float a, Float16 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline TENSOR_HOST_DEVICE float& operator+=(float& a, const Float16& b) {
  return a += static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float& operator-=(float& a, const Float16& b) {
  return a -= static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float& operator*=(float& a, const Float16& b) {
  return a *= static_cast<float>(b);
}
inline TENSOR_HOST_DEVICE float& operator/=(float& a, const Float16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline TENSOR_HOST_DEVICE double operator+(Float16 a, double b) {
  return static_cast<double>(a) + b;
}
inline TENSOR_HOST_DEVICE double operator-(Float16 a, double b) {
  return static_cast<double>(a) - b;
}
inline TENSOR_HOST_DEVICE double operator*(Float16 a, double b) {
  return static_cast<double>(a) * b;
}
inline TENSOR_HOST_DEVICE double operator/(Float16 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline TENSOR_HOST_DEVICE double operator+(double a, Float16 b) {
  return a + static_cast<double>(b);
}
inline TENSOR_HOST_DEVICE double operator-(double a, Float16 b) {
  return a - static_cast<double>(b);
}
inline TENSOR_HOST_DEVICE double operator*(double a, Float16 b) {
  return a * static_cast<double>(b);
}
inline TENSOR_HOST_DEVICE double operator/(double a, Float16 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline TENSOR_HOST_DEVICE Float16 operator+(Float16 a, int b) {
  return a + static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator-(Float16 a, int b) {
  return a - static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator*(Float16 a, int b) {
  return a * static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator/(Float16 a, int b) {
  return a / static_cast<Float16>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator+(int a, Float16 b) {
  return static_cast<Float16>(a) + b;
}
inline TENSOR_HOST_DEVICE Float16 operator-(int a, Float16 b) {
  return static_cast<Float16>(a) - b;
}
inline TENSOR_HOST_DEVICE Float16 operator*(int a, Float16 b) {
  return static_cast<Float16>(a) * b;
}
inline TENSOR_HOST_DEVICE Float16 operator/(int a, Float16 b) {
  return static_cast<Float16>(a) / b;
}

//// Arithmetic with int64_t

inline TENSOR_HOST_DEVICE Float16 operator+(Float16 a, int64_t b) {
  return a + static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator-(Float16 a, int64_t b) {
  return a - static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator*(Float16 a, int64_t b) {
  return a * static_cast<Float16>(b);
}
inline TENSOR_HOST_DEVICE Float16 operator/(Float16 a, int64_t b) {
  return a / static_cast<Float16>(b);
}

inline TENSOR_HOST_DEVICE Float16 operator+(int64_t a, Float16 b) {
  return static_cast<Float16>(a) + b;
}
inline TENSOR_HOST_DEVICE Float16 operator-(int64_t a, Float16 b) {
  return static_cast<Float16>(a) - b;
}
inline TENSOR_HOST_DEVICE Float16 operator*(int64_t a, Float16 b) {
  return static_cast<Float16>(a) * b;
}
inline TENSOR_HOST_DEVICE Float16 operator/(int64_t a, Float16 b) {
  return static_cast<Float16>(a) / b;
}

} // namespace tl

namespace std {

template <>
class numeric_limits<::tl::Float16> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr ::tl::Float16 min() {
    return ::tl::Float16::FromHex(0x0400);
  }
  static constexpr ::tl::Float16 lowest() {
    return ::tl::Float16::FromHex(0xFBFF);
  }
  static constexpr ::tl::Float16 max() {
    return ::tl::Float16::FromHex(0x7BFF);
  }
  static constexpr ::tl::Float16 epsilon() {
    return ::tl::Float16::FromHex(0x1400);
  }
  static constexpr ::tl::Float16 round_error() {
    return ::tl::Float16::FromHex(0x3800);
  }
  static constexpr ::tl::Float16 infinity() {
    return ::tl::Float16::FromHex(0x7C00);
  }
  static constexpr ::tl::Float16 quiet_NaN() {
    return ::tl::Float16::FromHex(0x7E00);
  }
  static constexpr ::tl::Float16 signaling_NaN() {
    return ::tl::Float16::FromHex(0x7D00);
  }
  static constexpr ::tl::Float16 denorm_min() {
    return ::tl::Float16::FromHex(0x0001);
  }
};
} // namespace std

#endif //TENSORLITE_FP16_IMPL_CUH_
