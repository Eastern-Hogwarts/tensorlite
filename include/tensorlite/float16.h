/*!
 * \file half.h
 * \brief A CPU implementation of IEEE-754 Half-Precision Floating-Point
 */
#ifndef TENSORLITE_FP16_H_
#define TENSORLITE_FP16_H_

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif // __CUDACC__

#include "tensorlite/macros.h"

namespace tl {

struct alignas(2) Float16 {
  uint16_t bits;

  inline Float16() = default;
  inline TENSOR_HOST_DEVICE Float16(float f);

  // useless: to avoid more than one ctor ...
  constexpr explicit Float16(uint16_t bits, int useless) : bits(bits) {}

#ifdef __CUDACC__
  inline TENSOR_HOST_DEVICE Float16(const __half &nv_half);
  inline TENSOR_HOST_DEVICE operator __half() const;
#endif // __CUDACC__

  inline TENSOR_HOST_DEVICE operator float() const;

  constexpr static Float16 FromHex(uint16_t b) { return Float16(b, 0); }
};

using fp16_t = Float16;

} // namespace tl

#include "tensorlite/float16_impl.h"
#endif // TENSORLITE_FP16_H_
