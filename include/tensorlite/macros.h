/*!
 * \file macros.h
 * \brief Useful macros used in tensor
 */
#ifndef TENSORLITE_MACROS_H_
#define TENSORLITE_MACROS_H_

#ifndef TENSORLITE_DLL
#ifdef _WIN32
#ifdef TENSORLITE_EXPORTS
#define TENSORLITE_DLL __declspec(dllexport)
#else // TENSORLITE_EXPORTS
#define TENSORLITE_DLL __declspec(dllimport)
#endif // TENSORLITE_EXPORTS
#else  // _WIN32
#define TENSORLITE_DLL
#endif // _WIN32
#endif // TENSORLITE_DLL

#define RESTRICT __restrict__

#if defined(__CUDACC__)
#define TENSOR_KERNEL __global__
#define TENSOR_HOST __host__
#define TENSOR_DEVICE __device__
#define TENSOR_HOST_DEVICE TENSOR_HOST TENSOR_DEVICE
#define CUDA_LAMBDA TENSOR_DEVICE TENSOR_HOST
#else
#define TENSOR_KERNEL
#define TENSOR_HOST
#define TENSOR_DEVICE
#define TENSOR_HOST_DEVICE
#define CUDA_LAMBDA
#endif

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
  __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_function__
#endif

#endif // TENSORLITE_MACROS_H_
