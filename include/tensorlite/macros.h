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

// #define RESTRICT __restrict__

// #define TENSOR_KERNEL __global__
// #define TENSOR_HOST   __host__
// #define TENSOR_DEVICE __device__
// #define TENSOR_HOST_DEVICE TENSOR_HOST TENSOR_DEVICE
// #define CUDA_LAMBDA TENSOR_DEVICE TENSOR_HOST

#endif // TENSORLITE_MACROS_H_
