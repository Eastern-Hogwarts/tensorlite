#ifndef TENSORLOTE_UTILS_CUDA_COMMON_H_
#define TENSORLOTE_UTILS_CUDA_COMMON_H_

#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/random.h"
#include "tensorlite/utils/singleton.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <limits>
#include <string>

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

#define CUDA_CALL_WITH_ERROR_VAR(func, e)                                      \
  {                                                                            \
    e = (func);                                                                \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

#define CURAND_CALL(func)                                                      \
  {                                                                            \
    curandStatus_t e = (func);                                                 \
    CHECK(e == CURAND_STATUS_SUCCESS)                                          \
        << "cuRAND: " << curandGetErrorString(e);                              \
  }

inline std::string curandGetErrorString(curandStatus_t err) {
  switch (err) {
  case CURAND_STATUS_SUCCESS:
    return "No error";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "Header file and linked library version do not match";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "Generator not initialized";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "Memory allocation failed";
  case CURAND_STATUS_TYPE_ERROR:
    return "Generator is wrong type";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "Argument out of range";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "Length requested is not a multple of dimension";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "GPU does not have double precision required by MRG32k3a";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "Kernel launch failure";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "Preexisting failure on library entry";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "Initialization of CUDA failed";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "Architecture mismatch, GPU does not support requested feature";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "Internal library error";
  default:
    return "Unknown error";
  }
}

namespace tl {

struct CUDAThreadLocalHandles {
  using CurandSeedType = unsigned long long;
  curandGenerator_t curand_gen;

  CUDAThreadLocalHandles() {
    // -- cuRAND -- //
    CURAND_CALL(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(
        curand_gen, RandomEngine::ThreadLocal().RandInt<CurandSeedType>(
                        std::numeric_limits<CurandSeedType>::max())));
  }

  ~CUDAThreadLocalHandles() {
    // -- cuRAND -- //
    CURAND_CALL(curandDestroyGenerator(curand_gen));
  }

  static CUDAThreadLocalHandles &ThreadLocal() {
    return ThreadLocalSingleton<CUDAThreadLocalHandles>::Get();
  }
};

} // namespace tl

#endif // TENSORLOTE_UTILS_CUDA_COMMON_H_
