#ifndef TENSORLITE_UTILS_CUDA_TOOLS_H_
#define TENSORLITE_UTILS_CUDA_TOOLS_H_

#include "tensorlite/device.h"
#include "tensorlite/dtype.h"
#include "tensorlite/macros.h"
#include "tensorlite/tensor.h"
#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/function_traits.h"
#include <cuda_fp16.h>
#include <type_traits>
#include <utility>

namespace tl {
namespace cuda {

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
TENSOR_HOST_DEVICE
    std::enable_if_t<!std::is_void_v<typename traits::return_t> &&
                         is_function_traits_v<traits>,
                     typename traits::return_t>
    invoke_impl(const func_t &f, char *const *data, const index_t *strides,
                int i, std::index_sequence<INDEX...>) {
  return f(
      *(typename traits::template arg<INDEX>::type *)(data[INDEX] +
                                                      i * strides[INDEX])...);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
TENSOR_HOST_DEVICE std::enable_if_t<std::is_void_v<typename traits::return_t> &&
                                    is_function_traits_v<traits>>
invoke_impl(const func_t &f, char *const *data, const index_t *strides, int i,
            std::index_sequence<INDEX...>) {
  f(*(typename traits::template arg<INDEX>::type *)(data[INDEX] +
                                                    i * strides[INDEX])...);
}

/*!
 * \brief Invoke functions on type-erased tensor data.
 *
 * \tparam func_t Type of the function to be invoked
 * \tparam index_t Tensor index type
 * \tparam traits Function traits of \a func_t
 * \param f The function to be invoked
 * \param data Type-erased data pointers of given tensors
 * \param strides Strides of tensors
 * \param i Index value
 * \return Result of the function.
 */
template <typename func_t, typename index_t,
          typename traits = function_traits<func_t>>
TENSOR_HOST_DEVICE std::enable_if_t<!std::is_void_v<typename traits::return_t>,
                                    typename traits::return_t>
invoke(const func_t &f, char *const *data, const index_t *strides, int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

/*!
 * \brief Invoke functions on type-erased tensor data.
 *
 * \tparam func_t Type of the function to be invoked
 * \tparam index_t Tensor index type
 * \tparam traits Function traits of \a func_t
 * \param f The function to be invoked
 * \param data Type-erased data pointers of given tensors
 * \param strides Strides of tensors
 * \param i Index value
 */
template <typename func_t, typename index_t,
          typename traits = function_traits<func_t>>
TENSOR_HOST_DEVICE std::enable_if_t<std::is_void_v<typename traits::return_t>>
invoke(const func_t &f, char *const *data, const index_t *strides, int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  invoke_impl<traits>(f, data, strides, i, Indices{});
}

/*!
 * \brief Helper for computing linear offsets of tensors with different
 *        memory layouts in a cuda thread. This class is helpful when
 *        performing element-wise operation on these tensors.
 *
 * \tparam ShapeElemType Type of shape's elements
 * \tparam NARGS Number of tensors involved in this element-wise operation.
 */
template <typename ShapeElemType, size_t NARGS> struct OffsetCalculator {

  static constexpr size_t kMaxTensorRank = ::tl::TensorShape::kMaxTensorRank;

  OffsetCalculator(size_t num_axes, const std::vector<ShapeElemType> &shape,
                   const std::vector<size_t> &strides,
                   const std::array<size_t, NARGS> &elem_size)
      : num_axes_(num_axes) {
    CHECK_LE(num_axes, kMaxTensorRank);

#ifndef _MSC_VER
#pragma unroll
#endif
    for (size_t t = 0; t < NARGS; ++t) {
      elem_size_[t] = elem_size[t];
    }

    for (size_t i = 0; i < num_axes; ++i) {
      shape_[i] = shape[i];
#ifndef _MSC_VER
#pragma unroll
#endif
      for (size_t t = 0; t < NARGS; ++t) {
        strides_[i][t] = strides[i * NARGS + t];
      }
    }
  }
  // offset must be a pointer points to a array with size kMaxTensorRank
  TENSOR_HOST_DEVICE void get(size_t gidx, size_t *offset) const {
#ifndef _MSC_VER
#pragma unroll
#endif
    for (size_t i = 0; i < kMaxTensorRank; ++i) {
      offset[i] = 0;
    }

    size_t mod;
    for (size_t d = num_axes_ - 1; d < num_axes_; --d) {
      mod = gidx % shape_[d];
      gidx = gidx / shape_[d];

#ifndef _MSC_VER
#pragma unroll
#endif
      for (size_t t = 0; t < NARGS; ++t) {
        offset[t] += mod * strides_[d][t] * elem_size_[t];
      }
    }
  }

  size_t num_axes_;
  ShapeElemType shape_[kMaxTensorRank]; // std::array is an experimental feature
                                        // in libcu++
  size_t strides_[kMaxTensorRank]
                 [std::max<ShapeElemType>(NARGS, ShapeElemType(1))];
  size_t elem_size_[std::max<size_t>(NARGS, size_t(1))];
};

template <size_t nt, size_t vt, typename func_t>
__global__ void elementwise_kernel(size_t N, func_t f) {
  int tid = threadIdx.x;
  int nv = (int)(nt * vt);
  int idx = nv * blockIdx.x + tid;
#ifndef _MSC_VER
#pragma unroll
#endif
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <size_t nt, size_t vt, typename Op>
void cudaElemwiseKernelImpl(size_t N, Op &&op) {
  if (N == 0)
    return;

  dim3 block(static_cast<unsigned int>(nt));
  dim3 grid(static_cast<unsigned int>((N + block.x * vt - 1) / (block.x * vt)));
  elementwise_kernel<nt, vt, Op><<<grid, block>>>(N, op);
  CUDA_CALL(cudaGetLastError());
}

/*!
 * \brief Element-wise operation on \b contiguous tensors.
 *
 * \tparam Op The type of \a 'op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 * \note This kernel is for contiguous tensors
 */
template <typename Op>
std::enable_if_t<!std::is_void_v<typename function_traits<Op>::return_t>>
CudaContiguousKernel(TensorIterator &iter, Op &&op) {
  CHECK(iter.IsValid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

  constexpr size_t num_tensors = traits::rank;
  std::array<char *, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;
#ifndef _MSC_VER
#pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char *>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].GetDataType().Size();
  }

  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;
  cudaElemwiseKernelImpl<128, unroll>(
      iter.NumElem(), [=] CUDA_LAMBDA(size_t idx) {
        size_t offset[traits::rank];
        for (int i = 0; i < traits::rank; ++i) {
          offset[i] = idx * elem_sizes[i];
        }
        arg0_t *out = (arg0_t *)(base_ptrs[0] + offset[0]);
        *out = invoke(op, &base_ptrs[1], &offset[1], 1);
      });
}

/*!
 * \brief Element-wise operation on \b contiguous tensors.
 *
 * \tparam Op The type of \a 'op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 * \note This kernel is for contiguous tensors
 */
template <typename Op>
std::enable_if_t<std::is_void_v<typename function_traits<Op>::return_t>>
CudaContiguousKernel(TensorIterator &iter, Op &&op) {
  CHECK(iter.IsValid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

  constexpr size_t num_tensors = traits::rank;
  std::array<char *, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;
#ifndef _MSC_VER
#pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char *>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].GetDataType().Size();
  }

  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;
  cudaElemwiseKernelImpl<128, unroll>(
      iter.NumElem(), [=] CUDA_LAMBDA(size_t idx) {
        size_t offset[traits::rank];
        for (int i = 0; i < traits::rank; ++i) {
          offset[i] = idx * elem_sizes[i];
        }
        invoke(op, &base_ptrs[0], &offset[0], 1);
      });
}

/**
 * \brief Element-wise operation on tensors.
 *
 * \tparam Op The type of \a 'op'.
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 * \return std::enable_if_t<std::is_void_v<typename function_traits<Op>::return_t>>
 *
 * \note The input tensors may be not contiguous.
 */
template <typename Op>
std::enable_if_t<std::is_void_v<typename function_traits<Op>::return_t>>
CudaElemwiseKernel(TensorIterator& iter, Op&& op) {
  CHECK(iter.IsValid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

  constexpr size_t num_tensors = traits::rank;
  std::array<char *, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;
#ifndef _MSC_VER
#pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char *>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].GetDataType().Size();
  }
  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;

  OffsetCalculator<shape_elem_t, num_tensors> offset_calc(iter.Rank(), iter.Shape(), iter.GetStridesInBytes(), elem_sizes);

  cudaElemwiseKernelImpl<128, unroll>(
      iter.NumElem(), [=] CUDA_LAMBDA(size_t idx) {
        size_t offset[traits::rank];
        offset_calc.get(idx, offset);
        invoke(op, &base_ptrs[0], &offset[0], 1);
      });
}

/**
 * \brief Element-wise operation on tensors.
 *
 * \tparam Op The type of \a 'op'.
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 * \return std::enable_if_t<!std::is_void_v<typename function_traits<Op>::return_t>>
 *
 * \note The input tensors may be not contiguous.
 */
template <typename Op>
std::enable_if_t<!std::is_void_v<typename function_traits<Op>::return_t>>
CudaElemwiseKernel(TensorIterator& iter, Op&& op) {
  CHECK(iter.IsValid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

  constexpr size_t num_tensors = traits::rank;
  std::array<char *, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;
#ifndef _MSC_VER
#pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char *>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].GetDataType().Size();
  }
  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;

  OffsetCalculator<shape_elem_t, num_tensors> offset_calc(iter.Rank(), iter.Shape(), iter.GetStridesInBytes(), elem_sizes);

  cudaElemwiseKernelImpl<128, unroll>(
      iter.NumElem(), [=] CUDA_LAMBDA(size_t idx) {
        size_t offset[traits::rank];
        offset_calc.get(idx, offset);
        arg0_t *out = (arg0_t *)(base_ptrs[0] + offset[0]);
        *out = invoke(op, &base_ptrs[1], &offset[1], 1);
      });
}

} // namespace cuda

// TENSOR_HOST_DEVICE: actually these are only __device__ function, but we need
// to call it in a
// __host__ __device__ lambda function. Why we use __host__ __device__ lambda?
// Because we need its function_traits on CPU.
template <typename SrcTy, typename DstTy>
struct dtype_cast<SrcTy, DstTy, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static DstTy cast(SrcTy src) {
    return static_cast<DstTy>(src);
  }
};

template <typename SrcTy> struct dtype_cast<SrcTy, fp16_t, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static fp16_t cast(SrcTy src) {
    __half temp = __float2half_rn(static_cast<float>(src));
    return *reinterpret_cast<fp16_t *>(&temp);
  }
};

template <typename DstTy> struct dtype_cast<fp16_t, DstTy, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static DstTy cast(fp16_t src) {
    float temp = __half2float(*reinterpret_cast<__half *>(&src));
    return static_cast<DstTy>(temp);
  }
};

template <> struct dtype_cast<fp16_t, fp16_t, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static fp16_t cast(fp16_t src) { return src; }
};

} // namespace tl

#endif // TENSORLITE_UTILS_CUDA_TOOLS_H_
