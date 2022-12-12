#ifndef TENSORLITE_UTILS_CPU_TOOLS_H_
#define TENSORLITE_UTILS_CPU_TOOLS_H_

#include <type_traits>

#include "tensorlite/tensor_op/tensor_iterator.h"
#include "tensorlite/utils/function_traits.h"

namespace tl {

namespace cpu {

template <typename traits, typename func_t, typename index_t, size_t... INDEX>

std::enable_if_t<!std::is_void_v<typename traits::return_t> &&
                     is_function_traits_v<traits>,
                 typename traits::return_t>
invoke_impl(const func_t &f, char *const *data, const index_t *strides, int i,
            std::index_sequence<INDEX...>) {
  return f(
      *(typename traits::template arg<INDEX>::type *)(data[INDEX] +
                                                      i * strides[INDEX])...);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
std::enable_if_t<std::is_void_v<typename traits::return_t> &&
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
std::enable_if_t<!std::is_void_v<typename traits::return_t>,
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
 *
 * \note this is the no-return version
 */
template <typename func_t, typename index_t,
          typename traits = function_traits<func_t>>
std::enable_if_t<std::is_void_v<typename traits::return_t>>
invoke(const func_t &f, char *const *data, const index_t *strides, int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  invoke_impl<traits>(f, data, strides, i, Indices{});
}

/**
 * \brief A basic loop function used by host(cpu) code for contiguous tensor
 * looping.
 *
 * \tparam Op The type of operator performed while looping
 * \param op The operator performed while looping
 * \param dptrs All data pointers of input and output tensors
 * \param strides All stride size of input and output tensors in bytes
 * \param N Number of elements in this loop
 * \return std::enable_if_t<!std::is_void_v<typename
 * function_traits<Op>::return_t>>
 */
template <typename Op>
std::enable_if_t<!std::is_void_v<typename function_traits<Op>::return_t>>
BasicCpuLoopFunc(const Op &op, char **dptrs, const size_t *strides, size_t N) {
  using trait = function_traits<Op>;

  // we assume the first tensor is the output tensor
  auto optr = reinterpret_cast<typename trait::return_t *>(dptrs[0]);

#ifdef USE_OPENMP
#pragma omp parallel for
#else // USE_OPENMP
#ifdef _MSC_VER
// https://docs.microsoft.com/en-us/cpp/preprocessor/loop?view=msvc-170
#pragma loop(hint_parallel(4))
#endif // _MSC_VER
#endif // USE_OPENMP
  for (int i = 0; i < N; ++i) {
    optr[i] = invoke(op, &dptrs[1], &strides[1], i);
  }
}

/**
 * \brief A basic loop function used by host(cpu) code for contiguous tensor
 * looping.
 *
 * \tparam Op The type of operator performed while looping
 * \param op The operator performed while looping
 * \param dptrs All data pointers of input and output tensors
 * \param strides All stride size of input and output tensors in bytes
 * \param N Number of elements in this loop
 * \return std::enable_if_t<!std::is_void_v<typename
 * function_traits<Op>::return_t>>
 *
 * \note This is the no-return version
 */
template <typename Op>
std::enable_if_t<std::is_void_v<typename function_traits<Op>::return_t>>
BasicCpuLoopFunc(const Op &op, char **dptrs, const size_t *strides, size_t N) {
  using trait = function_traits<Op>;

#ifdef USE_OPENMP
#pragma omp parallel for
#else // USE_OPENMP
#ifdef _MSC_VER
// https://docs.microsoft.com/en-us/cpp/preprocessor/loop?view=msvc-170
#pragma loop(hint_parallel(4))
#endif // _MSC_VER
#endif // USE_OPENMP
  for (int i = 0; i < N; ++i) {
    invoke(op, dptrs, strides, i);
  }
}

/**
 * \brief A 2d loop abstruct for elementwise tensor op on cpu devices
 *
 * \tparam Op The type of loop op
 *
 * \note DO NOT use this directly, this should be invoked by ForEach member
 * function in TensorIterator
 */
template <typename Op> struct Loop2d {
  Op op_;
  using op_trait = function_traits<Op>;
  static constexpr size_t ntensors = op_trait::rank;
  using data_t = std::array<char *, ntensors>;

  explicit Loop2d(const Op &op) : op_(op) {}
  explicit Loop2d(Op &&op) : op_(std::move(op)) {}

  static void advance(data_t &data, const size_t *outer_strides) {
    for (auto i = 0; i < data.size(); ++i) {
      data[i] += outer_strides[i];
    }
  }

  void operator()(char **dptrs, const size_t *strides, size_t inner_size,
                  size_t outer_size) {
    data_t data;
    std::copy_n(dptrs, ntensors, data.data());
    const size_t *inner_strides = &strides[ntensors];
    const size_t *outer_strides = &strides[0];

    for (size_t outer = 0; outer < outer_size; ++outer) {
      BasicCpuLoopFunc<Op>(op_, data.data(), inner_strides, inner_size);
      advance(data, outer_strides);
    }
  }
};

template <typename Op> decltype(auto) MakeLoop2d(Op &&op) {
  return Loop2d<Op>(std::forward<Op>(op));
}

/**
 * \brief Element-wise operation on \b contiguous tensors.
 *
 * \tparam Op The type of \a 'elem_op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param elem_op A callable object. The element-wise operation to be performed.
 */
template <typename Op> void CPUContiguousKernel(TensorIterator &iter, Op &&op) {
  CHECK(iter.Valid());
  using traits = function_traits<Op>;
  size_t num_elem = iter.NumElem();
  constexpr size_t num_tensors = traits::rank;

  std::array<char *, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;

#ifndef _MSC_VER
#pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char *>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].ElemSize();
  }

  BasicCpuLoopFunc<Op>(elem_op, base_ptrs.data(), elem_sizes.data(), num_elem);
}

/**
 * \brief Element-wise operation on tensors.
 *
 * \tparam Op The type of 'elem_op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param elem_op A callable object. The element-wise operation to be performed.
 */
template <typename Op>
void CPUElemwiseKernel(TensorIterator& iter, Op&& elem_op) {
  Loop2d<Op> loop = MakeLoop2d(std::forward<Op>(elem_op));
  iter.ForEach(loop);
}

} // namespace cpu
} // namespace tl

#endif // TENSORLITE_UTILS_CPU_TOOLS_H_
