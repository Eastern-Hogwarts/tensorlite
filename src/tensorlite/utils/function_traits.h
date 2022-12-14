#ifndef TENSORLITE_UTILS_FUNCTION_TRAITS_H_
#define TENSORLITE_UTILS_FUNCTION_TRAITS_H_

#include <tuple>
#include <type_traits>

namespace tl {

/**
 * \brief Strip operator() function from a class
 */
template <typename T> struct strip_class {};

template <typename C, typename R, typename... Args>
struct strip_class<R (C::*)(Args...)> {
  using type = R(Args...);
};

template <typename C, typename R, typename... Args>
struct strip_class<R (C::*)(Args...) const> {
  using type = R(Args...);
};

template <typename T> using strip_class_t = typename strip_class<T>::type;

/**
 * \brief Extract useful properties from a function type
 */
template <typename T>
struct function_traits
    : public function_traits<strip_class_t<decltype(&T::operator())>> {};

// normal function
template <typename R, typename... Args> struct function_traits<R(Args...)> {
  using return_t = R;
  using arg_tuple = std::tuple<Args...>;
  using type = R(Args...);

  // num of arguments this function takes
  static constexpr size_t arity = sizeof...(Args);

  // num of all variables (include return)
  static constexpr size_t rank = arity + (std::is_void_v<R> ? 0 : 1);

  template <size_t i> struct arg {
    using type = typename std::tuple_element_t<i, arg_tuple>;
  };
};

// remove reference and pointer
template <typename T>
struct function_traits<T &> : public function_traits<T> {};
template <typename T>
struct function_traits<T *> : public function_traits<T> {};

// function pointer
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> : public function_traits<R(Args...)> {};

template <typename T> struct is_function_traits : std::false_type {};

template <typename T>
struct is_function_traits<function_traits<T>> : std::true_type {};

template <typename T>
constexpr inline bool is_function_traits_v = is_function_traits<T>::value;

template <typename T, int Arity> struct is_function_with_arity {
  static constexpr bool value = (function_traits<T>::arity == Arity);
};

template <typename T, int Arity>
constexpr inline bool is_function_with_arity_v =
    is_function_with_arity<T, Arity>::value;

#define DEFINE_IS_FUNCTION_WITH_ARITY(type_, arity_)                           \
  template <typename T>                                                        \
  struct is_##type_##_function : is_function_with_arity<T, arity_> {};         \
  template <typename T>                                                        \
  constexpr inline bool is_##type_##_function_v =                              \
      is_##type_##_function<T>::value;

DEFINE_IS_FUNCTION_WITH_ARITY(unary, 1);
DEFINE_IS_FUNCTION_WITH_ARITY(binary, 2);
DEFINE_IS_FUNCTION_WITH_ARITY(ternary, 3);

#undef DEFINE_IS_FUNCTION_WITH_ARITY

/**
 * \brief An efficient, type-erasing, non-owning reference to a callable. This
 * is intended for use as the type of a function parameter that is not used
 * after the function in question returns.
 *
 * \note This class does not own the callable, so it is not in general safe to
 * store a function_ref.
 */
template <typename T>
struct function_reference
    : public function_reference<typename function_traits<T>::type> {};

template <typename R, typename... Args> struct function_reference<R(Args...)> {
  using func_type = R(Args...);
  using CallbackFnType = R(intptr_t, Args...);
  using Self = function_reference<R(Args...)>;
  CallbackFnType *callback_ = nullptr;
  intptr_t func_ptr_;

  template <typename Callable>
  static R CallbackFn(intptr_t fn_ptr, Args... args) {
    return (*reinterpret_cast<Callable *>(fn_ptr))(std::forward<Args>(args)...);
  }

  template <
      typename Callable,
      typename std::enable_if_t<!std::is_same_v<
          std::remove_reference_t<Callable>, function_reference>> * = nullptr,
      typename std::enable_if_t<std::is_convertible_v<
          std::invoke_result_t<Callable &&, Args &&...>, R>> * = nullptr>
  function_reference(Callable &&callable)
      : func_ptr_(reinterpret_cast<intptr_t>(&callable)),
        callback_(CallbackFn<std::remove_reference_t<Callable>>) {}

  R operator()(Args... args) {
    return callback_(func_ptr_, std::forward<Args>(args)...);
  }
};

/**
 * \brief Get the Function Ref object
 *
 * \tparam Callable The callable type
 * \param callable The callable object
 * \return function_reference<
 * typename function_traits<std::remove_reference_t<Callable>>::type>
 */
template <typename Callable>
function_reference<
    typename function_traits<std::remove_reference_t<Callable>>::type>
GetFunctionRef(Callable &&callable) {
  return function_reference<
      typename function_traits<std::remove_reference_t<Callable>>::type>(
      callable);
}

} // namespace tl

#endif // TENSORLITE_UTILS_FUNCTION_TRAITS_H_
