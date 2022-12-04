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
  using return = R;
  using args = std::tuple<Args...>;
  using type = R(Args...);
  static constexpr size_t arity = sizeof...(Args);

  template <size_t i> struct arg {
    using type = typename std::tuple_element_t<i, args>;
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

} // namespace tl

#endif // TENSORLITE_UTILS_FUNCTION_TRAITS_H_
