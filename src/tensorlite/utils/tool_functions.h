#ifndef TENSORLITE_UTILS_TOOL_FUNCTIONS_H_
#define TENSORLITE_UTILS_TOOL_FUNCTIONS_H_

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace tl {

/**
 * \brief Computate total number of elements from a given shape (vector).
 *
 * \param shape The given shape
 * \return T Total number of elements
 */
template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>> * = nullptr>
inline T ShapeNumElem(const std::vector<T> &shape) {
  if (shape.empty())
    return T(0);
#ifdef __GNUC__
  return std::accumulate(shape.cbegin(), shape.cend(), static_cast<T>(1),
                         std::multiplies<T>{});
#else  // __GNUC__
  return std::reduce(shape.cbegin(), shape.cend(), static_cast<T>(1),
                     std::multiplies<T>{});
#endif // __GNUC__
}

} // namespace tl

#endif // TENSORLITE_UTILS_TOOL_FUNCTIONS_H_
