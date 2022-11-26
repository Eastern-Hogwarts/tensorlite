#ifndef TENSORLITE_ALLOCATOR_UTILS_H_
#define TENSORLITE_ALLOCATOR_UTILS_H_

#include <cstdlib>

namespace tl {
namespace utils {

inline constexpr size_t ceil_align(size_t val, size_t align) {
  return (val + align - 1) / align * align;
}

} // namespace utils
} // namespace tl

#endif // TENSORLITE_ALLOCATOR_UTILS_H_
