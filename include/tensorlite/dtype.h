#ifndef TENSORLITE_DTYPE_H_
#define TENSORLITE_DTYPE_H_

#include <cstdint>

namespace tl {

/**
 * \brief
 *
 */
enum class DataTypeTag {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat16,
  kFloat32,
  kFloat64,
  kBool
};

/**
 * \brief
 *
 */
struct DataType {};

} // namespace tl

#endif // TENSORLITE_DTYPE_H_
