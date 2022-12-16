#ifndef TENSORLITE_DTYPE_H_
#define TENSORLITE_DTYPE_H_

#include <cassert>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>

#include "tensorlite/device.h"
#include "tensorlite/float16.h"
#include "tensorlite/macros.h"

namespace tl {

/**
 * \brief
 *
 */
enum class DataTypeTag {
  kInt8,
  kInt32,
  kInt64,
  kUInt8,
  kUInt32,
  kUInt64,
  kFloat16,
  kFloat32,
  kFloat64,
  kBool,
  kInvalid
};

/**
 * \brief Define trait struct mapping from c/c++ type to tensorlite dtype
 *
 * \tparam CTy The input c/c++ type
 */
template <typename CTy> struct CRuntimeTypeToDataType {};

#define DEFINE_CRT_TO_DTYPE(CTy, DTy)                                          \
  template <> struct CRuntimeTypeToDataType<CTy> {                             \
    static constexpr DataTypeTag type = DTy;                                   \
  };

DEFINE_CRT_TO_DTYPE(int8_t, DataTypeTag::kInt8);
DEFINE_CRT_TO_DTYPE(uint8_t, DataTypeTag::kUInt8);
DEFINE_CRT_TO_DTYPE(int32_t, DataTypeTag::kInt32);
DEFINE_CRT_TO_DTYPE(uint32_t, DataTypeTag::kUInt32);
DEFINE_CRT_TO_DTYPE(int64_t, DataTypeTag::kInt64);
DEFINE_CRT_TO_DTYPE(uint64_t, DataTypeTag::kUInt64);
DEFINE_CRT_TO_DTYPE(fp16_t, DataTypeTag::kFloat16);
DEFINE_CRT_TO_DTYPE(float, DataTypeTag::kFloat32);
DEFINE_CRT_TO_DTYPE(double, DataTypeTag::kFloat64);
DEFINE_CRT_TO_DTYPE(bool, DataTypeTag::kBool);
#undef DEFINE_CRT_TO_DTYPE

template <typename CRT>
constexpr inline DataTypeTag crt_to_dtype_v = CRuntimeTypeToDataType<CRT>::type;

/**
 * \brief Supported c/c++ data type
 *
 * \tparam CRT
 */
template <typename CRT> struct SupportCRT {
  static constexpr bool value = false;
};

#define DEFINE_SUPPORT_CRT(CRT)                                                \
  template <> struct SupportCRT<CRT> {                                         \
    static constexpr bool value = true;                                        \
  };

DEFINE_SUPPORT_CRT(int8_t)
DEFINE_SUPPORT_CRT(int32_t)
DEFINE_SUPPORT_CRT(int64_t)
DEFINE_SUPPORT_CRT(uint8_t)
DEFINE_SUPPORT_CRT(uint32_t)
DEFINE_SUPPORT_CRT(uint64_t)
DEFINE_SUPPORT_CRT(fp16_t)
DEFINE_SUPPORT_CRT(uint16_t) // for pure byte computation
DEFINE_SUPPORT_CRT(float)
DEFINE_SUPPORT_CRT(double)
DEFINE_SUPPORT_CRT(bool)

#undef DEFINE_SUPPORT_CRT
template <typename T>
inline constexpr bool support_crt_v = SupportCRT<T>::value;

/**
 * \brief A unified data type cast template for c/c++ data type casting
 *
 * \tparam SrcTy source type
 * \tparam DstTy destination type
 * \tparam XPU device type
 */
template <typename SrcTy, typename DstTy, DeviceType XPU> struct dtype_cast {};

template <typename SrcTy, typename DstTy>
struct dtype_cast<SrcTy, DstTy, DeviceType::kCPU> {
  static DstTy cast(SrcTy src) { return static_cast<DstTy>(src); }
};

/**
 * \brief Get the size in bytes of a tensorlite data type
 *
 * \param dtype_tag The tag of the input data type
 * \return size_t
 */
inline size_t DataTypeSize(DataTypeTag dtype_tag) {
  switch (dtype_tag) {
  case DataTypeTag::kInt8:
  case DataTypeTag::kUInt8:
    return 1;
  case DataTypeTag::kFloat16:
    return 2;
  case DataTypeTag::kInt32:
  case DataTypeTag::kUInt32:
  case DataTypeTag::kFloat32:
    return 4;
  case DataTypeTag::kInt64:
  case DataTypeTag::kUInt64:
  case DataTypeTag::kFloat64:
    return 8;
  default:
    return 0;
  }
}

/**
 * \brief Get the alignment in bytes of a tensorlite data type
 *
 * \param dtype_tag The tag of the input data type
 * \return size_t
 * \note Currently all types we support have the same size of alignment as its
 * shape
 */
inline size_t DataTypeAlignment(DataTypeTag dtype_tag) {
  return DataTypeSize(dtype_tag);
}

/**
 * \brief Runtime dispatch according to the tensorlite dtype
 */
#define DTYPE_SWITCH_CASE(switch_t, crt, ...)                                  \
  case switch_t: {                                                             \
    using scalar_t = crt;                                                      \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define DTYPE_SWITCH(dtype, ...)                                               \
  {                                                                            \
    DataTypeTag _st = DataTypeTag(dtype);                                      \
    switch (_st) {                                                             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt8, int8_t, __VA_ARGS__)               \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt8, uint8_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt32, int32_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt32, uint32_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt64, int64_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt64, uint64_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat16, fp16_t, __VA_ARGS__)            \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat32, float, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat64, double, __VA_ARGS__)            \
    }                                                                          \
  }

#define DTYPE_SWITCH_CUDA_HALF(dtype, ...)                                     \
  {                                                                            \
    DataTypeTag _st = DataTypeTag(dtype);                                      \
    switch (_st) {                                                             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt8, int8_t, __VA_ARGS__)               \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt8, uint8_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt32, int32_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt32, uint32_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt64, int64_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt64, uint64_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat16, __half, __VA_ARGS__)            \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat32, float, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat64, double, __VA_ARGS__)            \
    }                                                                          \
  }

#define DTYPE_SWITCH_WITHOUT_HALF(dtype, ...)                                  \
  {                                                                            \
    DataTypeTag _st = DataTypeTag(dtype);                                      \
    switch (_st) {                                                             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt8, int8_t, __VA_ARGS__)               \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt8, uint8_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt32, int32_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt32, uint32_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kInt64, int64_t, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kUInt64, uint64_t, __VA_ARGS__)           \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat32, float, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat64, double, __VA_ARGS__)            \
    }                                                                          \
  }

#define DTYPE_SWITCH_FLOAT(dtype, ...)                                         \
  {                                                                            \
    DataTypeTag _st = DataTypeTag(dtype);                                      \
    switch (_st) {                                                             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat16, fp16_t, __VA_ARGS__)            \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat32, float, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat64, double, __VA_ARGS__)            \
    }                                                                          \
  }

#define DTYPE_SWITCH_FLOAT_WITHOUT_HALF(dtype, ...)                            \
  {                                                                            \
    DataTypeTag _st = DataTypeTag(dtype);                                      \
    switch (_st) {                                                             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat32, float, __VA_ARGS__)             \
      DTYPE_SWITCH_CASE(DataTypeTag::kFloat64, double, __VA_ARGS__)            \
    }                                                                          \
  }

/**
 * \brief Traits checking whether a given dtype is floating point number type
 *
 * \tparam DataTy
 */
template <typename DataTy> struct is_floatint_point {
  static constexpr bool value = false;
};

#define IS_FLOATINT_POINT_CASE(type)                                           \
  template <> struct is_floatint_point<type> {                                 \
    static constexpr bool value = true;                                        \
  }
IS_FLOATINT_POINT_CASE(float);
IS_FLOATINT_POINT_CASE(double);
IS_FLOATINT_POINT_CASE(fp16_t);
#undef IS_FLOATINT_POINT_CASE

/**
 * \brief Traits checking whether a given dtype is signed integer
 *
 * \tparam DataTy
 */
template <typename DataTy> struct is_signed_integral {
  static constexpr bool value = false;
};

#define IS_SIGNED_INTEGRAL_CASE(type)                                          \
  template <> struct is_signed_integral<type> {                                \
    static constexpr bool value = true;                                        \
  }
IS_SIGNED_INTEGRAL_CASE(int8_t);
IS_SIGNED_INTEGRAL_CASE(int32_t);
IS_SIGNED_INTEGRAL_CASE(int64_t);
#undef IS_SIGNED_INTEGRAL_CASE

/**
 * \brief Traits checking whether a given dtype is unsigned integer
 *
 * \tparam DataTy
 */
template <typename DataTy> struct is_unsigned_integral {
  static constexpr bool value = false;
};

#define IS_UNSIGNED_INTEGRAL_CASE(type)                                        \
  template <> struct is_unsigned_integral<type> {                              \
    static constexpr bool value = true;                                        \
  }
IS_UNSIGNED_INTEGRAL_CASE(uint8_t);
IS_UNSIGNED_INTEGRAL_CASE(uint32_t);
IS_UNSIGNED_INTEGRAL_CASE(uint64_t);
#undef IS_UNSIGNED_INTEGRAL_CASE

/**
 * \brief The data type of a tensor
 *
 */
class DataType {
public:
  /**
   * \brief Default constructor of DataType class.
   *
   * Default dtype is double-precision floating number
   */
  DataType() : tag_(DataTypeTag::kFloat64) {}

  /**
   * \brief Construct a new Data Type object
   *
   * \param tag Tag object indicating the data type
   */
  explicit DataType(DataTypeTag tag) : tag_(tag) {
    assert(tag_ != DataTypeTag::kInvalid);
  }

  /**
   * \brief Construct a new Data Type object from data type name
   *
   * \param type_str The given data type name
   *
   * \note Supported names: int8, int32, int64, uint8, uint32,
   *                        uint64, float16, float32, float64,
   *                        bool, int, float, double, half
   */
  explicit DataType(const std::string &type_str) : tag_(StringToTag(type_str)) {
    assert(tag_ != DataTypeTag::kInvalid);
  }

  DataType(const DataType &) = default;
  DataType(DataType &&) = default;
  DataType &operator=(const DataType &) = default;
  DataType &operator=(DataType &&) = default;
  DataType &operator=(DataTypeTag tag) {
    tag_ = tag;
    return *this;
  }
  ~DataType() = default;

  /**
   * \brief Get the size in bytes of this data type
   *
   * \return size_t
   */
  size_t Size() const { return DataTypeSize(tag_); }

  /**
   * \brief Get the alignment in bytes of this data type
   *
   * \return size_t
   */
  size_t Alignment() const { return DataTypeAlignment(tag_); }

  /**
   * \brief Get the name in string form of this data type
   *
   * \return std::string_view
   */
  std::string_view Name() const { return TagToString(tag_); }

  friend std::ostream &operator<<(std::ostream &os, const DataType &dtype) {
    os << dtype.Name();
    return os;
  }

  /**
   * \brief Get the Tag object
   *
   * \return DataTypeTag
   */
  DataTypeTag GetTag() const { return tag_; }

  /**
   * \brief Check two data type are the same
   */
  bool operator==(const DataType &other) const {
    return other.tag_ == this->tag_;
  }

private:
  /**
   * \brief Get data type tag from its name
   *
   * \param type_str Input data type name
   * \return DataTypeTag
   */
  TENSORLITE_DLL static DataTypeTag StringToTag(const std::string &type_str);

  /**
   * \brief Get data type in string from its tag
   *
   * \param tag Input data type tag
   * \return constexpr std::string_view
   */
  static constexpr std::string_view TagToString(DataTypeTag tag);

private:
  DataTypeTag tag_;
};

constexpr std::string_view DataType::TagToString(DataTypeTag tag) {
  switch (tag) {
  case DataTypeTag::kInt8:
    return "int8";
  case DataTypeTag::kInt32:
    return "int32";
  case DataTypeTag::kInt64:
    return "int64";
  case DataTypeTag::kUInt8:
    return "uint8";
  case DataTypeTag::kUInt32:
    return "uint32";
  case DataTypeTag::kUInt64:
    return "uint64";
  case DataTypeTag::kFloat16:
    return "float16";
  case DataTypeTag::kFloat32:
    return "float32";
  case DataTypeTag::kFloat64:
    return "float64";
  case DataTypeTag::kBool:
    return "bool";
  default:
    return "invalid";
  }
}

/**
 * \brief A scalar type used to store all types supported
 */
class Scalar {
  using ValueType = std::variant<int8_t, uint8_t, int32_t, uint32_t, int64_t,
                                 uint64_t, fp16_t, float, double, bool>;

public:
  Scalar() = default;

  template <typename T> Scalar(T val) : val_(val) {}

  template <typename T> T To() const {
    return std::visit([](auto &&arg) -> T { return static_cast<T>(arg); },
                      val_);
  }

  DataType GetDataType() const {
    return DataType(std::visit(
        [](auto &&arg) {
          return crt_to_dtype_v<
              std::remove_cv_t<std::remove_reference_t<decltype(arg)>>>;
        },
        val_));
  }

#define DEFINE_CAST(type)                                                      \
  operator type() const { return To<type>(); }

  DEFINE_CAST(float)
  DEFINE_CAST(double)
  DEFINE_CAST(int8_t)
  DEFINE_CAST(uint8_t)
  DEFINE_CAST(int32_t)
  DEFINE_CAST(uint32_t)
  DEFINE_CAST(int64_t)
  DEFINE_CAST(uint64_t)
  DEFINE_CAST(bool)

private:
  ValueType val_;
};

} // namespace tl

#endif // TENSORLITE_DTYPE_H_
