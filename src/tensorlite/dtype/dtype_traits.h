#include <cstdint>
#include <cuda_fp16.h>

#include "tensorlite/dtype.h"
#include "tensorlite/float16.h"

namespace tl {

template <DataTypeTag DataTy> struct DataTypeToCRT {};

template <> struct DataTypeToCRT<DataTypeTag::kInt8> {
  using CpuType = int8_t;
  using CudaType = signed char;
};

template <> struct DataTypeToCRT<DataTypeTag::kInt32> {
  using CpuType = int32_t;
  using CudaType = int32_t;
};

template <> struct DataTypeToCRT<DataTypeTag::kInt64> {
  using CpuType = int64_t;
  using CudaType = int64_t;
};

template <> struct DataTypeToCRT<DataTypeTag::kUInt8> {
  using CpuType = uint8_t;
  using CudaType = unsigned char;
};

template <> struct DataTypeToCRT<DataTypeTag::kUInt32> {
  using CpuType = uint32_t;
  using CudaType = uint32_t;
};

template <> struct DataTypeToCRT<DataTypeTag::kUInt64> {
  using CpuType = uint64_t;
  using CudaType = uint64_t;
};

template <> struct DataTypeToCRT<DataTypeTag::kFloat16> {
  using CpuType = fp16_t;
  using CudaType = __half;
};

template <> struct DataTypeToCRT<DataTypeTag::kFloat32> {
  using CpuType = float;
  using CudaType = float;
};

template <> struct DataTypeToCRT<DataTypeTag::kFloat64> {
  using CpuType = double;
  using CudaType = double;
};

template <> struct DataTypeToCRT<DataTypeTag::kBool> {
  using CpuType = bool;
  using CudaType = bool;
};

} // namespace tl
