#include "tensorlite/dtype.h"

#include <mutex>
#include <unordered_map>

namespace tl {

static void
InitStringToTagMap(std::unordered_map<std::string, DataTypeTag> &map) {
  // normalized names
  map["int8"] = DataTypeTag::kInt8;
  map["int32"] = DataTypeTag::kInt32;
  map["int64"] = DataTypeTag::kInt64;

  map["uint8"] = DataTypeTag::kUInt8;
  map["uint32"] = DataTypeTag::kUInt32;
  map["uint64"] = DataTypeTag::kUInt64;

  map["float16"] = DataTypeTag::kFloat16;
  map["float32"] = DataTypeTag::kFloat32;
  map["float64"] = DataTypeTag::kFloat64;

  map["bool"] = DataTypeTag::kBool;

  // alias
  map["int"] = DataTypeTag::kInt32;
  map["float"] = DataTypeTag::kFloat32;
  map["double"] = DataTypeTag::kFloat64;
  map["half"] = DataTypeTag::kFloat16;
}

DataTypeTag DataType::StringToTag(const std::string &type_str) {
  static std::unordered_map<std::string, DataTypeTag> map_;
  std::once_flag init_flag;
  std::call_once(init_flag, InitStringToTagMap, map_);

  if (map_.find(type_str) == map_.end()) {
    return DataTypeTag::kInvalid;
  }
  return map_.at(type_str);
}

} // namespace tl
