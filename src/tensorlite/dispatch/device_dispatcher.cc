#include "tensorlite/dispatch/device_dispatch.h"

#include <string>
#include <unordered_map>

#include "tensorlite/utils/logging.h"

namespace tl {

class DeviceDispatcher::DeviceDispatcherImpl {
public:
  bool HasOperator(const std::string &name) const {
    return op_table_.find(name) != op_table_.end();
  }

  bool HasDefinition(const std::string &name, DeviceType key) const {
    if (!HasOperator(name)) {
      return false;
    }
    return op_table_.at(name)->HasDefinition(key);
  }

  void DefineOperator(const std::string &name, const std::string &doc) {
    if (!HasOperator(name)) {
      op_table_[name] = std::make_unique<OperatorHandle>(name, doc);
    }
  }

  void Register(const std::string &name, DeviceType key, void *func_ptr) {
    if (!HasOperator(name)) {
      DefineOperator(name, "");
    }
    op_table_.at(name)->Register(key, func_ptr);
  }

  void *GetKernel(const std::string &name, DeviceType key) const {
    if (!HasOperator(name)) {
      return nullptr;
    }
    return op_table_.at(name)->GetKernel(key);
  }

private:
  std::unordered_map<std::string, std::unique_ptr<OperatorHandle>> op_table_;
};

DeviceDispatcher::DeviceDispatcher()
    : impl_(std::make_unique<DeviceDispatcherImpl>()) {}

DeviceDispatcher::~DeviceDispatcher() = default;

void DeviceDispatcher::DefineOperator(const std::string &name,
                                      const std::string &doc) {
  impl_->DefineOperator(name, doc);
}

bool DeviceDispatcher::HasOperator(const std::string &name) {
  return impl_->HasOperator(name);
}

bool DeviceDispatcher::HasDefinition(const std::string &name, DeviceType key) {
  return impl_->HasDefinition(name, key);
}

void DeviceDispatcher::Register(const std::string &name, DeviceType key,
                                void *func_ptr) {
  impl_->Register(name, key, func_ptr);
}

const void* DeviceDispatcher::GetKernel(const std::string& name, DeviceType key) const {
  return impl_->GetKernel(name, key);
}

} // namespace tl
