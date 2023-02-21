#include "tensorlite/dispatch/device_dispatch.h"

#include <unordered_map>
#include <utility>

#include "tensorlite/utils/logging.h"

namespace tl {

class OperatorHandle::OperatorHandleImpl {
public:
  OperatorHandleImpl() = default;

  OperatorHandleImpl(const std::string &name, const std::string &doc = "")
      : name_(name), doc_(doc) {}

  OperatorHandleImpl(OperatorHandleImpl &&other)
      : name_(std::move(other.name_)), doc_(other.doc_),
        kernel_table_(other.kernel_table_) {}

  OperatorHandleImpl(const OperatorHandleImpl &) = delete;

  friend void swap(OperatorHandleImpl &op1, OperatorHandleImpl &op2) {
    using std::swap;
    swap(op1.name_, op2.name_);
    swap(op1.doc_, op2.doc_);
    swap(op1.kernel_table_, op2.kernel_table_);
  }

  OperatorHandleImpl &operator=(OperatorHandleImpl &&other) {
    auto tmp(std::move(other));
    swap(tmp, *this);
    return *this;
  }

  OperatorHandleImpl &operator=(const OperatorHandleImpl &) = delete;

  void Register(DeviceType key, void *func_ptr) {
    if (HasDefinition(key)) {
      CHECK_EQ(func_ptr, kernel_table_.at(key))
          << "Find multiple definition of device type " << DeviceTypeName(key)
          << " in operator " << name_;
    }
    kernel_table_[key] = func_ptr;
  }

  void *Get(DeviceType key) const {
    if (HasDefinition(key)) {
      return kernel_table_.at(key);
    }
    return nullptr;
  }

  bool HasDefinition(DeviceType key) const {
    return kernel_table_.find(key) != kernel_table_.end();
  }

private:
  std::string name_;
  std::string doc_;
  std::unordered_map<DeviceType, void *> kernel_table_;
};

OperatorHandle::OperatorHandle() = default;

OperatorHandle::OperatorHandle(const std::string &name, const std::string &doc)
    : impl_(std::make_unique<OperatorHandleImpl>(name, doc)) {}

OperatorHandle::OperatorHandle(OperatorHandle &&other)
    : impl_(std::move(other.impl_)) {}

OperatorHandle &OperatorHandle::operator=(OperatorHandle &&other) {
  auto tmp(std::move(other));
  swap(tmp, *this);
  return *this;
}

void swap(OperatorHandle &op1, OperatorHandle &op2) {
  using std::swap;
  swap(op1.impl_, op2.impl_);
}

OperatorHandle::~OperatorHandle() = default;

void OperatorHandle::Register(DeviceType key, void *func_ptr) {
  impl_->Register(key, func_ptr);
}

bool OperatorHandle::HasDefinition(DeviceType key) const {
  return impl_->HasDefinition(key);
}

void *OperatorHandle::GetKernel(DeviceType key) const {
  return impl_->Get(key);
}

} // namespace tl
