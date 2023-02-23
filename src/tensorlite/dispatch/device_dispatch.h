#ifndef TENSORLITE_DISPATCH_DEVICE_DISPATCH_H_
#define TENSORLITE_DISPATCH_DEVICE_DISPATCH_H_

#include <memory>
#include <string>
#include <type_traits>

#include "tensorlite/device.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/singleton.h"

namespace tl {

/** TODO: docstring here
 * \brief
 *
 */
class OperatorHandle final {
public:
  OperatorHandle();
  OperatorHandle(const std::string &name, const std::string &doc = "");
  OperatorHandle(OperatorHandle &&);
  ~OperatorHandle();
  OperatorHandle &operator=(OperatorHandle &&);
  friend void swap(OperatorHandle &op1, OperatorHandle &op2);

  // only moveable
  OperatorHandle(const OperatorHandle &) = delete;
  OperatorHandle &operator=(const OperatorHandle &) = delete;

  void Register(DeviceType key, void *func_ptr);
  void *GetKernel(DeviceType key) const;
  bool HasDefinition(DeviceType key) const;

private:
  class OperatorHandleImpl;
  std::unique_ptr<OperatorHandleImpl> impl_;
};

/** TODO: docstring here
 * \brief
 *
 */
class DeviceDispatcher final : public GlobalSingleton<DeviceDispatcher> {
public:
  DeviceDispatcher();
  DeviceDispatcher(const DeviceDispatcher &) = delete;
  DeviceDispatcher(DeviceDispatcher &&) = delete;
  ~DeviceDispatcher();
  DeviceDispatcher &operator=(const DeviceDispatcher &) = delete;
  DeviceDispatcher &operator=(DeviceDispatcher &&) = delete;

  void DefineOperator(const std::string &name, const std::string &doc);
  bool HasOperator(const std::string &name);
  bool HasDefinition(const std::string &name, DeviceType key);
  void Register(const std::string &name, DeviceType key, void *func_ptr);

  const void *GetKernel(const std::string &name, DeviceType key) const;

private:
  class DeviceDispatcherImpl;
  std::unique_ptr<DeviceDispatcherImpl> impl_;
};

template <typename Return, typename... Args>
std::enable_if_t<std::is_void_v<Return>, void>
DeviceDispatchCall(const std::string &name, DeviceType key, Args... args) {
  using TypedFnPtr = Return (*)(Args...);
  const void *raw_fn_ptr =
      DeviceDispatcher::GetSingleton().GetKernel(name, key);
  CHECK_NE(raw_fn_ptr, nullptr) << "Cannot find definition of operator " << name
                                << " on device " << DeviceTypeName(key);
  TypedFnPtr fn = reinterpret_cast<TypedFnPtr>(raw_fn_ptr);
  fn(std::forward<Args>(args)...);
}

template <typename Return, typename... Args>
std::enable_if_t<!std::is_void_v<Return>, Return>
DeviceDispatchCall(const std::string &name, DeviceType key, Args... args) {
  using TypedFnPtr = Return (*)(Args...);
  const void *raw_fn_ptr =
      DeviceDispatcher::GetSingleton().GetKernel(name, key);
  CHECK_NE(raw_fn_ptr, nullptr) << "Cannot find definition of operator " << name
                                << " on device " << DeviceTypeName(key);
  TypedFnPtr fn = reinterpret_cast<TypedFnPtr>(raw_fn_ptr);
  return fn(std::forward<Args>(args)...);
}

/** TODO: docstring here
 * \brief
 *
 */
class StaticInitializer final {
public:
  using static_init_fn_t = void (*)();
  StaticInitializer(static_init_fn_t init_fn) { init_fn(); }
};

} // namespace tl

#ifdef __COUNTER__
#define OP_DEF_UID __COUNTER__
#else
#define OP_DEF_UID __LINE__
#endif

#define OP_DEF_CONCAT(a, b) a##b

#define OP_DEF(name) _OP_DEF(name, OP_DEF_UID)

#define _OP_DEF(name, uid)                                                     \
  static void OP_DEF_CONCAT(OPERATOR_DEF_##name, uid)();                       \
  static ::tl::StaticInitializer OP_DEF_CONCAT(                                \
      OPERATOR_DEF_STATIC_INIT_##name,                                         \
      uid)(&OP_DEF_CONCAT(OPERATOR_DEF_##name, uid));                          \
  void OP_DEF_CONCAT(OPERATOR_DEF_##name, uid)() {                             \
    ::tl::DeviceDispatcher::GetSingleton().DefineOperator(#name, "");          \
  }

#define OP_IMPL(name, device, func_name)                                       \
  _OP_IMPL(name, device, func_name, OP_DEF_UID)

#define _OP_IMPL(name, device, func_name, uid)                                 \
  static void OP_DEF_CONCAT(OPERATOR_IMPL_##name_##device, uid)();             \
  static ::tl::StaticInitializer OP_DEF_CONCAT(                                \
      OPERATOR_IMPL_STATIC_INIT_##name_##device,                               \
      uid)(&OP_DEF_CONCAT(OPERATOR_IMPL_##name_##device, uid));                \
  void OP_DEF_CONCAT(OPERATOR_IMPL_##name_##device, uid)() {                   \
    ::tl::DeviceDispatcher::GetSingleton().Register(                           \
        #name, ::tl::DeviceType::##device,                                     \
        reinterpret_cast<void *>(&func_name));                                 \
  }

#endif // TENSORLITE_DISPATCH_DEVICE_DISPATCH_H_
