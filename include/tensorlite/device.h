#ifndef TENSORLITE_DEVICE_H_
#define TENSORLITE_DEVICE_H_

#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace tl {

enum class DeviceType { kCPU, kCUDA };

inline constexpr std::string_view DeviceTypeName(DeviceType type) {
  switch (type) {
  case DeviceType::kCPU:
    return "cpu";
  case DeviceType::kCUDA:
    return "cuda";
  default:
    return "unknown";
  }
}

/**
 * \brief Device class
 *
 */
class Device {
public:
  /**
   * \brief Default constructor of Device.
   *
   * \note Defalut device is CPU device
   */
  Device() : id_(0), type_(DeviceType::kCPU) {}

  /**
   * \brief Construct a new Device object
   *
   * \param id device id
   * \param type device type
   */
  explicit Device(int id, DeviceType type) : id_(id), type_(type) {}

  Device(const Device &other) : id_(other.id_), type_(other.type_) {}

  Device(Device &&other) : id_(other.id_), type_(other.type_) {}

  Device &operator=(const Device &other) {
    using std::swap;
    auto tmp(other);
    swap(*this, tmp);
    return *this;
  }

  Device &operator=(Device &&other) {
    using std::swap;
    auto tmp(std::move(other));
    swap(*this, tmp);
    return *this;
  }

  bool operator==(const Device &other) const {
    return other.id_ == id_ && other.type_ == type_;
  }

  bool operator!=(const Device &other) const { return !(*this == other); }

  ~Device() {}

  /**
   * \brief Return the name of this device object
   *
   * \return const std::string
   */
  const std::string Name() const {
    std::string type_name(DeviceTypeName(type_));
    return type_name + "_" + std::to_string(id_);
  }

  /**
   * \brief Get the id of this device
   *
   * \return int
   */
  int GetId() const { return id_; }

  /**
   * \brief Get the device type of this device
   *
   * \return DeviceType
   */
  DeviceType GetType() const { return type_; }

  friend std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.Name();
    return os;
  }

  /**
   * \brief Construct a cuda device with the given id
   *
   * \param id cuda device id. (default: 0)
   * \return Device
   */
  static Device CudaDevice(int id = 0) { return Device(id, DeviceType::kCUDA); }

  /**
   * \brief Construct a host device with the given id
   *
   * \param id host processor id. (default: 0)
   * \return Device
   */
  static Device CpuDevice(int id = 0) { return Device(id, DeviceType::kCPU); }

  /**
   * \brief Get the default device (CPU:0)
   *
   * \return Device
   */
  static Device DefaultDevice() { return CpuDevice(); }

private:
  int id_;
  DeviceType type_;
};

} // namespace tl

#endif // TENSORLITE_DEVICE_H_
