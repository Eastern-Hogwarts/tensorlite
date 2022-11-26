#ifndef TENSORLITE_BUFFER_H_
#define TENSORLITE_BUFFER_H_

#include "tensorlite/device.h"
#include <cassert>
#include <cstdlib>

namespace tl {

class Buffer;
using BufferDeleter = void(Buffer *);

/**
 * \brief A buffer object holding the underlying data buffer of tensors.
 */
class Buffer {
public:
  /**
   * \brief Construct a new Buffer object
   */
  Buffer(){};

  /**
   * \brief Destroy the Buffer object
   */
  ~Buffer() {
    if (data_) {
      deleter_(this);
      data_ = nullptr;
    }
  }

  explicit Buffer(void *data, size_t size, size_t align, Device device,
                  BufferDeleter *deleter)
      : data_(data), size_(size), align_(align), device_(device),
        deleter_(deleter) {
    if (data_) {
      assert(deleter_ != nullptr);
    }
  }

  // no copy/move is allowed, buffer should be managed by shared_ptr
  Buffer(const Buffer &) = delete;
  Buffer(Buffer &&) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer &operator=(Buffer &&) = delete;

  /**
   * \brief Return true if the underlying buffer is not empty
   */
  operator bool() const { return data_ != nullptr; }

  /**
   * \brief Return a pointer with specific type pointing to the underlying
   * buffer
   *
   * \tparam DataTy The data type of the return pointer.
   * \return const DataTy*
   */
  template <typename DataTy> const DataTy *TypedPtr() const {
    return static_cast<DataTy *>(data_);
  }

  /**
   * \brief Return a pointer with specific type pointing to the underlying
   * buffer
   *
   * \tparam DataTy The data type of the return pointer.
   * \return DataTy*
   */
  template <typename DataTy> DataTy *TypedPtr() {
    return static_cast<DataTy *>(data_);
  }

  /**
   * \brief Return a pointer pointing to the underlying buffer
   *
   * \return const void*
   */
  const void *UntypedData() const { return data_; }

  /**
   * \brief Return a pointer pointing to the underlying buffer
   *
   * \return void*
   */
  void *UntypedData() { return data_; }

  /**
   * \brief Get the device of this buffer
   *
   * \return Device
   */
  Device GetDevice() const { return device_; }

  /**
   * \brief Get the size of this buffer
   *
   * \return size_t
   */
  size_t GetSize() const { return size_; }

  /**
   * \brief Get the alignment of this buffer
   *
   * \return size_t
   */
  size_t GetAlignment() const { return align_; }

private:
  void *data_ = nullptr;
  size_t size_ = 0;
  size_t align_ = 0;
  Device device_;
  BufferDeleter *deleter_ = nullptr;
};

} // namespace tl

#endif // TENSORLITE_BUFFER_H_
