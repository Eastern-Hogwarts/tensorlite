#ifndef TENSORLITE_ALLOCATOR_ALLOCATOR_H_
#define TENSORLITE_ALLOCATOR_ALLOCATOR_H_

#include "tensorlite/buffer.h"
#include "tensorlite/device.h"
namespace tl {

/**
 * \brief
 *
 * \tparam DeviceTy
 */
template <typename DerivedTy> struct MemoryAllocator {
  /**
   * \brief
   *
   * \param device_id
   * \param size
   * \param align
   * \return void*
   */
  static void *Allocate(int device_id, size_t size, size_t align) {
    return DerivedTy::Allocate(device_id, size, align);
  }

  /**
   * \brief
   *
   * \param ptr
   * \param align
   */
  static void Free(int device_id, void *ptr, size_t align) {
    DerivedTy::Free(device_id, ptr, align);
  }
};

/**
 * \brief Create a new buffer object on a specific device
 *
 * \tparam Device The device where new buffer is allocated
 * \param device_id Id of the cpu device
 * \param size Size of this buffer
 * \param align alignment requirement of this buffer
 * \return std::shared_ptr<Buffer> A shared pointer pointing to the buffer
 * object
 *
 * \note The actual size allocated may be larger then the given size to meet the
 * alignment.
 */
template <DeviceType Device>
inline std::shared_ptr<Buffer> NewBuffer(int device_id, size_t size,
                                         size_t align);

} // namespace tl

#endif // TENSORLITE_ALLOCATOR_ALLOCATOR_H_
