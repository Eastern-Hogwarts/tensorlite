#ifndef TENSORLITE_ALLOCATOR_CPU_ALLOCATOR_H_
#define TENSORLITE_ALLOCATOR_CPU_ALLOCATOR_H_

#include <memory>

#include "tensorlite/allocator/allocator.h"
#include "tensorlite/buffer.h"

namespace tl {

/**
 * \brief
 *
 */
struct CpuMemoryAllocator : public MemoryAllocator<CpuMemoryAllocator> {
  /**
   * \brief
   *
   * \param device_id
   * \param size
   * \param align
   * \return void*
   */
  static void *Allocate(int device_id, size_t size, size_t align);

  /**
   * \brief
   *
   * \param ptr
   * \param align
   */
  static void Free(int device_id, void *ptr, size_t align);
};

/**
 * \brief Delete a buffer object reside on cpu
 *
 * \param buffer
 */
void CpuBufferDeleter(Buffer *buffer);

/**
 * \brief Create a new buffer object on cpu
 *
 * \param device_id Id of the cpu device
 * \param size Size of this buffer
 * \param align alignment requirement of this buffer
 * \return std::shared_ptr<Buffer> A shared pointer pointing to the buffer
 * object
 *
 * \note The actual size allocated may be larger then the given size to meet the
 * alignment.
 */
std::shared_ptr<Buffer> NewCpuBuffer(int device_id, size_t size, size_t align);

} // namespace tl

#endif // TENSORLITE_ALLOCATOR_CPU_ALLOCATOR_H_
