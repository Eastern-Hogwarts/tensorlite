#ifndef TENSORLITE_ALLOCATOR_CUDA_ALLOCATOR_H_
#define TENSORLITE_ALLOCATOR_CUDA_ALLOCATOR_H_

#include <memory>

#include "tensorlite/allocator/allocator.h"
#include "tensorlite/buffer.h"

namespace tl {

/**
 * \brief
 *
 */
struct CudaMemoryAllocator : public MemoryAllocator<CudaMemoryAllocator> {
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
 * \brief Delete a buffer object reside on cuda device
 *
 * \param buffer
 */
void CudaBufferDeleter(Buffer *buffer);

/**
 * \brief Create a new buffer object on cuda device
 *
 * \param device_id Id of the cuda device
 * \param size Size of this buffer
 * \param align alignment requirement of this buffer
 * \return std::shared_ptr<Buffer> A shared pointer pointing to the buffer
 * object
 */
std::shared_ptr<Buffer> NewCudaBuffer(int device_id, size_t size, size_t align);

} // namespace tl

#endif // TENSORLITE_ALLOCATOR_CUDA_ALLOCATOR_H_
