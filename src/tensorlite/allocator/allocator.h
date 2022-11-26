#ifndef TENSORLITE_ALLOCATOR_ALLOCATOR_H_
#define TENSORLITE_ALLOCATOR_ALLOCATOR_H_

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

} // namespace tl

#endif // TENSORLITE_ALLOCATOR_ALLOCATOR_H_
