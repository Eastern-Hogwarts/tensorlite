#include "tensorlite/allocator/cpu_allocator.h"
#include "tensorlite/allocator/utils.h"
#include "tensorlite/utils/logging.h"
#include <cstdlib>
#include <new>

namespace tl {

void *CpuMemoryAllocator::Allocate(int device_id, size_t size, size_t align) {
  void *alloc_ptr = nullptr;
  size = utils::ceil_align(size, align);
#if _MSC_VER || defined(__MINGW32__) // MSVC, _WIN32 if for platform check
  // MSVC does not provide std::aligned_alloc, see Notes in
  // https://en.cppreference.com/w/cpp/memory/c/aligned_alloc
  alloc_ptr = _aligned_malloc(size, align);
#else
  // std aligned_alloc, if implemented by posix_memalign, align
  // should be 2^N or multiple of sizeof(void*)
  alloc_ptr = std::aligned_alloc(align, size);
#endif
  if (!alloc_ptr)
    throw std::bad_alloc();
  return alloc_ptr;
}

void CpuMemoryAllocator::Free(int device_id, void *ptr, size_t align) {
  CHECK_NE(ptr, nullptr);
#if _MSC_VER || defined(__MINGW32__) // MSVC, _WIN32 if for platform check
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

std::shared_ptr<Buffer> NewCpuBuffer(int device_id, size_t size, size_t align) {
  void *ptr = CpuMemoryAllocator::Allocate(device_id, size, align);
  if (ptr) {
    return std::make_shared<Buffer>(ptr, size, align,
                                    Device(device_id, DeviceType::kCPU),
                                    &CpuBufferDeleter);
  } else {
    return nullptr;
  }
}

} // namespace tl
