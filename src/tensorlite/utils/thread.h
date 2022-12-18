#ifndef TENSORLITE_UTILS_THREAD_H_
#define TENSORLITE_UTILS_THREAD_H_

#include <cstdint>
#include <mutex>

namespace tl {

// Get a unique integer ID representing this thread.
inline uint32_t GetThreadId() {
  static int num_threads = 0;
  static std::mutex mutex;
  static thread_local int id = -1;

  if (id == -1) {
    std::lock_guard<std::mutex> guard(mutex);
    id = num_threads;
    num_threads++;
  }
  return id;
}

} // namespace tl

#endif // TENSORLITE_UTILS_THREAD_H_
