#include "tensorlite/allocator/cpu_allocator.h"
#include "tensorlite/buffer.h"
#include "tensorlite/device.h"
#include "tensorlite/utils/logging.h"

namespace tl {

void CpuBufferDeleter(Buffer *buffer) {
  CHECK_EQ(buffer->GetDevice().GetType(), DeviceType::kCPU);
  if (buffer->UntypedData()) {
    CpuMemoryAllocator::Free(buffer->GetDevice().GetId(), buffer->UntypedData(),
                             buffer->GetAlignment());
  }
}

} // namespace tl
