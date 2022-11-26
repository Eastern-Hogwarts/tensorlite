#include "tensorlite/allocator/cuda_allocator.h"
#include "tensorlite/buffer.h"
#include "tensorlite/device.h"
#include "tensorlite/utils/logging.h"

namespace tl {

void CudaBufferDeleter(Buffer *buffer) {
  CHECK_EQ(buffer->GetDevice().GetType(), DeviceType::kCUDA);
  if (buffer->UntypedData()) {
    CudaMemoryAllocator::Free(buffer->GetDevice().GetId(),
                              buffer->UntypedData(), buffer->GetAlignment());
  }
}

} // namespace tl
