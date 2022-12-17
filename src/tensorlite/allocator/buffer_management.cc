#include "tensorlite/allocator/allocator.h"

#include "tensorlite/device/data_transfer.h"
#include "tensorlite/device/utils.h"

namespace tl {

std::shared_ptr<Buffer> CopyBuffer(std::shared_ptr<Buffer> buffer) {
  std::shared_ptr<Buffer> new_buffer;
  auto device = buffer->GetDevice();
  DEVICE_SWITCH(device.GetType(), buffer_device, {
    new_buffer = NewBuffer<buffer_device>(device.GetId(), buffer->GetSize(),
                                          buffer->GetAlignment());
    DataTransfer<buffer_device, buffer_device>(
        buffer->UntypedData(), new_buffer->UntypedData(), buffer->GetSize(),
        device.GetId(), device.GetId());
  });
  return new_buffer;
}

} // namespace tl
