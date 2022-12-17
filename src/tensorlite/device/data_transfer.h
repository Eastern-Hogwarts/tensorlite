/**
 * \file data_transfer.h
 * \brief Manage data transfer between devices
 */
#ifndef TENSORLITE_DEVICE_DATA_TRANSFER_H_
#define TENSORLITE_DEVICE_DATA_TRANSFER_H_

#include "tensorlite/device.h"
#include "tensorlite/utils/logging.h"
#include <cstdint>
#include <cstdlib>

namespace tl {

template <DeviceType SrcDev, DeviceType DstDev>
void DataTransfer(const void *src_ptr, void *dst_ptr, size_t size,
                  int src_id = 0, int dst_id = 0, void *stream = nullptr);

} // namespace tl

#endif // TENSORLITE_DEVICE_DATA_TRANSFER_H_
