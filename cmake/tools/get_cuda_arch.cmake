#######################################################
# Get the compute capability of the first cuda device.
# Usage:
#   dgl_option()
# These variables will be set:
# - CUDA_DEVICE_CC: the compute capability of the first cuda device.
function(get_cuda_arch)
  execute_process(COMMAND nvidia-smi "--query-gpu=compute_cap" "--format=csv,noheader" OUTPUT_VARIABLE _CUDA_DEVICE_CC)
  string(REGEX MATCH "([0-9]+)\.([0-9]+)" _FIRST_CUDA_DEVICE_CC ${_CUDA_DEVICE_CC})
  message(STATUS "Get compute capability of the first cuda device: ${_FIRST_CUDA_DEVICE_CC}")
  set(CUDA_DEVICE_CC "${CMAKE_MATCH_1}${CMAKE_MATCH_2}" PARENT_SCOPE)
endfunction(get_cuda_arch)
