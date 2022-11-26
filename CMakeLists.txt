cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TENSORLITE LANGUAGES CXX VERSION 0.1)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "The compute capability id of this cuda arch")

# use c++17 standard
set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "Cannot find CUDA toolkit in this machine")
endif()

# use shared cudart
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
message(STATUS "Use CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID}:${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Compiler Path: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda") # enable lambda in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # enable constexpr expression in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build with CUDA Computation Capability: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB_RECURSE TENSORLITE_SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/tensorlite/*.cc")
file(GLOB_RECURSE TENSORLITE_CU_LIST "${CMAKE_SOURCE_DIR}/src/tensorlite/*.cu")
list(APPEND TENSORLITE_SOURCE_LIST ${TENSORLITE_CU_LIST})
add_library(tensorlite SHARED ${TENSORLITE_SOURCE_LIST})
set_target_properties(tensorlite PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tensorlite PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(tensorlite
                           PUBLIC "${CMAKE_SOURCE_DIR}/include"
                           PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(tensorlite PRIVATE -DTENSORLITE_EXPORTS)
target_compile_features(tensorlite PUBLIC cxx_std_17)
target_link_libraries(tensorlite CUDART)
