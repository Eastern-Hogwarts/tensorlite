cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TENSORLITE LANGUAGES CXX VERSION 0.1)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "The compute capability id of this cuda arch")
option(BUILD_TEST "whether build test for ths project" ON)
option(ENABLE_OPENMP "whether use openmp" ON)

# use c++17 standard
set(CMAKE_CXX_STANDARD 17)

# msvc does not have rpath, we need to put all things into one directory
# TODO: build libraries in lib directory and copy to bin directory
if(MSVC)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

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
                           PUBLIC "${CMAKE_SOURCE_DIR}/3rdparty/FP16/include"
                           PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(tensorlite PRIVATE -DTENSORLITE_EXPORTS)
target_compile_features(tensorlite PUBLIC cxx_std_17)
target_link_libraries(tensorlite CUDART)

if (ENABLE_OPENMP)
  target_compile_definitions(tensorlite PRIVATE -DUSE_OPENMP)
  message(STATUS "Enable OpenMP")
  if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif ()
endif ()

if(BUILD_TEST)
  message(STATUS "Build with unittest")

  # By default, new Visual Studio projects link the C runtimes dynamically but GoogleTest links them statically.
  # We force google test to link C runtimes dynamically here.
  # see https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()

  add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/googletest")
  enable_testing()
  include(GoogleTest)
  message(STATUS "Find google test in ${gtest_SOURCE_DIR}")

  # unit tests
  add_subdirectory("${CMAKE_SOURCE_DIR}/tests/cpp")
endif()

