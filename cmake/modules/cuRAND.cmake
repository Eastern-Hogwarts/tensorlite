message(STATUS "Finding cuRAND...")

# find cuRAND
find_path(_CURAND_INCLUDE_PATH
          curand.h
          HINTS
            ${CURAND_INCLUDE_PATH}
            $ENV{CURAND_INCLUDE_PATH}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            $ENV{CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${CUDA_TOOLKIT_ROOT_DIR}
            $ENV{CUDA_PATH}
          PATH_SUFFIXES
            include
          )
find_library(_CURAND_LIBRARY_PATH
            curand
            HINTS
              ${CURAND_LIBRARY_PATH}
              $ENV{CURAND_LIBRARY_PATH}
              ${CUDA_TOOLKIT_ROOT_DIR}
              $ENV{CUDA_PATH}
            PATH_SUFFIXES
              lib64
              lib/x64
              lib
              lib/Win32
            )

if(_CURAND_INCLUDE_PATH AND _CURAND_LIBRARY_PATH)
  message(STATUS "Find cuRAND INCLUDE DIR at ${_CURAND_INCLUDE_PATH}")
  message(STATUS "Find cuRAND LIB DIR at: ${_CURAND_LIBRARY_PATH}")
else()
  message(FATAL_ERROR "cuRAND not found, configuration failed")
endif()

set(CURAND_INCLUDE_PATH ${_CURAND_INCLUDE_PATH})
set(CURAND_LIBRARY ${_CURAND_LIBRARY_PATH})
