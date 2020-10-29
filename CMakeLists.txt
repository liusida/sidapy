cmake_minimum_required(VERSION 3.12.0)
project(sidapy LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

include_directories(src)
include_directories("pybind11/include")
file( GLOB SRC_cu src/*/*.cu )
file( GLOB SRC_cpp src/*/*.cpp )

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} src/pybind_main.cpp ${SRC_cpp} ${SRC_cu})
# add_executable(tests tests/test.cpp ${SRC_cpp} ${SRC_cu})

# Python
# find_package (Python COMPONENTS Interpreter Development)
# if (Python_FOUND)
#   include_directories(${Python_INCLUDE_DIRS})
#   # target_link_directories(tests PRIVATE ${Python_LIBRARIES})
# endif()

find_package(OpenCV)
if (OpenCV_FOUND)
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif()

# CUDA options
find_package(CUDA) # for CUDA_SELECT_NVCC_ARCH_FLAGS function
# if one want to force use certain Gpu Architecture, one can add `CUDA_ARCH` like `cmake -DCUDA_ARCH=arch=compute_61,code=sm_61 ..`
if (NOT DEFINED CUDA_ARCH)
  CUDA_SELECT_NVCC_ARCH_FLAGS(RET_ARCH_FLAGS)
  message("CUDA_SELECT_NVCC_ARCH_FLAGS returns: " ${RET_ARCH_FLAGS})
  list(LENGTH RET_ARCH_FLAGS FLAG_LENGTH)
  message("Flag length: " ${FLAG_LENGTH})
  if (FLAG_LENGTH GREATER 3)
    # CUDA_SELECT_NVCC_ARCH_FLAGS resulted more than 3 items, which is probably GPU not found error.
    message(FATAL_ERROR "Nvidia GPU not found. Abort.")
  endif()
  list(GET RET_ARCH_FLAGS -1 CUDA_ARCH) # use newest architecture
endif()
message("Using GPU Architecture: " ${CUDA_ARCH})
set(ARCH_FLAGS "-gencode" ${CUDA_ARCH})
list(APPEND CMAKE_CUDA_FLAGS -gencode=${CUDA_ARCH})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  if (NOT DEFINED CUDA_DEBUG)
    set(CUDA_DEBUG "-G or not" ON)
  endif()
endif(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#-G means debug info in GPU, so you can step in GPU kernels
if (CUDA_DEBUG)
  message("Set CUDA_DEBUG to ON.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-G ${ARCH_FLAGS}>)
else (CUDA_DEBUG)
  message("Set CUDA_DEBUG to OFF.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS}>)
endif(CUDA_DEBUG)

target_compile_options(${PROJECT_NAME} PUBLIC ${CUDA_OPTIONS})
# target_compile_options(tests PUBLIC ${CUDA_OPTIONS})

set(CMAKE_VERBOSE_MAKEFILE off) #turn on if we want to debug in compilation.
