cmake_minimum_required(VERSION 3.18)
project(OpenCVCUDATensorRTTests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
)

# Link directories
link_directories(
    ${CUDA_LIBRARY_DIRS}
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
)

# Executable for comprehensive test
add_executable(test_all test_all.cpp)

# Link libraries
target_link_libraries(test_all
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    nvparsers
    cudart
    pthread
    dl
)

# Compiler flags
target_compile_options(test_all PRIVATE
    -Wall
    -Wextra
    -O3
)

# Install target
install(TARGETS test_all DESTINATION /opt/tests)

# Print configuration
message(STATUS "=================================")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "=================================")