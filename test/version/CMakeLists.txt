cmake_minimum_required(VERSION 3.16)
project(cpp_version_check LANGUAGES CXX)

# Try C++20 first, fallback to C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF) # Don't fail if not available
set(CMAKE_CXX_EXTENSIONS OFF)

# Fallback if compiler doesn't support C++20
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        set(CMAKE_CXX_STANDARD 17)
        message(WARNING "GCC < 10.0 detected, falling back to C++17")
    endif()
endif()

add_executable(check_version check_cpp_version.cpp)

# Print CMake info
message(STATUS "=== CMake Configuration ===")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")

# Check for OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
else()
    message(STATUS "✗ OpenCV NOT found")
endif()

# Check for CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "✓ CUDA ${CUDA_VERSION} found")
else()
    message(STATUS "✗ CUDA NOT found")
endif()

# Check for TensorRT (common paths for Jetson)
if(EXISTS "/usr/src/tensorrt/include/NvInfer.h")
    message(STATUS "✓ TensorRT found (Jetson path)")
elseif(EXISTS "/usr/include/NvInfer.h")
    message(STATUS "✓ TensorRT found (system path)")
else()
    message(STATUS "✗ TensorRT NOT found")
endif()