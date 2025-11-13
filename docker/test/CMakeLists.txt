cmake_minimum_required(VERSION 3.20)
project(verify_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

include_directories(
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(test_all test_all.cpp)

target_link_libraries(test_all
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)