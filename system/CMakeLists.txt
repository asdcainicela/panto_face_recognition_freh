cmake_minimum_required(VERSION 3.10)
project(panto)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ========== DEPENDENCIES ==========
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include /opt/tensorrt/include
)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib /opt/tensorrt/lib
)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib /opt/tensorrt/lib
)
find_library(TENSORRT_ONNX nvonnxparser
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib /opt/tensorrt/lib
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TENSORRT_INCLUDE_DIR})

# ========== LIBRARIES ==========

add_library(panto_utils SHARED src/utils.cpp)
target_link_libraries(panto_utils ${OpenCV_LIBS} spdlog::spdlog)

add_library(panto_draw SHARED src/draw_utils.cpp)
target_link_libraries(panto_draw ${OpenCV_LIBS} spdlog::spdlog)

add_library(panto_stream SHARED src/stream_capture.cpp)
target_link_libraries(panto_stream 
    panto_utils
    panto_draw
    ${OpenCV_LIBS}
    spdlog::spdlog
    pthread
)

# ========== CUDA KERNELS ==========
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -O3 
    -Xcompiler -fPIC 
    -gencode arch=compute_87,code=sm_87
    --expt-relaxed-constexpr
)

cuda_add_library(panto_cuda_kernels STATIC
    src/cuda_kernels.cu
    OPTIONS -Xcompiler -fPIC
)

# ========== DETECTOR OPTIMIZADO ==========
add_library(panto_detector_opt SHARED 
    src/detector_optimized.cpp
)

target_link_libraries(panto_detector_opt
    panto_cuda_kernels
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${TENSORRT_ONNX}
    cudart
    spdlog::spdlog
)

# ========== EXECUTABLES ==========

# Main App (hace todo)
add_executable(panto main.cpp)
target_link_libraries(panto 
    panto_stream
    panto_utils
    panto_draw
    panto_detector_opt
)

# Benchmark (solo para tests)
add_executable(benchmark test/benchmark_detector.cpp)
target_link_libraries(benchmark
    panto_detector_opt
    ${OpenCV_LIBS}
    spdlog::spdlog
)

# ========== INSTALL ==========
install(TARGETS panto_utils panto_draw panto_stream panto_detector_opt panto_cuda_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS panto benchmark
    RUNTIME DESTINATION bin
)

install(FILES config.toml DESTINATION .)

# ========== STATUS ==========
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION} (sm_87)")
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "cuBLAS: ${CUDA_CUBLAS_LIBRARIES}")