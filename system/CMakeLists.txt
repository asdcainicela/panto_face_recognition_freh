# ============= CMakeLists.txt - ESTRUCTURA MODULAR =============

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

# ===== FIND PACKAGES =====
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)
find_package(SQLite3 REQUIRED)

# ===== INCLUDE DIRECTORIES =====
# ⭐ Esto permite usar #include "module/file.hpp"
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${SQLite3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# ===== TENSORRT =====
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
include_directories(${TENSORRT_INCLUDE_DIR})

# ===== CUDA SETUP =====
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    -Xcompiler -fPIC
    -gencode arch=compute_87,code=sm_87
    --expt-relaxed-constexpr
)

# ===== CORE LIBRARIES =====

add_library(panto_utils SHARED 
    src/core/utils.cpp
)
target_link_libraries(panto_utils 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)

add_library(panto_draw SHARED 
    src/core/draw_utils.cpp
)
target_link_libraries(panto_draw 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)

# ===== DETECTION MODULE =====

cuda_add_library(panto_cuda_kernels STATIC
    src/detection/cuda_kernels.cu
    OPTIONS -Xcompiler -fPIC
)

add_library(panto_detector_opt SHARED 
    src/detection/detector_optimized.cpp
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

# ===== TRACKING MODULE =====

add_library(panto_tracker SHARED 
    src/tracking/tracker.cpp
)
target_link_libraries(panto_tracker 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)

# ===== RECOGNITION MODULE =====

add_library(panto_recognizer SHARED 
    src/recognition/recognizer.cpp
)
target_link_libraries(panto_recognizer
    panto_cuda_kernels
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${TENSORRT_ONNX}
    cudart
    spdlog::spdlog
)

add_library(panto_emotion SHARED 
    src/recognition/emotion_recognizer.cpp
)
target_link_libraries(panto_emotion
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${TENSORRT_ONNX}
    cudart
    spdlog::spdlog
)

add_library(panto_age_gender SHARED 
    src/recognition/age_gender_predictor.cpp
)
target_link_libraries(panto_age_gender
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${TENSORRT_ONNX}
    cudart
    spdlog::spdlog
)

# ===== DATABASE MODULE =====

add_library(panto_thread_pool SHARED 
    src/database/thread_pool.cpp
)
target_link_libraries(panto_thread_pool 
    pthread 
    spdlog::spdlog
)

add_library(panto_vector_index SHARED 
    src/database/vector_index.cpp
)
target_link_libraries(panto_vector_index 
    pthread 
    spdlog::spdlog
)

add_library(panto_database SHARED 
    src/database/face_database.cpp
)
target_link_libraries(panto_database 
    ${SQLite3_LIBRARIES} 
    spdlog::spdlog
)

add_library(panto_db_manager SHARED 
    src/database/face_db_manager.cpp
)
target_link_libraries(panto_db_manager
    panto_thread_pool
    panto_vector_index
    ${SQLite3_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
    spdlog::spdlog
)

# ===== STREAMING MODULE =====

add_library(panto_stream SHARED 
    src/streaming/stream_capture.cpp
)
target_link_libraries(panto_stream 
    panto_utils
    panto_draw
    ${OpenCV_LIBS}
    spdlog::spdlog
    pthread
)

# ===== MAIN EXECUTABLE =====

add_executable(panto main.cpp)
target_link_libraries(panto
    panto_stream
    panto_utils
    panto_draw
    panto_detector_opt
    panto_tracker
    panto_recognizer
    panto_emotion
    panto_age_gender
    panto_database
    panto_db_manager
    panto_thread_pool
    panto_vector_index
)

# ===== TEST EXECUTABLES =====

add_executable(test_validator test/test_validator.cpp)
target_link_libraries(test_validator
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDA_LIBRARIES}
    cudart
    spdlog::spdlog
)

add_executable(benchmark test/benchmark_detector.cpp)
target_link_libraries(benchmark 
    panto_detector_opt 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)

add_executable(test_recognizer test/test_recognizer.cpp)
target_link_libraries(test_recognizer 
    panto_recognizer 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)

# ===== INSTALL =====

install(TARGETS
    panto_utils
    panto_draw
    panto_stream
    panto_detector_opt
    panto_tracker
    panto_recognizer
    panto_emotion
    panto_age_gender
    panto_database
    panto_db_manager
    panto_thread_pool
    panto_vector_index
    panto_cuda_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS panto benchmark test_recognizer test_validator RUNTIME DESTINATION bin)
install(FILES config.toml DESTINATION .)

# ===== STATUS MESSAGES =====
message(STATUS "=== PANTO Build Configuration ===")
message(STATUS "opencv: ${OpenCV_VERSION}")
message(STATUS "cuda: ${CUDA_VERSION} (sm_87)")
message(STATUS "tensorrt: ${TENSORRT_LIBRARY}")
message(STATUS "cublas: ${CUDA_CUBLAS_LIBRARIES}")
message(STATUS "sqlite3: ${SQLite3_LIBRARIES}")
message(STATUS "")
message(STATUS "=== Module Organization ===")
message(STATUS "Core: utils, draw_utils")
message(STATUS "Detection: detector, cuda_kernels")
message(STATUS "Tracking: tracker")
message(STATUS "Recognition: recognizer, emotion, age_gender")
message(STATUS "Database: face_db, db_manager, vector_index, thread_pool")
message(STATUS "Streaming: stream_capture")