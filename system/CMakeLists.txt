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

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION}")
include_directories(${CUDA_INCLUDE_DIRS})

# Buscar TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include
    /usr/local/include
    /opt/tensorrt/include
)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib
    /usr/local/lib
    /opt/tensorrt/lib
)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib
    /usr/local/lib
    /opt/tensorrt/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found\n"
        "  Headers searched in: /usr/include/aarch64-linux-gnu, /usr/include\n"
        "  Library searched in: /usr/lib/aarch64-linux-gnu, /usr/lib\n"
        "  Found headers: ${TENSORRT_INCLUDE_DIR}\n"
        "  Found library: ${TENSORRT_LIBRARY}")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "TensorRT Plugin: ${TENSORRT_PLUGIN_LIBRARY}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TENSORRT_INCLUDE_DIR})

add_library(panto_utils SHARED src/utils.cpp)
target_link_libraries(panto_utils 
    ${OpenCV_LIBS} 
    spdlog::spdlog
)
set_target_properties(panto_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

add_library(panto_config INTERFACE)
target_include_directories(panto_config INTERFACE include)

add_library(panto_draw SHARED src/draw_utils.cpp)
target_link_libraries(panto_draw 
    ${OpenCV_LIBS}
    spdlog::spdlog
)
set_target_properties(panto_draw PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

add_library(panto_stream SHARED src/stream_capture.cpp)
target_link_libraries(panto_stream 
    panto_utils
    panto_draw
    ${OpenCV_LIBS}
    spdlog::spdlog
    pthread
)
set_target_properties(panto_stream PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

add_library(panto_detector SHARED src/detector.cpp)
target_link_libraries(panto_detector 
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDA_LIBRARIES}
    cudart
    spdlog::spdlog
)
set_target_properties(panto_detector PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

add_executable(record test/record.cpp)
target_link_libraries(record 
    panto_stream
    panto_utils
)

add_executable(view test/view.cpp)
target_link_libraries(view 
    panto_stream
    panto_utils
    panto_draw
)

add_executable(record_view test/record_view.cpp)
target_link_libraries(record_view 
    panto_stream
    panto_utils
    panto_draw
)

add_executable(panto main.cpp)
target_link_libraries(panto 
    panto_stream
    panto_utils
    panto_draw
    panto_detector
)

add_executable(test_detector test/test_detector.cpp)
target_link_libraries(test_detector
    panto_detector
    panto_utils
    ${OpenCV_LIBS}
)

add_executable(test_detector_video test/test_detector_video.cpp)
target_link_libraries(test_detector_video
    panto_detector
    ${OpenCV_LIBS}
    spdlog::spdlog
)

install(TARGETS panto_utils panto_draw panto_stream panto_detector
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS record view record_view panto test_detector test_detector_video
    RUNTIME DESTINATION bin
)

install(DIRECTORY configs/ DESTINATION configs)
install(FILES config.toml DESTINATION .)

message(STATUS "========================================")
message(STATUS "PANTO Build Configuration")
message(STATUS "========================================")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "========================================")