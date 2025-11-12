cmake_minimum_required(VERSION 3.10)
project(panto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find packages
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h 
    HINTS /usr/local/include /usr/include)
find_library(ONNXRUNTIME_LIBRARIES onnxruntime 
    HINTS /usr/local/lib /usr/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found!")
endif()

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# ============================================
# MODULAR SHARED LIBRARIES
# ============================================

# 1. Utils library
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

# 2. Config library (INTERFACE - solo headers)
add_library(panto_config INTERFACE)
target_include_directories(panto_config INTERFACE include)

# 3. Draw utilities library
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

# 4. Stream capture library
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

# 5. Detector library (RetinaFace)
add_library(panto_detector SHARED src/detector.cpp)
target_link_libraries(panto_detector 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    spdlog::spdlog
)
set_target_properties(panto_detector PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

# ============================================
# EXECUTABLES
# ============================================

# Test: record (solo grabar)
add_executable(record test/record.cpp)
target_link_libraries(record 
    panto_stream
    panto_utils
)

# Test: view (solo visualizar)
add_executable(view test/view.cpp)
target_link_libraries(view 
    panto_stream
    panto_utils
    panto_draw
)

# Test: record + view
add_executable(record_view test/record_view.cpp)
target_link_libraries(record_view 
    panto_stream
    panto_utils
    panto_draw
)

# Aplicación principal
add_executable(panto main.cpp)
target_link_libraries(panto 
    panto_stream
    panto_utils
    panto_draw
    panto_detector
)

# Test: face detection con imagen/webcam
add_executable(test_detector test/test_detector.cpp)
target_link_libraries(test_detector
    panto_detector
    panto_utils
    ${OpenCV_LIBS}
)

# Test: face detection con video MP4
add_executable(test_detector_video test/test_detector_video.cpp)
target_link_libraries(test_detector_video
    panto_detector
    ${OpenCV_LIBS}
    spdlog::spdlog
)

# Test: diagnose retinaface (low-level)
add_executable(diagnose_retinaface test/diagnose_retinaface.cpp)
target_link_libraries(diagnose_retinaface
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    spdlog::spdlog
)

# ============================================
# INSTALACIÓN
# ============================================
install(TARGETS panto_utils panto_draw panto_stream panto_detector
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS record view record_view panto test_detector test_detector_video diagnose_retinaface
    RUNTIME DESTINATION bin
)

install(DIRECTORY configs/ DESTINATION configs)
install(FILES config.toml DESTINATION .)

# ============================================
# BUILD INFO
# ============================================
message(STATUS "================================")
message(STATUS "PANTO Modular Build")
message(STATUS "================================")
message(STATUS "Libraries:")
message(STATUS "  - libpanto_utils.so")
message(STATUS "  - libpanto_draw.so")
message(STATUS "  - libpanto_stream.so")
message(STATUS "  - libpanto_detector.so")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - record")
message(STATUS "  - view")
message(STATUS "  - record_view")
message(STATUS "  - panto")
message(STATUS "  - test_detector")
message(STATUS "  - test_detector_video (NEW)")
message(STATUS "  - diagnose_retinaface")
message(STATUS "")
message(STATUS "Build: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "================================")