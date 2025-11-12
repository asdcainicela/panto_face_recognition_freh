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

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================
# MODULAR SHARED LIBRARIES (1 .so per module)
# ============================================

# 1. Utils library (base - sin dependencias)
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

# 2. Config library (solo headers, pero incluido por si crece)
add_library(panto_config INTERFACE)
target_include_directories(panto_config INTERFACE include)

# 3. Draw utilities library (depende de config)
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

# 4. Stream capture library (depende de utils, config, draw)
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

# ============================================
# EXECUTABLES - Linkean solo lo que necesitan
# ============================================

# Test: Solo grabar (necesita stream + utils)
add_executable(record test/record.cpp)
target_link_libraries(record 
    panto_stream
    panto_utils
)

# Test: Solo visualizar (necesita todo)
add_executable(view test/view.cpp)
target_link_libraries(view 
    panto_stream
    panto_utils
    panto_draw
)

# Test: Grabar + visualizar
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
)

# ============================================
# INSTALACIÓN
# ============================================
install(TARGETS panto_utils panto_draw panto_stream
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS record view record_view panto
    RUNTIME DESTINATION bin
)

# ============================================
# BUILD INFO
# ============================================
message(STATUS "=================================")
message(STATUS "PANTO Modular Build")
message(STATUS "=================================")
message(STATUS "Libraries:")
message(STATUS "  - libpanto_utils.so    (base utilities)")
message(STATUS "  - libpanto_draw.so     (drawing functions)")
message(STATUS "  - libpanto_stream.so   (stream capture)")
message(STATUS "")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "=================================")