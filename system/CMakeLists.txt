cmake_minimum_required(VERSION 3.10)
project(panto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find packages
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Shared library - Stream utilities
set(STREAM_LIB_SOURCES
    src/utils.cpp
    src/stream_capture.cpp
    src/draw_utils.cpp
)

add_library(stream_lib SHARED ${STREAM_LIB_SOURCES})
target_link_libraries(stream_lib ${OpenCV_LIBS} spdlog::spdlog)

# Test executables
add_executable(record test/record.cpp)
target_link_libraries(record stream_lib pthread)

add_executable(view test/view.cpp)
target_link_libraries(view stream_lib pthread)

add_executable(record_view test/record_view.cpp)
target_link_libraries(record_view stream_lib pthread)

# Main panto application
add_executable(panto src/main.cpp)
target_link_libraries(panto stream_lib pthread)