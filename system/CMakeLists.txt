cmake_minimum_required(VERSION 3.10)
project(rtsp_viewer_recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SHARED_SOURCES
    src/utils.cpp
    src/stream_viewer.cpp
)

add_executable(rtsp_main src/rtsp.cpp ${SHARED_SOURCES})
target_link_libraries(rtsp_main ${OpenCV_LIBS} pthread spdlog::spdlog)
