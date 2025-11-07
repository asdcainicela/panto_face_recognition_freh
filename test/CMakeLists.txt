cmake_minimum_required(VERSION 3.10)
project(JetsonCppDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rtsp_main src/rtsp_main.cpp)
target_link_libraries(rtsp_main ${OpenCV_LIBS} pthread)

add_executable(rtsp_sub src/rtsp_sub.cpp)
target_link_libraries(rtsp_sub ${OpenCV_LIBS} pthread)