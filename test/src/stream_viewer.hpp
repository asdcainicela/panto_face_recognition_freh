#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

class StreamViewer {
private:
    std::string user, pass, ip, stream_type;
    int port;
    cv::Size display_size;
    int fps_interval;
    
    cv::VideoCapture cap;
    cv::Mat frame, display;

    int w = display.cols;
    int h = display.rows;

    int frames, lost;
    std::chrono::steady_clock::time_point start_main, start_fps;
    std::string window_name;
    std::string pipeline;

    bool reconnect();
    void print_stats();
    void print_final_stats();

public:
    StreamViewer(const std::string& user, const std::string& pass, 
                 const std::string& ip, int port, const std::string& stream_type,
                 cv::Size display_size = cv::Size(640, 480), int fps_interval = 30);
    
    void run();
};
