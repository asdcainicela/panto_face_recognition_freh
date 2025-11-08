#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

struct StreamStats {
    double fps;
    int frames;
    int lost;
};

class StreamViewer {
private:
    std::string user, pass, ip, stream_type;
    int port;
    cv::Size display_size;
    int fps_interval;
    
    cv::VideoCapture cap;
    cv::Mat frame, display;

    int frames, lost;
    double current_fps;
    std::chrono::steady_clock::time_point start_main, start_fps;
    std::string window_name;
    std::string pipeline;

    bool reconnect();
    void update_fps();
    const StreamStats& get_stats();
    void print_stats();
    void print_final_stats();
    
    StreamStats cached_stats;

public:
    StreamViewer(const std::string& user, const std::string& pass, 
                 const std::string& ip, int port, const std::string& stream_type,
                 cv::Size display_size = cv::Size(640, 480), int fps_interval = 30);
    
    void run();
};