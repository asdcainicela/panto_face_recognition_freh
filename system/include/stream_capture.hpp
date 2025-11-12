#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <atomic>

struct StreamStats {
    double fps;
    int frames;
    int lost;
    cv::Size resolution;
};

class StreamCapture {
private:
    std::string user, pass, ip, stream_type;
    int port;
    int fps_interval;
    
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    
    int frames, lost;
    double current_fps;
    std::chrono::steady_clock::time_point start_time, start_fps;
    std::string pipeline;
    
    bool recording_enabled;
    std::string output_filename;
    std::atomic<bool>* stop_signal;
    int max_duration;
    
    StreamStats cached_stats;
    
    bool reconnect();
    void update_fps();
    bool init_recording();
    void stop_recording();

public:
    StreamCapture(const std::string& user, const std::string& pass,
                  const std::string& ip, int port, const std::string& stream_type);
    ~StreamCapture();
    
    void enable_recording(const std::string& output_dir = "../videos");
    void set_max_duration(int seconds);
    void set_stop_signal(std::atomic<bool>* signal);
    void set_fps_interval(int interval);
    
    bool open();
    bool read(cv::Mat& frame);
    void release();
    
    const StreamStats& get_stats();
    void print_stats() const;
    void print_final_stats() const;
    
    bool is_recording() const { return recording_enabled && writer.isOpened(); }
};