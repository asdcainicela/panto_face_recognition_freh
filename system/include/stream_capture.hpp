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
    std::string window_name;
    
    bool recording_enabled;
    bool viewing_enabled;
    cv::Size display_size;
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
    
    // Configuration
    void enable_recording(const std::string& output_dir = "../videos");
    void enable_viewing(const cv::Size& size = cv::Size(640, 480));
    void set_max_duration(int seconds);
    void set_stop_signal(std::atomic<bool>* signal);
    void set_fps_interval(int interval);
    
    // Basic operations
    bool open();
    bool read(cv::Mat& frame);
    void release();
    
    // High-level operation (like StreamViewer::run())
    void run();
    
    // Stats
    const StreamStats& get_stats();
    void print_stats() const;
    void print_final_stats() const;
    
    bool is_recording() const { return recording_enabled && writer.isOpened(); }
    bool is_viewing() const { return viewing_enabled; }
};