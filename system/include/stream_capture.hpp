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
    int reconnects;
    int latency_warnings;
};

class StreamCapture {
private:
    std::string user, pass, ip, stream_type;
    int port;
    int fps_interval;
    
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    
    int frames, lost;
    int last_frame_count;
    int latency_warnings;
    int reconnect_count;
    int frames_recorded;
    
    double current_fps;
    std::chrono::steady_clock::time_point start_time, start_fps, last_health_check;
    std::string pipeline;
    std::string window_name;
    
    bool recording_enabled;
    bool recording_paused;
    bool viewing_enabled;
    bool use_adaptive_latency;
    
    cv::Size display_size;
    std::string output_filename;
    std::atomic<bool>* stop_signal;
    int max_duration;
    
    StreamStats cached_stats;
    
    bool reconnect();
    void update_fps();
    bool init_recording();
    void stop_recording();
    void pause_recording();
    void resume_recording();

public:
    StreamCapture(const std::string& user, const std::string& pass,
                  const std::string& ip, int port, const std::string& stream_type);
    ~StreamCapture();
    
    // Configuration
    void enable_recording(const std::string& output_dir = "../videos");
    void enable_viewing(const cv::Size& size = cv::Size(640, 480));
    void enable_adaptive_latency(bool enable);
    void set_max_duration(int seconds);
    void set_stop_signal(std::atomic<bool>* signal);
    void set_fps_interval(int interval);
    
    // Basic operations
    bool open();
    bool read(cv::Mat& frame);
    void release();
    
    // High-level operation
    void run();
    
    // Stats
    const StreamStats& get_stats();
    void print_stats() const;
    void print_final_stats() const;
    
    bool is_recording() const { return recording_enabled && writer.isOpened(); }
    bool is_viewing() const { return viewing_enabled; }
    int get_reconnect_count() const { return reconnect_count; }
    int get_frames_recorded() const { return frames_recorded; }
};