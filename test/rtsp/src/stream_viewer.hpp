#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <atomic>

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
    cv::VideoWriter writer;

    int frames, lost;
    int last_frame_count;   // NUEVO: para tracking de frames
    double current_fps;
    std::chrono::steady_clock::time_point start_main, start_fps, last_health_check;
    std::string window_name;
    std::string pipeline;

    bool recording_enabled;
    bool recording_paused;  // NUEVO: para pausar grabación sin cerrar writer
    int frames_recorded;    // NUEVO: contador de frames grabados
    std::string output_filename;
    std::atomic<bool>* stop_signal;
    int max_duration;
    int reconnect_count;
    bool use_adaptive_latency;

    bool reconnect();
    void update_fps();
    const StreamStats& get_stats();
    void print_stats();
    void print_final_stats();
    bool init_recording();
    void stop_recording();
    void pause_recording();   // NUEVO
    void resume_recording();  // NUEVO
    
    StreamStats cached_stats;

public:
    StreamViewer(const std::string& user, const std::string& pass, 
                 const std::string& ip, int port, const std::string& stream_type,
                 cv::Size display_size = cv::Size(640, 480), int fps_interval = 30);
    
    void enable_recording(const std::string& output_dir = "../videos_rtsp");
    void set_max_duration(int seconds);
    void set_stop_signal(std::atomic<bool>* signal);
    void enable_adaptive_latency(bool enable);
    void run();
    
    bool is_recording() const { return recording_enabled && writer.isOpened(); }
    int get_reconnect_count() const { return reconnect_count; }
    int get_frames_recorded() const { return frames_recorded; }  // NUEVO
};