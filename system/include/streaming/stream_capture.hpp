#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

struct StreamStats {
    double fps;
    int frames;
    int lost;
    cv::Size resolution;
    int reconnects;
    int latency_warnings;
};

class FrameQueue {
private:
    std::queue<cv::Mat> queue;
    std::mutex mutex;
    std::condition_variable cv;
    size_t max_size;
    
public:
    FrameQueue(size_t max_size = 3) : max_size(max_size) {}
    
    void push(const cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.size() >= max_size) queue.pop();
        queue.push(frame.clone());
        cv.notify_one();
    }
    
    bool pop(cv::Mat& frame, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue.empty(); })) {
            frame = queue.front();
            queue.pop();
            return true;
        }
        return false;
    }
    
    size_t size() {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }
    
    void clear() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!queue.empty()) queue.pop();
    }
};

class CaptureThread {
private:
    std::string pipeline;
    std::atomic<bool> running;
    std::thread thread;
    FrameQueue& frame_queue;
    std::atomic<int> frames_captured;
    std::atomic<int> frames_lost;
    std::atomic<int> reconnects;
    std::atomic<int> consecutive_fails;
    int max_errors_before_reconnect = 30;
    int max_consecutive_fails = 100;
    int reconnect_threshold = 10;
    cv::VideoCapture cap;
    
    bool open_capture(int retries = 5);
    bool reconnect();
    void capture_loop();
    
public:
    CaptureThread(const std::string& pipeline, FrameQueue& queue);
    ~CaptureThread();
    
    void start();
    void stop();
    bool is_running() const { return running.load(); }
    
    int get_frames() const { return frames_captured.load(); }
    int get_lost() const { return frames_lost.load(); }
    int get_reconnects() const { return reconnects.load(); }
};

class StreamCapture {
private:
    std::string user, pass, ip, stream_type;
    int port;
    int fps_interval;
    std::string capture_backend;
    std::string pipeline;
    std::string window_name;
    
    FrameQueue frame_queue;
    std::unique_ptr<CaptureThread> capture_thread;
    
    int frames_displayed;
    double current_fps;
    std::chrono::steady_clock::time_point start_time, start_fps;
    
    bool viewing_enabled;
    cv::Size display_size;
    std::atomic<bool>* stop_signal;
    
    StreamStats cached_stats;
    
    void update_fps();

public:
    StreamCapture(const std::string& user, const std::string& pass,
                  const std::string& ip, int port, const std::string& stream_type,
                  const std::string& backend = "ffmpeg"); //"ffmpeg" o "gstreamer"
    ~StreamCapture();
    
    void enable_viewing(const cv::Size& size = cv::Size(640, 480));
    void set_stop_signal(std::atomic<bool>* signal);
    void set_fps_interval(int interval);
    
    bool open();
    bool read(cv::Mat& frame);  // Thread-safe read
    void release();
    
    // Display loop en thread principal
    void run();
    
    const StreamStats& get_stats();
    void print_stats() const;
    void print_final_stats() const;
    
    bool is_capturing() const {
        return capture_thread && capture_thread->is_running();
    }
};