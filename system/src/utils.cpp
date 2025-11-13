#include "utils.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type) {
    int latency = (stream_type == "main") ? Config::LATENCY_MAIN : Config::LATENCY_SUB;
    
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + " latency=" + std::to_string(latency) + 
           " ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1" +
           " ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx" +
           " ! nvvidconv ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries) {
    if (retries <= 0) retries = Config::RECONNECT_RETRIES;
    
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            spdlog::info("stream conectado");
            return cap;
        }
        spdlog::warn("reintento {}/{}", i+1, retries);
        std::this_thread::sleep_for(std::chrono::seconds(Config::RETRY_DELAY_SEC));
    }
    throw std::runtime_error("no se pudo conectar al stream");
}