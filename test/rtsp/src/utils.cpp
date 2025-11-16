#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type) {
    int latency = (stream_type == "main") ? 50 : 20;
    
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + " latency=" + std::to_string(latency) + " ! "
           "rtph264depay ! h264parse ! nvv4l2decoder ! "
           "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
           "video/x-raw,format=BGR ! appsink drop=1 max-buffers=2 sync=0";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            spdlog::info("conectado exitosamente");
            return cap;
        }
        spdlog::warn("intento {}/{} fallido. reintentando en 2s...", i+1, retries);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    spdlog::error("no se pudo conectar al pipeline");
    throw std::runtime_error("no se pudo conectar a: " + pipeline);
}