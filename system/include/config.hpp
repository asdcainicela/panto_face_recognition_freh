#pragma once
#include <string>
#include <opencv2/opencv.hpp>

namespace Config {
    
    // Camera defaults
    constexpr const char* DEFAULT_USER = "admin";
    constexpr const char* DEFAULT_PASS = "Panto2025";
    constexpr const char* DEFAULT_IP = "192.168.0.101";
    constexpr int DEFAULT_PORT = 554;
    constexpr const char* DEFAULT_STREAM = "main";
    
    // Display defaults
    constexpr int DEFAULT_DISPLAY_WIDTH = 640;
    constexpr int DEFAULT_DISPLAY_HEIGHT = 480;
    constexpr int DEFAULT_FPS_INTERVAL = 30;
    
    // Recording defaults
    constexpr const char* DEFAULT_OUTPUT_DIR = "../videos";
    constexpr const char* DEFAULT_VIDEO_CODEC = "mp4v";
    constexpr double DEFAULT_VIDEO_FPS = 25.0;
    
    // Window positions
    constexpr int WINDOW_OFFSET_X = 650;
    constexpr int WINDOW_OFFSET_Y = 0;
    
    // GStreamer settings
    constexpr int LATENCY_MAIN = 50;
    constexpr int LATENCY_SUB = 20;
    constexpr int RECONNECT_DELAY_SEC = 1;
    constexpr int RECONNECT_RETRIES = 5;
    constexpr int RETRY_DELAY_SEC = 2;
    
    // Config profiles
    struct Resolution {
        int width;
        int height;
        std::string config_file;
    };
    
    const Resolution RESOLUTIONS[] = {
        {3840, 2160, "configs/config_4k.toml"},
        {2560, 1440, "configs/config_1440p.toml"},
        {1920, 1080, "configs/config_1080p_roi.toml"},
        {1280, 720,  "configs/config_720p.toml"}
    };
    
    inline std::string get_config_for_resolution(const cv::Size& size) {
        int pixels = size.width * size.height;
        
        for (const auto& res : RESOLUTIONS) {
            int res_pixels = res.width * res.height;
            if (pixels >= static_cast<int>(res_pixels * 0.9)) {
                return res.config_file;
            }
        }
        
        return "configs/config_1080p_roi.toml";
    }
}