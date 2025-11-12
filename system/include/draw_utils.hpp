#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct StreamStats;

namespace DrawUtils {
    
    struct DrawConfig {
        bool show_border = true;
        bool show_stream_name = true;
        bool show_frames = true;
        bool show_fps = true;
        bool show_lost = true;
        bool show_resolution = true;
        bool show_recording = false;
        
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::Scalar rec_color = cv::Scalar(0, 0, 255);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int line_spacing = 20;
        int margin_x = 10;
        int margin_y = 20;
    };
    
    void draw_stream_info(cv::Mat& frame, const StreamStats& stats, 
                         const std::string& stream_type, 
                         const DrawConfig& config = DrawConfig());
    
    void draw_recording_indicator(cv::Mat& frame, bool is_recording,
                                  const DrawConfig& config = DrawConfig());
    
    void draw_fps_counter(cv::Mat& frame, double fps, 
                         const cv::Point& position = cv::Point(10, 20),
                         const DrawConfig& config = DrawConfig());
    
    void draw_text_with_background(cv::Mat& frame, const std::string& text,
                                   const cv::Point& position,
                                   const cv::Scalar& text_color,
                                   const cv::Scalar& bg_color,
                                   const DrawConfig& config = DrawConfig());
}