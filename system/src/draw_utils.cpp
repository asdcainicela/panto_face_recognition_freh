#include "draw_utils.hpp"
#include "stream_capture.hpp"

namespace DrawUtils {

void draw_stream_info(cv::Mat& frame, const StreamStats& stats, 
                     const std::string& stream_type, 
                     const DrawConfig& config) {
    int y = config.margin_y;
    int line_height = config.line_spacing;
    
    // Calculate info panel size
    int panel_height = 0;
    if (config.show_stream_name) panel_height += line_height;
    if (config.show_frames) panel_height += line_height;
    if (config.show_fps) panel_height += line_height;
    if (config.show_lost) panel_height += line_height;
    if (config.show_resolution) panel_height += line_height;
    if (config.show_recording) panel_height += line_height;
    
    // Draw border
    if (config.show_border && panel_height > 0) {
        cv::rectangle(frame, 
                     cv::Rect(0, 0, 250, panel_height + config.margin_y), 
                     config.color, 1);
    }
    
    // Draw info
    if (config.show_stream_name) {
        cv::putText(frame, "stream: " + stream_type, 
                   cv::Point(config.margin_x, y),
                   config.font, config.font_scale, config.color, config.thickness);
        y += line_height;
    }
    
    if (config.show_frames) {
        cv::putText(frame, "frames: " + std::to_string(stats.frames), 
                   cv::Point(config.margin_x, y),
                   config.font, config.font_scale, config.color, config.thickness);
        y += line_height;
    }
    
    if (config.show_fps) {
        cv::putText(frame, "fps: " + std::to_string(static_cast<int>(stats.fps)), 
                   cv::Point(config.margin_x, y),
                   config.font, config.font_scale, config.color, config.thickness);
        y += line_height;
    }
    
    if (config.show_lost) {
        cv::putText(frame, "lost: " + std::to_string(stats.lost), 
                   cv::Point(config.margin_x, y),
                   config.font, config.font_scale, config.color, config.thickness);
        y += line_height;
    }
    
    if (config.show_resolution) {
        std::string res_text = "res: " + std::to_string(stats.resolution.width) + 
                              "x" + std::to_string(stats.resolution.height);
        cv::putText(frame, res_text, 
                   cv::Point(config.margin_x, y),
                   config.font, config.font_scale, config.color, config.thickness);
        y += line_height;
    }
}

void draw_recording_indicator(cv::Mat& frame, bool is_recording, const DrawConfig& config) {
    if (!is_recording) return;
    
    int y = config.margin_y;
    int line_height = config.line_spacing;
    
    // Calculate position (after other info)
    int offset = 0;
    if (config.show_stream_name) offset += line_height;
    if (config.show_frames) offset += line_height;
    if (config.show_fps) offset += line_height;
    if (config.show_lost) offset += line_height;
    if (config.show_resolution) offset += line_height;
    
    cv::putText(frame, "REC", 
               cv::Point(config.margin_x, y + offset),
               config.font, 0.7, config.rec_color, 2);
    
    // Red dot
    cv::circle(frame, 
              cv::Point(config.margin_x + 60, y + offset - 7),
              5, config.rec_color, -1);
}

void draw_fps_counter(cv::Mat& frame, double fps, const cv::Point& position, const DrawConfig& config) {
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fps_text, position,
               config.font, config.font_scale, config.color, config.thickness);
}

void draw_text_with_background(cv::Mat& frame, const std::string& text,
                               const cv::Point& position,
                               const cv::Scalar& text_color,
                               const cv::Scalar& bg_color,
                               const DrawConfig& config) {
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, config.font, config.font_scale, 
                                         config.thickness, &baseline);
    
    // Draw background rectangle
    cv::rectangle(frame,
                 cv::Point(position.x - 2, position.y - text_size.height - 2),
                 cv::Point(position.x + text_size.width + 2, position.y + baseline + 2),
                 bg_color, -1);
    
    // Draw text
    cv::putText(frame, text, position,
               config.font, config.font_scale, text_color, config.thickness);
}

}