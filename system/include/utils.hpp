#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <spdlog/spdlog.h>

// FFMPEG URL (más estable que GStreamer)
std::string ffmpeg_rtsp_url(const std::string& user, const std::string& pass, 
                            const std::string& ip, int port, const std::string& stream_type);

// Pipeline GStreamer (legacy)
std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type);

// Pipeline con latencia adaptativa
std::string gst_pipeline_adaptive(const std::string& user, const std::string& pass, 
                                  const std::string& ip, int port, 
                                  const std::string& stream_type, double current_fps);

// Abrir captura con reintentos robustos (USA FFMPEG POR DEFECTO)
cv::VideoCapture open_cap(const std::string& pipeline, int retries=5);

// Verificar salud del stream
bool verify_stream_health(cv::VideoCapture& cap);