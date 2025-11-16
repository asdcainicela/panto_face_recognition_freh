#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <spdlog/spdlog.h>

// Pipeline estándar robusto (basado en test/rtsp)
std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type);

// Pipeline con latencia adaptativa basada en FPS actual
std::string gst_pipeline_adaptive(const std::string& user, const std::string& pass, 
                                  const std::string& ip, int port, 
                                  const std::string& stream_type, double current_fps);

// Abrir captura con reintentos robustos
cv::VideoCapture open_cap(const std::string& pipeline, int retries=5);

// Verificar salud del stream
bool verify_stream_health(cv::VideoCapture& cap);