#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <spdlog/spdlog.h>

// Pipeline estándar con configuración robusta anti-caídas
std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type);

// Pipeline con latencia dinámica basada en FPS actual
std::string gst_pipeline_adaptive(const std::string& user, const std::string& pass, 
                                  const std::string& ip, int port, 
                                  const std::string& stream_type, double current_fps);

// Abrir captura con reintentos
cv::VideoCapture open_cap(const std::string& pipeline, int retries=5);

// Verificar salud del stream (para detectar caídas tempranas)
bool verify_stream_health(cv::VideoCapture& cap);