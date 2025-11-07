#pragma once
#include <opencv2/opencv.hpp>
#include <string>

std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type);

cv::VideoCapture open_cap(const std::string& pipeline, int retries=5);