// ============= include/tensorrt_logger.hpp =============
#pragma once
#include <NvInfer.h>
#include <spdlog/spdlog.h>
namespace panto {
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TensorRT] {}", msg);
        }
    }
};
} // namespace panto