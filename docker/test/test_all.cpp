#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <NvInfer.h>

int main() {
    std::cout << "============================================================\n";
    std::cout << "VERIFICACIÓN COMPLETA: OpenCV + CUDA + TensorRT (C++)\n";
    std::cout << "============================================================\n\n";
    
    // ===== 1. OPENCV BÁSICO =====
    std::cout << "[1] OpenCV Información:\n";
    std::cout << "  - Versión: " << CV_VERSION << "\n";
    std::cout << "  - Major: " << CV_MAJOR_VERSION << "\n";
    std::cout << "  - Minor: " << CV_MINOR_VERSION << "\n\n";
    
    // ===== 2. BUILD INFO =====
    std::cout << "[2] OpenCV Build Configuration:\n";
    std::cout << cv::getBuildInformation() << "\n\n";
    
    // ===== 3. CUDA =====
    std::cout << "[3] CUDA Support:\n";
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "  - Dispositivos CUDA: " << cuda_devices << "\n";
    
    if (cuda_devices > 0) {
        std::cout << "  - ✓ CUDA HABILITADO\n";
        
        cv::cuda::DeviceInfo dev_info;
        std::cout << "  - GPU Name: " << dev_info.name() << "\n";
        std::cout << "  - Compute Capability: " 
                  << dev_info.majorVersion() << "." 
                  << dev_info.minorVersion() << "\n";
        std::cout << "  - Total Memory: " 
                  << dev_info.totalMemory() / (1024.0 * 1024.0 * 1024.0) 
                  << " GB\n";
        std::cout << "  - Free Memory: " 
                  << dev_info.freeMemory() / (1024.0 * 1024.0 * 1024.0) 
                  << " GB\n";
        std::cout << "  - Multi Processor Count: " 
                  << dev_info.multiProcessorCount() << "\n";
        std::cout << "  - Clock Rate: " 
                  << dev_info.clockRate() / 1000.0 << " MHz\n";
    } else {
        std::cout << "  - ✗ CUDA NO DISPONIBLE\n";
    }
    std::cout << "\n";
    
    // ===== 4. DNN MODULE (TensorRT) =====
    std::cout << "[4] OpenCV DNN Module:\n";
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = 
        cv::dnn::getAvailableBackends();
    
    std::cout << "  - Backends disponibles (" << backends.size() << "):\n";
    bool has_cuda_backend = false;
    
    for (const auto& backend : backends) {
        std::string backend_name;
        switch(backend.first) {
            case cv::dnn::DNN_BACKEND_DEFAULT: 
                backend_name = "DEFAULT"; break;
            case cv::dnn::DNN_BACKEND_HALIDE: 
                backend_name = "HALIDE"; break;
            case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE: 
                backend_name = "INFERENCE_ENGINE"; break;
            case cv::dnn::DNN_BACKEND_OPENCV: 
                backend_name = "OPENCV"; break;
            case cv::dnn::DNN_BACKEND_VKCOM: 
                backend_name = "VKCOM"; break;
            case cv::dnn::DNN_BACKEND_CUDA: 
                backend_name = "CUDA"; 
                has_cuda_backend = true; 
                break;
            default: 
                backend_name = "UNKNOWN";
        }
        
        std::string target_name;
        switch(backend.second) {
            case cv::dnn::DNN_TARGET_CPU: 
                target_name = "CPU"; break;
            case cv::dnn::DNN_TARGET_OPENCL: 
                target_name = "OPENCL"; break;
            case cv::dnn::DNN_TARGET_OPENCL_FP16: 
                target_name = "OPENCL_FP16"; break;
            case cv::dnn::DNN_TARGET_MYRIAD: 
                target_name = "MYRIAD"; break;
            case cv::dnn::DNN_TARGET_VULKAN: 
                target_name = "VULKAN"; break;
            case cv::dnn::DNN_TARGET_CUDA: 
                target_name = "CUDA"; break;
            case cv::dnn::DNN_TARGET_CUDA_FP16: 
                target_name = "CUDA_FP16"; break;
            default: 
                target_name = "UNKNOWN";
        }
        
        std::cout << "    • " << backend_name << " -> " << target_name << "\n";
    }
    
    std::cout << "\n  - CUDA Backend: " 
              << (has_cuda_backend ? "✓ SÍ" : "✗ NO") << "\n\n";
    
    // ===== 5. TENSORRT =====
    std::cout << "[5] TensorRT C++ API:\n";
    std::cout << "  - Version: " << NV_TENSORRT_MAJOR << "." 
              << NV_TENSORRT_MINOR << "." 
              << NV_TENSORRT_PATCH << "\n";
    std::cout << "  - Header: /usr/include/aarch64-linux-gnu/NvInfer.h\n";
    std::cout << "  - Library: /usr/lib/aarch64-linux-gnu/libnvinfer.so\n\n";
    
    // ===== 6. TEST PRÁCTICO CUDA =====
    std::cout << "[6] Test Práctico CUDA:\n";
    if (cuda_devices > 0) {
        try {
            // Crear imagen en CPU
            cv::Mat img_cpu(1920, 1080, CV_8UC3);
            cv::randu(img_cpu, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
            
            // Upload a GPU
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(img_cpu);
            std::cout << "  ✓ Upload a GPU exitoso\n";
            
            // Convertir a escala de grises en GPU
            cv::cuda::GpuMat gray_gpu;
            cv::cuda::cvtColor(img_gpu, gray_gpu, cv::COLOR_BGR2GRAY);
            std::cout << "  ✓ Conversión BGR->GRAY en GPU exitosa\n";
            
            // Download de GPU
            cv::Mat gray_cpu;
            gray_gpu.download(gray_cpu);
            std::cout << "  ✓ Download de GPU exitoso\n";
            
            std::cout << "  - Tamaño procesado: " 
                      << img_cpu.cols << "x" << img_cpu.rows << "\n";
            std::cout << "  - ✓ TODAS LAS OPERACIONES CUDA EXITOSAS\n";
            
        } catch (const cv::Exception& e) {
            std::cout << "  ✗ Error en test CUDA: " << e.what() << "\n";
        }
    } else {
        std::cout << "  ✗ No se puede ejecutar (CUDA no disponible)\n";
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "RESUMEN:\n";
    std::cout << "  OpenCV:    " << CV_VERSION << "\n";
    std::cout << "  CUDA:      " << (cuda_devices > 0 ? "✓ Disponible" : "✗ No disponible") << "\n";
    std::cout << "  DNN CUDA:  " << (has_cuda_backend ? "✓ Disponible" : "✗ No disponible") << "\n";
    std::cout << "  TensorRT:  " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "\n";
    std::cout << "============================================================\n";
    
    return 0;
}
