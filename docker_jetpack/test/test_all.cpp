#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

void test_onnx() {
    std::cout << "\nTest ONNX Runtime C++" << std::endl;
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "ONNX Env: OK" << std::endl;
        
        auto providers = Ort::GetAvailableProviders();
        std::cout << "Providers: ";
        for (const auto& p : providers) std::cout << p << " ";
        std::cout << std::endl;
        
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        std::cout << "SessionOptions: OK" << std::endl;
        std::cout << "ONNX Runtime: PASS" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX Runtime: FAIL - " << e.what() << std::endl;
    }
}

void test_opencv() {
    std::cout << "\nTest OpenCV C++" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    try {
        int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA devices: " << cuda_devices << std::endl;
        
        if (cuda_devices > 0) {
            cv::cuda::printCudaDeviceInfo(0);
            
            cv::Mat cpu_mat = cv::Mat::ones(1000, 1000, CV_32F);
            cv::cuda::GpuMat gpu_mat;
            gpu_mat.upload(cpu_mat);
            cv::cuda::GpuMat gpu_result;
            cv::cuda::multiply(gpu_mat, cv::Scalar(2.0), gpu_result);
            cv::Mat result;
            gpu_result.download(result);
            
            std::cout << "CUDA operation: OK" << std::endl;
            std::cout << "OpenCV with CUDA: PASS" << std::endl;
        } else {
            std::cout << "OpenCV without CUDA: WARNING" << std::endl;
        }
        
        cv::Mat test_img = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::circle(test_img, cv::Point(50, 50), 30, cv::Scalar(0, 255, 0), -1);
        std::cout << "Basic operations: OK" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "OpenCV: FAIL - " << e.what() << std::endl;
    }
}

void test_cuda() {
    std::cout << "\nTest CUDA" << std::endl;
    
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA: FAIL - " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "CUDA devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  MPs: " << prop.multiProcessorCount << std::endl;
    }
    
    std::cout << "CUDA: PASS" << std::endl;
}

int main() {
    std::cout << "\nC++ Tests Starting" << std::endl;
    std::cout << "GCC: " << __VERSION__ << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    
    test_cuda();
    test_opencv();
    test_onnx();
    
    std::cout << "\nC++ Tests Complete" << std::endl;
    return 0;
}