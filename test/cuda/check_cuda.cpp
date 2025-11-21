#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    
    return 0;
}