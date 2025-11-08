# Usar imagen base oficial de NVIDIA para Jetson con JetPack
FROM nvcr.io/nvidia/l4t-base:35.4.1

# Configurar variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV DISPLAY=:0

# Actualizar sistema e instalar dependencias base
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libgtk-3-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    gstreamer1.0-libav \
    libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== NUEVO: Dependencias X11 y display =====
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libice6 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    x11-apps \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ===== NUEVO: Herramientas de red y RTSP =====
RUN apt-get update && apt-get install -y \
    ffmpeg \
    v4l-utils \
    net-tools \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar CUDA toolkit completo
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-4 \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Verificar que CUDA está instalado
RUN nvcc --version

# Instalar OpenCV con soporte CUDA optimizado para Jetson
WORKDIR /tmp
RUN git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 \
          -D CUDA_ARCH_BIN="8.7" \
          -D CUDA_ARCH_PTX="" \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_LIBV4L=ON \
          -D WITH_GSTREAMER=ON \
          -D WITH_GSTREAMER_0_10=OFF \
          -D WITH_GTK=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_TBB=ON \
          -D WITH_FFMPEG=ON \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D BUILD_opencv_cudacodec=OFF \
          -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
          .. && \
    make -j4 && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

# Verificar que OpenCV tiene CUDA y GTK
RUN pkg-config --modversion opencv4 && \
    ldconfig && \
    echo "OpenCV instalado correctamente"

# Instalar Eigen3 para álgebra lineal
RUN apt-get update && apt-get install -y libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Instalar TensorRT runtime
RUN apt-get update && apt-get install -y \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvparsers8 \
    libnvonnxparsers8 \
    python3-libnvinfer \
    || true && \
    rm -rf /var/lib/apt/lists/*

# Clonar repositorio demo de TensorRT
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/TensorRT.git tensorrt-demos && \
    cd tensorrt-demos && \
    git checkout main || true

# Crear directorios para el proyecto
RUN mkdir -p /workspace/models /workspace/videos /workspace/src /workspace/build /workspace/faces_db /workspace/logs /workspace/new_faces

# ===== NUEVO: Demo RTSP con display =====
RUN cat > /workspace/src/rtsp_viewer.cpp <<'RTSPEOF'
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

std::string camera_url(const std::string& user, const std::string& pass, 
                       const std::string& ip, int port, bool sub=true) {
    std::string stream = sub ? "sub" : "main";
    return "rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream;
}

cv::VideoCapture open_cap(const std::string &url, int retries=5) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(url, cv::CAP_FFMPEG);
        if (cap.isOpened()) {
            std::cout << "? Conexión exitosa al stream" << std::endl;
            return cap;
        }
        std::cerr << "Reintento " << (i+1) << "/" << retries << "...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    throw std::runtime_error("No se pudo conectar: " + url);
}

int main(int argc, char** argv) {
    std::cout << "=== RTSP Viewer con OpenCV + CUDA ===" << std::endl;
    std::cout << "OpenCV: " << CV_VERSION << std::endl;
    std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    
    // Configuración por defecto (puedes cambiarla)
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";
    int port = 554;
    
    // Permitir URL custom como argumento
    std::string url = (argc > 1) ? argv[1] : camera_url(user, pass, ip, port, true);
    
    std::cout << "Conectando a: " << url << std::endl;
    cv::VideoCapture cap = open_cap(url);
    
    cv::Mat frame;
    int frame_count = 0;
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Frame perdido, reconectando...\n";
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            cap = open_cap(url);
            continue;
        }
        
        frame_count++;
        
        // Calcular FPS
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double fps = frame_count / elapsed;
        
        // Resize para display
        cv::Mat display;
        cv::resize(frame, display, cv::Size(1280, 720));
        
        // Overlay info
        std::string info = "Frame: " + std::to_string(frame_count) + 
                          " | FPS: " + std::to_string(int(fps));
        cv::putText(display, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("RTSP Stream", display);
        
        char c = (char) cv::waitKey(1);
        if (c == 27 || c == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    std::cout << "\nTotal frames: " << frame_count << std::endl;
    return 0;
}
RTSPEOF

# Demo original con CUDA
RUN cat > /workspace/src/demo.cpp <<'DEMOCPP'
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "=== OpenCV + CUDA Demo ===" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices available: " << cuda_devices << std::endl;
    
    if (cuda_devices > 0) {
        cv::cuda::DeviceInfo deviceInfo;
        std::cout << "GPU: " << deviceInfo.name() << std::endl;
        std::cout << "Compute capability: " << deviceInfo.majorVersion() 
                  << "." << deviceInfo.minorVersion() << std::endl;
    }
    
    std::string video_path = (argc > 1) ? argv[1] : "/workspace/videos/sample.mp4";
    cv::VideoCapture cap(video_path);
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << video_path << std::endl;
        return -1;
    }
    
    std::cout << "Processing video with GPU..." << std::endl;
    
    cv::Mat frame;
    cv::cuda::GpuMat gpu_frame, gpu_gray;
    int frame_count = 0;
    
    while (cap.read(frame)) {
        gpu_frame.upload(frame);
        cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat gray;
        gpu_gray.download(gray);
        
        cv::imshow("Original", frame);
        cv::imshow("GPU Grayscale", gray);
        
        frame_count++;
        if (cv::waitKey(30) == 27) break;
    }
    
    std::cout << "Processed " << frame_count << " frames" << std::endl;
    return 0;
}
DEMOCPP

# Crear CMakeLists.txt
RUN cat > /workspace/CMakeLists.txt <<'CMAKEEOF'
cmake_minimum_required(VERSION 3.10)
project(JetsonCppDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Demo original
add_executable(demo src/demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS})

# RTSP viewer
add_executable(rtsp_viewer src/rtsp_viewer.cpp)
target_link_libraries(rtsp_viewer ${OpenCV_LIBS} pthread)
CMAKEEOF

# Compilar demos
WORKDIR /workspace/build
RUN cmake .. && make -j$(nproc)

WORKDIR /workspace

CMD ["/bin/bash"]
