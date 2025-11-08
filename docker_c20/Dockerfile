# ============================================================================
# Dockerfile optimizado para NVIDIA Jetson Orin Nano
# JetPack 5.1.2 (L4T R35.4.1) + C++20 + OpenCV 4.10 + TensorRT
# ============================================================================

FROM nvcr.io/nvidia/l4t-base:35.4.1

# Variables de entorno
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-11.4 \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} \
    DISPLAY=:0 \
    OPENCV_VERSION=4.10.0 \
    CMAKE_VERSION=3.28.1

# ============================================================================
# FASE 1: Actualizar sistema e instalar compiladores modernos (GCC 11 + C++20)
# ============================================================================

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y \
        gcc-11 \
        g++-11 \
        build-essential \
        pkg-config \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
    && rm -rf /var/lib/apt/lists/*

# Verificar versión GCC
RUN gcc --version && g++ --version

# ============================================================================
# FASE 2: Instalar CMake moderno (3.28+)
# ============================================================================

WORKDIR /tmp
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz \
    && tar -xzf cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cmake /usr/local/bin/cmake \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/ctest /usr/local/bin/ctest \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cpack /usr/local/bin/cpack \
    && rm -f cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz

# Verificar CMake
RUN cmake --version

# ============================================================================
# FASE 3: CUDA Toolkit completo + cuDNN + TensorRT
# ============================================================================

RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-4 \
    libcudnn8 \
    libcudnn8-dev \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvparsers8 \
    libnvonnxparsers8 \
    libnvinfer-dev \
    python3-libnvinfer \
    && rm -rf /var/lib/apt/lists/*

# Fix CUDA paths y verificar
RUN if [ -d "/usr/local/cuda-11.4" ]; then \
        export PATH=/usr/local/cuda-11.4/bin:$PATH; \
    fi && \
    if command -v nvcc >/dev/null 2>&1; then \
        nvcc --version; \
    else \
        echo "CUDA installed but nvcc not in PATH (will be available at runtime)"; \
    fi

# ============================================================================
# FASE 4: Dependencias de OpenCV y multimedia
# ============================================================================

RUN apt-get update && apt-get install -y \
    # Desarrollo de imágenes
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    # Video codecs
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    # GStreamer (crítico para RTSP en Jetson)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    # GUI (GTK3 para imshow)
    libgtk-3-dev \
    libgtk2.0-dev \
    # Álgebra lineal
    libatlas-base-dev \
    gfortran \
    libeigen3-dev \
    # Threading
    libtbb-dev \
    libtbb2 \
    # Otros
    libdc1394-dev \
    # X11 y OpenGL
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libxrender-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxi-dev \
    libxext6 \
    libsm6 \
    libice6 \
    x11-apps \
    mesa-utils \
    # Networking
    ffmpeg \
    v4l-utils \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# FASE 5: Librerías C++ modernas y utilidades
# ============================================================================

RUN apt-get update && apt-get install -y \
    # === Logging y debugging ===
    libspdlog-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    # === Serialización y formato ===
    libfmt-dev \
    libjsoncpp-dev \
    nlohmann-json3-dev \
    libyaml-cpp-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libmsgpack-dev \
    # === Bases de datos ===
    libsqlite3-dev \
    libpq-dev \
    libmysqlclient-dev \
    libhiredis-dev \
    libmongoc-dev \
    # === Networking y comunicación ===
    libzmq3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libwebsocketpp-dev \
    libmicrohttpd-dev \
    libgrpc++-dev \
    # === HTTP y REST ===
    libcpprest-dev \
    libpistache-dev \
    # === Boost (colección extensa) ===
    libboost-all-dev \
    # === Álgebra lineal y matemáticas ===
    libarmadillo-dev \
    libopenblas-dev \
    liblapack-dev \
    libgsl-dev \
    # === Computer Vision extras ===
    libpcl-dev \
    libvtk9-dev \
    # === Testing ===
    libgtest-dev \
    libgmock-dev \
    catch2 \
    # === Utilidades generales ===
    libtool \
    autoconf \
    automake \
    ccache \
    clang-format \
    cppcheck \
    valgrind \
    gdb \
    htop \
    tmux \
    vim \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Compilar Google Test (si no viene precompilado)
RUN cd /usr/src/gtest && \
    cmake . && make && \
    cp lib/*.a /usr/lib || true

# === Instalar bibliotecas header-only modernas desde GitHub ===

WORKDIR /tmp

# CLI11 - Command line parser
RUN git clone --depth 1 https://github.com/CLIUtils/CLI11.git && \
    cd CLI11 && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf CLI11

# spdlog (latest version desde source si apt es vieja)
RUN git clone --depth 1 --branch v1.12.0 https://github.com/gabime/spdlog.git && \
    cd spdlog && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd /tmp && rm -rf spdlog

# backward-cpp - Stack trace bonito
RUN git clone --depth 1 https://github.com/bombela/backward-cpp.git && \
    cd backward-cpp && \
    cp backward.hpp /usr/local/include/ && \
    cd /tmp && rm -rf backward-cpp

# magic_enum - Enum to string
RUN git clone --depth 1 https://github.com/Neargye/magic_enum.git && \
    cd magic_enum && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf magic_enum

# cpptrace - Better stack traces
RUN git clone --depth 1 https://github.com/jeremy-rifkin/cpptrace.git && \
    cd cpptrace && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd /tmp && rm -rf cpptrace

# toml++ - TOML parser
RUN git clone --depth 1 https://github.com/marzer/tomlplusplus.git && \
    cd tomlplusplus && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf tomlplusplus

# Expected - std::expected backport
RUN git clone --depth 1 https://github.com/TartanLlama/expected.git && \
    cd expected && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf expected

# ============================================================================
# FASE 6: Compilar OpenCV 4.10 con CUDA optimizado
# ============================================================================

WORKDIR /tmp
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git \
    && git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git \
    && cd opencv && mkdir build && cd build \
    && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D CMAKE_CXX_STANDARD=20 \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        # CUDA optimizations
        -D WITH_CUDA=ON \
        -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
        -D CUDA_ARCH_BIN="8.7" \
        -D CUDA_ARCH_PTX="" \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        # Video support
        -D WITH_LIBV4L=ON \
        -D WITH_V4L=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_GSTREAMER_0_10=OFF \
        -D WITH_FFMPEG=ON \
        # GUI and display
        -D WITH_GTK=ON \
        -D WITH_GTK_2_X=OFF \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=ON \
        # Performance
        -D WITH_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_LAPACK=ON \
        # Python (deshabilitado para reducir tamaño)
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        # Testing (deshabilitado)
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        # Extra features
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_opencv_cudacodec=ON \
        # Compiler flags
        -D CMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
        .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

# Verificar OpenCV
RUN pkg-config --modversion opencv4 \
    && echo "OpenCV instalado con éxito"

# ============================================================================
# FASE 7: Python y herramientas de ML (opcional pero útil)
# ============================================================================

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip3 install --upgrade pip

# Instalar librerías Python útiles para prototipado
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    onnx \
    onnxruntime-gpu \
    pillow \
    matplotlib \
    pyyaml \
    tqdm \
    requests \
    Flask \
    fastapi \
    uvicorn

# ============================================================================
# FASE 8: Descargar ONNXRuntime (opcional, para modelos ONNX)
# ============================================================================

WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz -C /opt \
    && ln -s /opt/onnxruntime-linux-aarch64-1.16.3 /opt/onnxruntime \
    && rm onnxruntime-linux-aarch64-1.16.3.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime

# ============================================================================
# FASE 9: Configurar workspace del proyecto
# ============================================================================

WORKDIR /workspace

# Crear estructura de directorios
RUN mkdir -p \
    /workspace/src \
    /workspace/include \
    /workspace/models \
    /workspace/videos \
    /workspace/build \
    /workspace/data/faces_db \
    /workspace/data/embeddings \
    /workspace/logs \
    /workspace/config

# ============================================================================
# FINALIZACION
# ============================================================================

WORKDIR /workspace

# Script de bienvenida simple
RUN cat > /workspace/welcome.sh <<'EOF'
#!/bin/bash
echo "=========================================="
echo "  Jetson Orin Nano - Development Env"
echo "  C++20 | OpenCV 4.10 | CUDA 11.4"
echo "=========================================="
echo ""
gcc --version | head -1
cmake --version | head -1
pkg-config --modversion opencv4 2>/dev/null || echo "OpenCV: checking..."
nvcc --version 2>/dev/null | grep release || echo "CUDA: checking..."
echo ""
echo "Workspace: /workspace"
echo "=========================================="
EOF

RUN chmod +x /workspace/welcome.sh

CMD ["/bin/bash", "-c", "/workspace/welcome.sh && /bin/bash"]