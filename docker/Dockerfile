FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH} \
    DISPLAY=:0 \
    OPENCV_VERSION=4.10.0 \
    CMAKE_VERSION=3.28.1

# ===== COMPILADORES =====
RUN apt-get update && apt-get install -y \
    software-properties-common wget curl git \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y \
        gcc-11 g++-11 gcc-9 g++-9 \
        build-essential pkg-config \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    && rm -rf /var/lib/apt/lists/*

# ===== CMAKE ACTUALIZADO =====
WORKDIR /tmp
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz \
    && tar -xzf cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/* /usr/local/bin/ \
    && rm -f cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz

# ===== DEPENDENCIAS OPENCV =====
RUN apt-get update && apt-get install -y \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    libeigen3-dev libtbb-dev libdc1394-22-dev \
    libgl1-mesa-dev libglu1-mesa-dev libglew-dev \
    ffmpeg v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ===== LIBRERÍAS C++ ESENCIALES =====
RUN apt-get update && apt-get install -y \
    libspdlog-dev libgoogle-glog-dev libgflags-dev \
    libfmt-dev libjsoncpp-dev nlohmann-json3-dev \
    libyaml-cpp-dev libprotobuf-dev protobuf-compiler \
    libboost-all-dev libgtest-dev \
    ccache gdb htop tmux vim nano tree sudo \
    && rm -rf /var/lib/apt/lists/*

# ===== HEADERS-ONLY LIBRARIES =====
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/CLIUtils/CLI11.git && \
    cd CLI11 && mkdir build && cd build && cmake .. && make install && \
    cd /tmp && rm -rf CLI11

RUN git clone --depth 1 https://github.com/bombela/backward-cpp.git && \
    cd backward-cpp && cp backward.hpp /usr/local/include/ && \
    cd /tmp && rm -rf backward-cpp

RUN git clone --depth 1 https://github.com/Neargye/magic_enum.git && \
    cd magic_enum && mkdir build && cd build && cmake .. && make install && \
    cd /tmp && rm -rf magic_enum

# ===== PYTHON MÍNIMO (para OpenCV y Jupyter después) =====
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# ===== COMPILAR OPENCV CON CUDA =====
RUN update-alternatives --set gcc /usr/bin/gcc-9

WORKDIR /tmp
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git \
    && git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git \
    && cd opencv && mkdir build && cd build \
    && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D CMAKE_CXX_STANDARD=17 \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN="8.7" \
        -D CUDA_ARCH_PTX="" \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GTK=ON \
        -D WITH_OPENGL=ON \
        -D WITH_TBB=ON \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D CMAKE_CXX_FLAGS="-O3" \
        .. \
    && make -j2 \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

RUN update-alternatives --set gcc /usr/bin/gcc-11

# ===== VERIFICAR TENSORRT =====
RUN ls -la /usr/include/aarch64-linux-gnu/NvInfer.h && \
    ls -la /usr/lib/aarch64-linux-gnu/libnvinfer.so* && \
    echo "✓ TensorRT C++ API disponible"

# ===== LDCONFIG =====
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf \
    && echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/tegra.conf \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && ldconfig

# ===== LOCALE Y TIMEZONE =====
RUN apt-get update && apt-get install -y locales \
    && locale-gen es_ES.UTF-8 && rm -rf /var/lib/apt/lists/*

ENV LANG=es_ES.UTF-8

RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime

# ===== GIT CONFIG =====
RUN git config --global user.email "gerald.cainicela.a@gmail.com" \
    && git config --global user.name "userasd"

RUN mkdir -p /workspace

WORKDIR /workspace

CMD ["bash"]