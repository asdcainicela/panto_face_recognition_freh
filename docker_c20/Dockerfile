# ============================================================================
# Dockerfile optimizado para NVIDIA Jetson Orin Nano
# JetPack 5.1.2 (L4T R35.4.1) + C++20 + OpenCV 4.10 + TensorRT + Jupyter
# ============================================================================

FROM nvcr.io/nvidia/l4t-base:35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-11.4 \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} \
    DISPLAY=:0 \
    OPENCV_VERSION=4.10.0 \
    CMAKE_VERSION=3.28.1

# Sistema base + compiladores GCC 11
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
        gcc-9 \
        g++-9 \
        build-essential \
        pkg-config \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
    && rm -rf /var/lib/apt/lists/*

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

RUN cmake --version

# CUDA Toolkit + cuDNN + TensorRT
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

RUN if [ -d "/usr/local/cuda-11.4" ]; then \
        export PATH=/usr/local/cuda-11.4/bin:$PATH; \
    fi && \
    if command -v nvcc >/dev/null 2>&1; then \
        nvcc --version; \
    else \
        echo "CUDA installed but nvcc not in PATH (will be available at runtime)"; \
    fi

# Dependencias OpenCV y multimedia
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgtk-3-dev \
    libgtk2.0-dev \
    libatlas-base-dev \
    gfortran \
    libeigen3-dev \
    libtbb-dev \
    libtbb2 \
    libdc1394-22-dev \
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
    ffmpeg \
    v4l-utils \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Librerías C++ modernas
RUN apt-get update && apt-get install -y \
    libspdlog-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libfmt-dev \
    libjsoncpp-dev \
    nlohmann-json3-dev \
    libyaml-cpp-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libmsgpack-dev \
    libsqlite3-dev \
    libpq-dev \
    libmysqlclient-dev \
    libhiredis-dev \
    libmongoc-dev \
    libzmq3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libwebsocketpp-dev \
    libmicrohttpd-dev \
    libgrpc++-dev \
    libcpprest-dev \
    libboost-all-dev \
    libarmadillo-dev \
    libopenblas-dev \
    liblapack-dev \
    libgsl-dev \
    libpcl-dev \
    libvtk7-dev \
    libgtest-dev \
    libgmock-dev \
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

RUN cd /usr/src/gtest && \
    cmake . && \
    make && \
    cp lib/*.a /usr/lib || cp *.a /usr/lib || true

# Librerías header-only desde GitHub
WORKDIR /tmp

RUN git clone --depth 1 https://github.com/CLIUtils/CLI11.git && \
    cd CLI11 && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf CLI11

RUN git clone --depth 1 https://github.com/bombela/backward-cpp.git && \
    cd backward-cpp && cp backward.hpp /usr/local/include/ && \
    cd /tmp && rm -rf backward-cpp

RUN git clone --depth 1 https://github.com/Neargye/magic_enum.git && \
    cd magic_enum && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf magic_enum

RUN git clone --depth 1 https://github.com/jeremy-rifkin/cpptrace.git && \
    cd cpptrace && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf cpptrace

RUN git clone --depth 1 https://github.com/marzer/tomlplusplus.git && \
    cd tomlplusplus && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf tomlplusplus

RUN git clone --depth 1 https://github.com/TartanLlama/expected.git && \
    cd expected && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf expected

RUN git clone --depth 1 --branch v3.3.0 https://github.com/catchorg/Catch2.git && \
    cd Catch2 && mkdir build && cd build && \
    cmake .. && make install && \
    cd /tmp && rm -rf Catch2

# Compilar OpenCV 4.10 con CUDA (usar GCC-9)
RUN apt-get update && apt-get install -y gcc-9 g++-9 && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-9

ENV CC=/usr/bin/gcc-9
ENV CXX=/usr/bin/g++-9

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
        -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
        -D CUDA_ARCH_BIN="8.7" \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_V4L=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GTK=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_TBB=OFF \
        -D WITH_EIGEN=ON \
        -D WITH_LAPACK=ON \
        -D WITH_OPENJPEG=OFF \
        -D WITH_JASPER=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_opencv_cudacodec=ON \
        -D CMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
        .. \
    && make -j2 \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

RUN pkg-config --modversion opencv4 \
    && echo "OpenCV instalado con éxito"

# Volver a GCC-11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-11

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# ONNXRuntime
WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz -C /opt \
    && ln -s /opt/onnxruntime-linux-aarch64-1.16.3 /opt/onnxruntime \
    && rm onnxruntime-linux-aarch64-1.16.3.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime

# Python 3.8 + Jupyter
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-numpy \
    python3-zmq \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN python3 -m pip install --no-cache-dir --upgrade pip==24.0

RUN pip3 install --no-cache-dir \
    "packaging>=21.0" \
    setuptools \
    wheel

RUN pip3 install --no-cache-dir "numpy<2.0"

RUN pip3 install --no-cache-dir \
    importlib_resources \
    importlib_metadata \
    platformdirs \
    pygments \
    "Cython<3.0"

RUN pip3 install --no-cache-dir --no-build-isolation "pyzmq<26.0"

RUN pip3 install --no-cache-dir \
    "notebook<7.0" \
    "jupyterlab<4.0" \
    ipykernel

RUN pip3 install --no-cache-dir \
    matplotlib \
    pandas \
    scipy \
    scikit-learn \
    seaborn \
    pillow \
    tqdm

RUN jupyter notebook --generate-config && \
    python3 -c "from notebook.auth import passwd; print(passwd('nvidia'))" > /tmp/jupyter_pass.txt && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = '$(cat /tmp/jupyter_pass.txt)'" >> ~/.jupyter/jupyter_notebook_config.py && \
    rm /tmp/jupyter_pass.txt && \
    echo "=== Jupyter configurado correctamente ===" && \
    jupyter --version

EXPOSE 8888

# Workspace y zona horaria
WORKDIR /workspace

# Limpieza final
#RUN apt-get autoremove -y && apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip /root/.cache || true

RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    echo "America/Lima" > /etc/timezone && \
    dpkg-reconfigure -fnoninteractive tzdata

# Copiar y ejecutar run.sh al iniciar
COPY run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

CMD ["/usr/local/bin/run.sh"]