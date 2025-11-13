FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH} \
    DISPLAY=:0 \
    OPENCV_VERSION=4.10.0 \
    CMAKE_VERSION=3.28.1 \
    OPENBLAS_CORETYPE=ARMV8

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

WORKDIR /tmp
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz \
    && tar -xzf cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cmake /usr/local/bin/cmake \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/ctest /usr/local/bin/ctest \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cpack /usr/local/bin/cpack \
    && rm -f cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    libopenjp2-7-dev \
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
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
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
    libzmq3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libwebsocketpp-dev \
    libmicrohttpd-dev \
    libboost-all-dev \
    libarmadillo-dev \
    libopenblas-dev \
    liblapack-dev \
    libgsl-dev \
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
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/src/gtest && \
    cmake . && \
    make && \
    cp lib/*.a /usr/lib || cp *.a /usr/lib || true

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

# Instalar Python ANTES de compilar OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    packaging \
    setuptools \
    wheel \
    "numpy<2.0" \
    Cython

# Cambiar a GCC 9 para compilar OpenCV
RUN update-alternatives --set gcc /usr/bin/gcc-9

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
        -D CUDA_ARCH_BIN="8.7" \
        -D CUDA_ARCH_PTX="" \
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
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_opencv_cudacodec=ON \
        -D CMAKE_CXX_FLAGS="-O3" \
        .. \
    && make -j2 \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

RUN update-alternatives --set gcc /usr/bin/gcc-11

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Instalar Python temporal solo para build
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

ENV PYTHON_BUILD=/usr/bin/python3.10

# Compilación de ONNX Runtime con soporte CUDA + TensorRT

WORKDIR /tmp
RUN git clone --recursive https://github.com/microsoft/onnxruntime.git && \
    cd onnxruntime && \
    ./build.sh \
        --config Release \
        --update --build \
        --build_shared_lib \
        --parallel \
        --use_tensorrt \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/aarch64-linux-gnu \
        --tensorrt_home /usr/lib/aarch64-linux-gnu \
        --skip_tests \
        --skip_onnx_tests \
        --skip_submodule_sync && \
    cp -r build/Linux/Release/lib* /usr/local/lib/ && \
    cp -r include /usr/local/include/onnxruntime && \
    ldconfig && \
    cd / && rm -rf /tmp/onnxruntime

RUN echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf \
    && echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/tegra.conf \
    && echo "/usr/lib/aarch64-linux-gnu/tegra-egl" >> /etc/ld.so.conf.d/tegra.conf \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen es_ES.UTF-8 en_US.UTF-8 \
    && update-locale LANG=es_ES.UTF-8 LANGUAGE=es_ES:es LC_ALL=es_ES.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=es_ES.UTF-8 \
    LANGUAGE=es_ES:es \
    LC_ALL=es_ES.UTF-8
        
# Paquetes Python y JupyterLab
RUN pip3 install --no-cache-dir \
    notebook \
    jupyterlab \
    ipykernel \
    matplotlib \
    pandas \
    scipy \
    scikit-learn \
    seaborn \
    pillow \
    tqdm \
    onnxruntime

# Temas para JupyterLab
RUN pip3 install --no-cache-dir \
    jupyterlab-catppuccin \
    jupyterlab-hale-theme \
    jupyterlab-theme-nord \
    jupyterlab-night || echo "Algunos temas no se pudieron instalar, continuando..."

# Verificar extensiones instaladas
RUN jupyter labextension list || true

# Configuracion JupyterLab
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    echo '{ \
      "theme": "Catppuccin Mocha", \
      "theme-scrollbars": true, \
      "overrides": { \
        "preferredDarkTheme": "Catppuccin Mocha", \
        "preferredLightTheme": "Hale" \
      } \
    }' > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# Configuración de traducción a español
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/translation-extension && \
    echo '{ \
      "locale": "es-ES" \
    }' > ~/.jupyter/lab/user-settings/@jupyterlab/translation-extension/plugin.jupyterlab-settings

# Configuración del servidor Jupyter
RUN mkdir -p ~/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" > ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.token = 'nvidia'" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_server_config.py

EXPOSE 8888

RUN git config --global user.email "gerald.cainicela.a@gmail.com" \
    && git config --global user.name "userasd" \
    && git config --global --add safe.directory '*'

RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    echo "America/Lima" > /etc/timezone

RUN mkdir -p /workspace /var/log /opt/tests

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY test/ /opt/tests/
RUN chmod +x /opt/tests/*.sh /opt/tests/*.py

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sleep", "infinity"]