FROM dustynv/l4t-ml:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH} \
    DISPLAY=:0 \
    OPENBLAS_CORETYPE=ARMV8

RUN ldconfig && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig

RUN if [ ! -d "/usr/local/include/onnxruntime" ]; then \
        cd /tmp && \
        python3 -c "import onnxruntime; import shutil; shutil.copytree(onnxruntime.__path__[0] + '/capi/include/onnxruntime', '/usr/local/include/onnxruntime')" || \
        (pip3 download onnxruntime-gpu && \
         unzip -q onnxruntime_gpu*.whl && \
         mkdir -p /usr/local/include && \
         cp -r onnxruntime/capi/include/onnxruntime /usr/local/include/ && \
         cd / && rm -rf /tmp/*); \
    fi

RUN if [ ! -f "/usr/local/lib/libonnxruntime.so" ]; then \
        find /usr -name "libonnxruntime.so*" -exec cp {} /usr/local/lib/ \; && \
        ldconfig; \
    fi

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

RUN mkdir -p /etc/apt/preferences.d && \
    echo 'Package: libopencv*' > /etc/apt/preferences.d/opencv-block && \
    echo 'Pin: release *' >> /etc/apt/preferences.d/opencv-block && \
    echo 'Pin-Priority: -1' >> /etc/apt/preferences.d/opencv-block && \
    echo '' >> /etc/apt/preferences.d/opencv-block && \
    echo 'Package: *opencv*' >> /etc/apt/preferences.d/opencv-block && \
    echo 'Pin: release *' >> /etc/apt/preferences.d/opencv-block && \
    echo 'Pin-Priority: -1' >> /etc/apt/preferences.d/opencv-block

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libavresample4 \
    libv4l-0 \
    libxvidcore4 \
    libx264-155 \
    libx265-179 \
    libvpx6 \
    ffmpeg \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/* || \
    echo "Some FFmpeg dev packages failed, continuing..."

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/* || \
    echo "GStreamer dev packages failed, runtime is available"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-dev \
    libgtk2.0-dev \
    libxrender-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxi-dev \
    libxext6 \
    libsm6 \
    libice6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    gfortran \
    libeigen3-dev \
    libtbb-dev \
    libtbb2 \
    libdc1394-22-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

RUN apt-mark unhold libopencv libopencv-dev libopencv-core libopencv-core-dev \
    libopencv-highgui libopencv-highgui-dev libopencv-imgproc libopencv-imgproc-dev \
    libopencv-python libopencv-contrib libopencv-contrib-dev 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libxi-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf \
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

RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    echo "America/Lima" > /etc/timezone

RUN git config --global user.email "gerald.cainicela.a@gmail.com" \
    && git config --global user.name "userasd" \
    && git config --global --add safe.directory '*'

RUN pip3 install --no-cache-dir \
    packaging \
    setuptools \
    wheel \
    tqdm \
    matplotlib \
    seaborn \
    scikit-learn

RUN pip3 install --no-cache-dir \
    jupyterlab-catppuccin \
    jupyterlab-hale-theme \
    jupyterlab-theme-nord \
    jupyterlab-night || echo "Algunos temas no se pudieron instalar, continuando..."

RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    echo '{ \
      "theme": "Catppuccin Mocha", \
      "theme-scrollbars": true, \
      "overrides": { \
        "preferredDarkTheme": "Catppuccin Mocha", \
        "preferredLightTheme": "Hale" \
      } \
    }' > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/translation-extension && \
    echo '{ \
      "locale": "es-ES" \
    }' > ~/.jupyter/lab/user-settings/@jupyterlab/translation-extension/plugin.jupyterlab-settings

RUN mkdir -p ~/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" > ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.token = 'nvidia'" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_server_config.py

RUN mkdir -p /workspace /var/log /opt/tests

WORKDIR /workspace

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY test/ /opt/tests/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    find /opt/tests -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod +x {} \; 2>/dev/null || true

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sleep", "infinity"]