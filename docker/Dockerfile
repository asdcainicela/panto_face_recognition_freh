FROM dustynv/l4t-ml:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH} \
    DISPLAY=:0

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    unzip \
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libxi-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Descargar e instalar ONNX Runtime desde wheel oficial de NVIDIA
RUN wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl \
    -O onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl && \
    unzip onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl && \
    mkdir -p /opt/onnxruntime/include /opt/onnxruntime/lib && \
    cp -r onnxruntime/capi/include/onnxruntime /opt/onnxruntime/include/ && \
    cp onnxruntime/capi/*.so* /opt/onnxruntime/lib/ && \
    rm -rf onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl onnxruntime onnxruntime.libs onnxruntime-1.15.1.dist-info

# Verificar la instalación desde Python (si ya existía)
RUN python3 -c "\
import onnxruntime as ort; \
print('ONNX Runtime version:', ort.__version__); \
print('Available providers:', ort.get_available_providers());" || echo "Python ONNX Runtime check skipped"

RUN ln -sf /opt/onnxruntime/include/onnxruntime /usr/local/include/onnxruntime \
    && ln -sf /opt/onnxruntime/lib/*.so* /usr/local/lib/ \
    && ldconfig

ENV ONNXRUNTIME_DIR=/opt/onnxruntime \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

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

RUN git config --global user.email "gerald.cainicela.a@gmail.com" \
    && git config --global user.name "userasd" \
    && git config --global --add safe.directory '*'

RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    echo "America/Lima" > /etc/timezone

RUN mkdir -p /workspace /var/log /opt/tests

WORKDIR /workspace

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY test/ /opt/tests/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    find /opt/tests -type f \( -name "*.sh" -o -name "*.cpp" \) -exec chmod +x {} \; 2>/dev/null || true

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]