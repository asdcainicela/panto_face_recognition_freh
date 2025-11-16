g++ verify_engine.cpp -o verify_engine \
    -std=c++17 -O3 \
    -I/usr/local/cuda/include \
    -I/usr/include/aarch64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/aarch64-linux-gnu \
    -lnvinfer -lcudart -lspdlog \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/lib/aarch64-linux-gnu