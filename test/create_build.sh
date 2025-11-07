apt-get update && apt-get install -y libspdlog-dev

mkdir -p build
cd build
cmake ..
make -j$(nproc)
