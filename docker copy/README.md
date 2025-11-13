# Face Recognition C++20
primero debemos ver la version del jetpack

 dpkg -l | grep "nvidia-l4t-core"
 nos debe dar como resultado 5.1.1 
 jorinbriq06@jorinbriq06:~/jetson_workspace/panto_face_recognition_freh/docker$  dpkg -l | grep "nvidia-l4t-core"
ii  nvidia-l4t-core                            35.4.1-20230801124926                arm64        NVIDIA Core Package

## Setup

```bash
cd ~
mkdir -p  ~/jetson_workspace
cd ~/jetson_workspace
git clone https://asdcainicela:{}@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || true
git clone https://asdcainicela:{}@github.com/asdcainicela/panto_face_recognition_freh.git2>/dev/null || true
cd ~/jetson_workspace/panto_face_recognition_freh/docker
chmod +x run.sh
chmod +x docker-entrypoint.sh
docker build -t l4t-mlcpp-py .
./run.sh
```

# Dentro del contenedor:
cd /opt/tests
./verify_all.sh

## Workspace

`~/jetson_workspace/` → `/workspace`

## Stack

C++20 | GCC 11 | CMake 3.28 | OpenCV 4.10 | CUDA 11.4 | TensorRT