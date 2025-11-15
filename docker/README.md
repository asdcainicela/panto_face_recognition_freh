# Face Recognition C++20

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
docker build -t l4t-ml-cpp-py .
DOCKER_BUILDKIT=0 docker build --no-cache -t l4t-mlcpp-py . 2>&1 | tee build.log
./run.sh
```

# Dentro del contenedor:
en el dockerfile copia test/ en /opt/ y da permisos
cd /opt/tests
./verify_all.sh

## Workspace

`~/jetson_workspace/` → `/workspace`

## Stack

C++20 | GCC 11 | CMake 3.28 | OpenCV 4.10 | CUDA 11.4 | TensorRT