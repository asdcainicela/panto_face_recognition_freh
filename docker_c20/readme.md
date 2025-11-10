# Face Recognition C++20

## Setup

```bash
cd ~
mkdir -p  ~/jetson_workspace
cd ~/jetson_workspace
git clone https://asdcainicela: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@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || true
git clone https://asdcainicela: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@github.com/asdcainicela/panto_face_recognition_freh.git 2>/dev/null || true
cd ~/jetson_workspace/panto_face_recognition_freh/docker_c20
docker build -t l4t-cpp20 .
chmod +x run.sh
./run.sh
```

## Workspace

`~/jetson_workspace/` → `/workspace`

## Stack

C++20 | GCC 11 | CMake 3.28 | OpenCV 4.10 | CUDA 11.4 | TensorRT