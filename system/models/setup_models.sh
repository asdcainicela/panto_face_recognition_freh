#!/bin/bash
set -e

download_model() {
    url="$1"
    output="$2"
    min_size="$3"

    if [ -f "$output" ]; then
        size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "$output OK ($(du -h "$output" | cut -f1))"
            return 0
        fi
        rm -f "$output"
    fi

    wget -q --timeout=30 --tries=2 "$url" -O "$output" 2>/dev/null || \
    curl -L --max-time 60 --retry 2 -o "$output" "$url" 2>/dev/null

    size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
    if [ "$size" -ge "$min_size" ]; then
        echo "$output OK ($(du -h "$output" | cut -f1))"
        return 0
    fi

    rm -f "$output"
    return 1
}

# Descarga modelos principales
download_model "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx" "retinaface.onnx" 100000000
download_model "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx" "arcface_r100.onnx" 120000000

# Real-ESRGAN x4
SUCCESS=false
URLS=(
    # GitHub Releases oficiales
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx"
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus_anime_6B.onnx"
    
    # HuggingFace - Repositorios oficiales
    "https://huggingface.co/Xintao/Real-ESRGAN/resolve/main/weights/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/Xintao/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/xinntao/realesrgan/resolve/main/RealESRGAN_x4plus.onnx"
    
    # HuggingFace - Mirrors y forks populares
    "https://huggingface.co/rockeycoss/RealESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx"
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/OpenVINO/realesrgan/resolve/main/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/onnx/models/resolve/main/vision/super_resolution/RealESRGAN_x4plus.onnx"
    
    # HuggingFace - Variantes alternativas
    "https://huggingface.co/spaces/akhaliq/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
    "https://huggingface.co/rockeycoss/Real-ESRGAN-General/resolve/main/realesr-general-x4v3.onnx"
    "https://huggingface.co/nateraw/Real-ESRGAN/resolve/main/weights/RealESRGAN_x4plus.onnx"
    
    # ModelScope (Alibaba Cloud)
    "https://www.modelscope.cn/models/damo/cv_rrdb_image-super-resolution_x4/resolve/master/RealESRGAN_x4plus.onnx"
    "https://modelscope.cn/models/iic/cv_rrdb_image-super-resolution/resolve/master/RealESRGAN_x4plus.onnx"
    
    # Kaggle Models
    "https://www.kaggle.com/models/xinntao/realesrgan/onnx/x4plus/1/RealESRGAN_x4plus.onnx"
    
    # Google Drive (mirrors públicos)
    "https://drive.google.com/uc?export=download&id=1KTVZDWzkO3q_3MV6NdLd5pJaVfj-S0dN"
    "https://drive.google.com/uc?export=download&id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene"
    
    # Dropbox (mirrors públicos)
    "https://www.dropbox.com/s/wxvtzqb9ys8cz9j/RealESRGAN_x4plus.onnx?dl=1"
    
    # MediaFire
    "https://www.mediafire.com/file/wxvtzqb9ys8cz9j/RealESRGAN_x4plus.onnx/file"
    
    # GitLab Mirrors
    "https://gitlab.com/xinntao/Real-ESRGAN/-/raw/master/weights/RealESRGAN_x4plus.onnx"
    
    # Raw GitHubusercontent
    "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/weights/RealESRGAN_x4plus.onnx"
    
    # ONNX Model Zoo
    "https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
    
    # Replicate CDN
    "https://replicate.delivery/pbxt/RealESRGAN_x4plus.onnx"
    
    # Cloudflare R2 (mirrors)
    "https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/RealESRGAN_x4plus.onnx"
    
    # Backblaze B2 (mirrors públicos)
    "https://f000.backblazeb2.com/file/ai-models/RealESRGAN_x4plus.onnx"
    
    # SourceForge
    "https://sourceforge.net/projects/realesrgan/files/RealESRGAN_x4plus.onnx/download"
    "https://downloads.sourceforge.net/project/realesrgan/RealESRGAN_x4plus.onnx"
    
    # Archive.org
    "https://archive.org/download/realesrgan-models/RealESRGAN_x4plus.onnx"
)

for u in "${URLS[@]}"; do
    download_model "$u" "realesr_x4.onnx" 60000000 && { SUCCESS=true; break; }
done

ls -lh *.onnx
[ -f "realesr_x4.onnx" ] && echo "realesr_x4.onnx OK" || echo "realesr_x4.onnx faltante"
