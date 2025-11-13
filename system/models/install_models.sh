#!/bin/bash
# setup_models.sh - Descarga de modelos ONNX con prioridad a buffalo_l.zip + Real-ESRGAN

set -e
cd "$(dirname "$0")"

echo "[INFO] Preparando modelos ONNX..."

download_model() {
    url="$1"
    output="$2"
    min_size="$3"

    if [ -f "$output" ]; then
        size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "[OK] $output ya existe ($(du -h "$output" | cut -f1))"
            return 0
        fi
        echo "[WARN] $output es muy pequeno, reintentando..."
        rm -f "$output"
    fi

    echo "[INFO] Descargando: $url"
    wget -q --timeout=30 --tries=2 "$url" -O "$output" 2>/dev/null || \
    curl -L --max-time 60 --retry 2 -o "$output" "$url" 2>/dev/null || \
    return 1

    size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
    [ "$size" -ge "$min_size" ] && echo "[OK] $output descargado ($(du -h "$output" | cut -f1))" && return 0

    echo "[ERR] $output corrupto o incompleto"
    rm -f "$output"
    return 1
}

BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_ZIP="buffalo_l.zip"
MIN_SIZE=100000000

# 1. Descargar buffalo_l.zip
if [ ! -f "retinaface.onnx" ] || [ ! -f "arcface_r100.onnx" ]; then
    echo "[INFO] Descargando paquete buffalo_l.zip..."
    if download_model "$BUFFALO_URL" "$BUFFALO_ZIP" "$MIN_SIZE"; then
        rm -rf buffalo_l
        unzip -q "$BUFFALO_ZIP"
        [ -f "buffalo_l/det_10g.onnx" ] && cp buffalo_l/det_10g.onnx retinaface.onnx && echo "[OK] retinaface.onnx extraido"
        [ -f "buffalo_l/w600k_r50.onnx" ] && cp buffalo_l/w600k_r50.onnx arcface_r100.onnx && echo "[OK] arcface_r100.onnx extraido"
        rm -rf buffalo_l "$BUFFALO_ZIP"
    else
        echo "[WARN] No se pudo descargar buffalo_l.zip, usando mirrors..."
    fi
fi

# 2. Mirrors alternativos
if [ ! -f "retinaface.onnx" ]; then
    RETINAFACE_URLS=(
        # GitHub Releases - InsightFace oficial
        "https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50_v1.onnx"
        "https://github.com/deepinsight/insightface/releases/download/v0.7/det_10g.onnx"
        
        # HuggingFace - Mirrors oficiales
        "https://huggingface.co/ezioruan/retinaface_onnx/resolve/main/retinaface-R50.onnx"
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx"
        "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx"
        
        # ONNX Model Zoo
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/retinaface/model/retinaface-R50.onnx"
        
        # Mirrors alternativos
        "https://huggingface.co/rockeycoss/retinaface/resolve/main/retinaface_r50_v1.onnx"
        "https://huggingface.co/MonsterMMORPG/insightface/resolve/main/models/buffalo_l/det_10g.onnx"
        
        # OneDrive mirror
        "https://onedrive.live.com/download?cid=CEC0E0B8342BA033&resid=CEC0E0B8342BA033%211059&authkey=AKqtJ-lWvCLBflo"
    )
    for u in "${RETINAFACE_URLS[@]}"; do
        if download_model "$u" "retinaface.onnx" 20000000; then
            break
        fi
    done
fi

if [ ! -f "arcface_r100.onnx" ]; then
    ARCFACE_URLS=(
        # HuggingFace - Modelos oficiales
        "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
        
        # InsightFace GitHub Releases
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        
        # Mirrors alternativos
        "https://huggingface.co/MonsterMMORPG/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
        "https://huggingface.co/ezioruan/insightface_onnx/resolve/main/arcface_r100_v1.onnx"
        
        # ONNX Model Zoo
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcface-resnet100-8.onnx"
    )
    for u in "${ARCFACE_URLS[@]}"; do
        if download_model "$u" "arcface_r100.onnx" 100000000; then
            break
        fi
    done
fi

# 3. Superresolución (Real-ESRGAN)
if [ ! -f "realesrgan_x4plus.onnx" ] && [ ! -f "realesrgan_x4plus_anime_6B.onnx" ]; then
    echo "[INFO] Descargando modelo Real-ESRGAN..."
    ESRGAN_URLS=(
        # GitHub Releases oficiales
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx"
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus_anime_6B.onnx"
        
        # HuggingFace - Repositorios oficiales
        "https://huggingface.co/Xintao/Real-ESRGAN/resolve/main/weights/RealESRGAN_x4plus.onnx"
        "https://huggingface.co/Xintao/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
        "https://huggingface.co/xinntao/realesrgan/resolve/main/RealESRGAN_x4plus.onnx"
        
        # HuggingFace - Mirrors populares
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
        
        # Google Drive mirrors públicos
        "https://drive.google.com/uc?export=download&id=1KTVZDWzkO3q_3MV6NdLd5pJaVfj-S0dN"
        "https://drive.google.com/uc?export=download&id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene"
        
        # Dropbox mirrors
        "https://www.dropbox.com/s/wxvtzqb9ys8cz9j/RealESRGAN_x4plus.onnx?dl=1"
        
        # SourceForge
        "https://sourceforge.net/projects/realesrgan/files/RealESRGAN_x4plus.onnx/download"
        "https://downloads.sourceforge.net/project/realesrgan/RealESRGAN_x4plus.onnx"
        
        # Archive.org
        "https://archive.org/download/realesrgan-models/RealESRGAN_x4plus.onnx"
        
        # Raw GitHub
        "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/weights/RealESRGAN_x4plus.onnx"
    )
    for u in "${ESRGAN_URLS[@]}"; do
        if [[ "$u" == *anime* ]]; then
            if download_model "$u" "realesrgan_x4plus_anime_6B.onnx" 40000000; then
                break
            fi
        else
            if download_model "$u" "realesrgan_x4plus.onnx" 40000000; then
                break
            fi
        fi
    done
fi

# 4. Verificación final
echo "[INFO] Verificando modelos..."
OK=true

for f in retinaface.onnx arcface_r100.onnx realesrgan_x4plus.onnx realesrgan_x4plus_anime_6B.onnx; do
    [ -f "$f" ] && echo "[OK] $f" || echo "[WARN] Falta $f"
done

if [ ! -f "retinaface.onnx" ] || [ ! -f "arcface_r100.onnx" ]; then
    OK=false
fi

if [ "$OK" = false ]; then
    echo "[ERR] Faltan modelos. Descargar manualmente desde:"
    echo "  https://github.com/deepinsight/insightface/releases/tag/v0.7"
    echo "  https://github.com/xinntao/Real-ESRGAN/releases"
    exit 1
else
    echo "[OK] Todos los modelos listos. Puedes continuar con:"
    echo "  cd .. && ./build.sh"
fi
