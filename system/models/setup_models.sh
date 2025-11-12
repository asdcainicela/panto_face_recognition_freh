#!/bin/bash
set -e

echo "[INFO] Descargando modelos ONNX..."

# ====================================================================
# Función auxiliar: descarga con reintentos y verificación de tamaño
# ====================================================================
download_model() {
    local url="$1"
    local output="$2"
    local min_size="$3"
    local name=$(basename "$output")
    
    # Si ya existe y es válido, skip
    if [ -f "$output" ] && [ -s "$output" ]; then
        local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "  - $name OK ($(du -h "$output" | cut -f1))"
            return 0
        fi
        rm -f "$output"
    fi
    
    echo "  - Descargando $name..."
    
    # Intentar con wget
    if wget -q --show-progress --timeout=30 --tries=2 --no-check-certificate "$url" -O "$output" 2>/dev/null; then
        local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "    [OK] $name descargado ($(du -h "$output" | cut -f1))"
            return 0
        else
            echo "    [FAIL] $name incompleto ($(du -h "$output" | cut -f1))"
            rm -f "$output"
            return 1
        fi
    fi
    
    # Intentar con curl
    echo "    wget falló, intentando con curl..."
    if curl -L --progress-bar --max-time 60 --retry 2 -o "$output" "$url" 2>/dev/null; then
        local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "    [OK] $name descargado ($(du -h "$output" | cut -f1))"
            return 0
        else
            echo "    [FAIL] $name incompleto ($(du -h "$output" | cut -f1))"
            rm -f "$output"
            return 1
        fi
    fi
    
    return 1
}

# ====================================================================
# 1. RetinaFace (detección rostros) - ~105MB
# ====================================================================
download_model \
    "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx" \
    "retinaface.onnx" \
    100000000

# ====================================================================
# 2. ArcFace R100 (reconocimiento) - ~131MB
# ====================================================================
download_model \
    "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx" \
    "arcface_r100.onnx" \
    120000000

# ====================================================================
# 3. Real-ESRGAN x4 (super-resolución) - ~67MB
#    TODAS LAS FUENTES POSIBLES
# ====================================================================
echo "  - Descargando realesr_x4.onnx..."
SUCCESS=false

# LISTA COMPLETA DE URLS (30+ fuentes)
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

for URL in "${URLS[@]}"; do
    echo "    probando: $URL"
    if download_model "$URL" "realesr_x4.onnx" 60000000; then
        SUCCESS=true
        break
    fi
done

# Si ninguna funcionó, dar opciones manuales
if [ "$SUCCESS" = false ]; then
    echo ""
    echo "  ╔════════════════════════════════════════════════════════════════╗"
    echo "  ║  [ERROR] No se pudo descargar realesr_x4.onnx automáticamente ║"
    echo "  ╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Opciones manuales:"
    echo ""
    echo "  1) Descargar desde navegador:"
    echo "     https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
    echo "     Guardar como: $(pwd)/realesr_x4.onnx"
    echo ""
    echo "  2) Usar modelo alternativo (más pequeño):"
    echo "     wget https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx -O realesr_x4.onnx"
    echo ""
    echo "  3) Deshabilitar super-resolución temporalmente:"
    echo "     cd ../configs && sed -i 's/enabled = true/enabled = false/g' config_*.toml"
    echo ""
    echo "  4) Usar git-lfs para clonar el repo completo:"
    echo "     git lfs clone https://github.com/xinntao/Real-ESRGAN.git"
    echo "     cp Real-ESRGAN/weights/RealESRGAN_x4plus.onnx ."
    echo ""
fi

# ====================================================================
# 4. Verificación final
# ====================================================================
echo ""
echo "[INFO] Modelos descargados:"
ls -lh *.onnx 2>/dev/null || echo "  - Ninguno disponible"

# ====================================================================
# 5. TensorRT (opcional)
# ====================================================================
echo ""
echo "[INFO] TensorRT:"
if command -v trtexec &>/dev/null; then
    echo "  - TensorRT detectado. Puedes convertir con:"
    echo "    trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=2048"
else
    echo "  - TensorRT no disponible (usará ONNX Runtime)"
fi

echo ""
if [ "$SUCCESS" = true ]; then
    echo "[DONE] ✓ Todos los modelos descargados correctamente."
else
    echo "[WARN] ⚠ Falta realesr_x4.onnx - ver opciones manuales arriba."
fi