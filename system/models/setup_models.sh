#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Descargando modelos ONNX para PANTO ==="
echo ""

download_model() {
    url="$1"
    output="$2"
    min_size="$3"

    if [ -f "$output" ]; then
        size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ "$size" -ge "$min_size" ]; then
            echo "✓ $output OK ($(du -h "$output" | cut -f1))"
            return 0
        fi
        echo "  $output existe pero es muy pequeño, re-descargando..."
        rm -f "$output"
    fi

    echo "  Descargando desde: $url"
    
    wget -q --timeout=30 --tries=2 "$url" -O "$output" 2>/dev/null || \
    curl -L --max-time 60 --retry 2 -o "$output" "$url" 2>/dev/null || \
    return 1

    size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
    if [ "$size" -ge "$min_size" ]; then
        echo "✓ $output descargado ($(du -h "$output" | cut -f1))"
        return 0
    fi

    echo "✗ $output muy pequeño (${size} bytes < ${min_size})"
    rm -f "$output"
    return 1
}

# ============================================
# 1. RETINAFACE (Detección de rostros)
# ============================================
echo "1. RetinaFace (detección de rostros)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

SUCCESS=false
for u in "${RETINAFACE_URLS[@]}"; do
    if download_model "$u" "retinaface.onnx" 20000000; then
        SUCCESS=true
        break
    fi
    sleep 1
done

if [ "$SUCCESS" = false ]; then
    echo "❌ RetinaFace no descargado"
    echo ""
else
    echo ""
fi

# ============================================
# 2. ARCFACE (Reconocimiento facial)
# ============================================
echo "2. ArcFace (reconocimiento facial)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

SUCCESS=false
for u in "${ARCFACE_URLS[@]}"; do
    # Si es ZIP, necesitamos extraer
    if [[ "$u" == *.zip ]]; then
        if wget -q --timeout=30 --tries=2 "$u" -O buffalo_l.zip 2>/dev/null; then
            unzip -q -o buffalo_l.zip "*/w600k_r50.onnx" 2>/dev/null && \
            mv buffalo_l/w600k_r50.onnx arcface_r100.onnx 2>/dev/null && \
            rm -rf buffalo_l buffalo_l.zip && \
            SUCCESS=true && \
            echo "✓ arcface_r100.onnx extraído del ZIP" && \
            break
        fi
    else
        if download_model "$u" "arcface_r100.onnx" 100000000; then
            SUCCESS=true
            break
        fi
    fi
    sleep 1
done

if [ "$SUCCESS" = false ]; then
    echo "❌ ArcFace no descargado"
    echo ""
else
    echo ""
fi

# ============================================
# 3. REAL-ESRGAN (Super-resolución)
# ============================================
echo "3. Real-ESRGAN (super-resolución)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

REALESR_URLS=(
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

SUCCESS=false
for u in "${REALESR_URLS[@]}"; do
    if download_model "$u" "realesr_x4.onnx" 60000000; then
        SUCCESS=true
        break
    fi
    sleep 1
done

if [ "$SUCCESS" = false ]; then
    echo "❌ Real-ESRGAN no descargado"
    echo ""
else
    echo ""
fi

# ============================================
# RESUMEN FINAL
# ============================================
echo ""
echo "=== RESUMEN ==="
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh *.onnx 2>/dev/null || echo "No hay archivos .onnx"
echo ""

# Verificar modelos críticos
CRITICAL=true
if [ -f "retinaface.onnx" ]; then
    echo "✓ RetinaFace disponible"
else
    echo "✗ RetinaFace FALTANTE (crítico)"
    CRITICAL=false
fi

if [ -f "arcface_r100.onnx" ]; then
    echo "✓ ArcFace disponible"
else
    echo "✗ ArcFace FALTANTE (crítico)"
    CRITICAL=false
fi

if [ -f "realesr_x4.onnx" ]; then
    echo "✓ Real-ESRGAN disponible"
else
    echo "⚠ Real-ESRGAN faltante (opcional)"
fi

echo ""

if [ "$CRITICAL" = false ]; then
    echo "❌ DESCARGA MANUAL NECESARIA"
    echo ""
    echo "Modelos faltantes deben descargarse manualmente:"
    echo ""
    echo "1. RetinaFace:"
    echo "   https://github.com/deepinsight/insightface/releases/tag/v0.7"
    echo "   Archivo: retinaface_r50_v1.onnx (27.8 MB)"
    echo ""
    echo "2. ArcFace:"
    echo "   https://github.com/deepinsight/insightface/releases/tag/v0.7"
    echo "   Archivo: buffalo_l.zip -> extraer w600k_r50.onnx"
    echo ""
    echo "Colócalos en: $(pwd)/"
    echo ""
    exit 1
else
    echo "✓ Todos los modelos críticos disponibles"
    echo ""
    echo "Puedes compilar y ejecutar PANTO:"
    echo "  cd .."
    echo "  ./build.sh"
    echo "  ./build/bin/test_detector models/retinaface.onnx test/img/test1.png"
    echo ""
fi