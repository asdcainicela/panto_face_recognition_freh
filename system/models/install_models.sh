#!/bin/bash
# install_models.sh - Descarga SCRFD + ArcFace + Real-ESRGAN

set -e
cd "$(dirname "$0")"

echo "[INFO] Preparando modelos ONNX (SCRFD)..."

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
        echo "[WARN] $output es muy pequeño, reintentando..."
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

# ------------------------------------
# SCRFD: 2.5G y 10G (mirrors verificados)
# ------------------------------------
SCRFD_2_5G="scrfd_2.5g_bnkps.onnx"
SCRFD_10G="scrfd_10g_bnkps.onnx"

SCRFD_2_5G_URLS=(
  "https://huggingface.co/MonsterMMORPG/files1/resolve/main/scrfd_2.5g_bnkps.onnx"
  "https://sourceforge.net/projects/insightface.mirror/files/v0.7/scrfd_person_2.5g.onnx/download"
)

SCRFD_10G_URLS=(
  "https://huggingface.co/okaris/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
)

# Descargar 2.5G
if [ ! -f "$SCRFD_2_5G" ]; then
  echo "[INFO] Descargando SCRFD 2.5G..."
  for u in "${SCRFD_2_5G_URLS[@]}"; do
    echo "[INFO] intentando: $u"
    if download_model "$u" "$SCRFD_2_5G" 3000000; then
      echo "[OK] SCRFD 2.5G descargado desde: $u"
      break
    fi
  done
fi

# Descargar 10G (opcional)
if [ ! -f "$SCRFD_10G" ]; then
  echo "[INFO] Descargando SCRFD 10G..."
  for u in "${SCRFD_10G_URLS[@]}"; do
    echo "[INFO] intentando: $u"
    if download_model "$u" "$SCRFD_10G" 15000000; then
      echo "[OK] SCRFD 10G descargado desde: $u"
      break
    fi
  done
fi

# ============================================
# 2. ArcFace - Reconocimiento facial
# ============================================

if [ ! -f "arcface_r100.onnx" ]; then
    echo "[INFO] Descargando ArcFace..."
    ARCFACE_URLS=(
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
        "https://huggingface.co/MonsterMMORPG/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    )
    
    for u in "${ARCFACE_URLS[@]}"; do
        if [[ "$u" == *".zip" ]]; then
            if download_model "$u" "buffalo_l.zip" 100000000; then
                unzip -q buffalo_l.zip
                [ -f "buffalo_l/w600k_r50.onnx" ] && cp buffalo_l/w600k_r50.onnx arcface_r100.onnx
                rm -rf buffalo_l buffalo_l.zip
                break
            fi
        else
            if download_model "$u" "arcface_r100.onnx" 100000000; then
                break
            fi
        fi
    done
fi

# ============================================
# 3. Real-ESRGAN - Super-resolución
# ============================================

if [ ! -f "realesrgan_x4plus.onnx" ]; then
    echo "[INFO] Descargando Real-ESRGAN..."
    ESRGAN_URLS=(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
        "https://huggingface.co/rockeycoss/RealESRGAN/resolve/main/RealESRGAN_x4plus.onnx"
    )
    
    for u in "${ESRGAN_URLS[@]}"; do
        if download_model "$u" "realesrgan_x4plus.onnx" 40000000; then
            break
        fi
    done
fi

# ============================================
# 4. Verificación final
# ============================================

echo ""
echo "[INFO] Verificando modelos..."
OK=true

if [ -f "scrfd_2.5g_bnkps.onnx" ]; then
    echo "[OK] scrfd_2.5g_bnkps.onnx (RECOMENDADO)"
else
    echo "[WARN] Falta scrfd_2.5g_bnkps.onnx"
    OK=false
fi

[ -f "scrfd_10g_bnkps.onnx" ] && echo "[OK] scrfd_10g_bnkps.onnx (opcional)" || echo "[INFO] scrfd_10g_bnkps.onnx no descargado (opcional)"

if [ -f "arcface_r100.onnx" ]; then
    echo "[OK] arcface_r100.onnx"
else
    echo "[WARN] Falta arcface_r100.onnx"
    OK=false
fi

[ -f "realesrgan_x4plus.onnx" ] && echo "[OK] realesrgan_x4plus.onnx" || echo "[INFO] realesrgan_x4plus.onnx no descargado (opcional)"

echo ""

if [ "$OK" = false ]; then
    echo "[ERR] Faltan modelos críticos. Descargar manualmente desde:"
    echo "  SCRFD: https://github.com/deepinsight/insightface/releases/tag/v0.7"
    echo "  HuggingFace: https://huggingface.co/public-data/insightface/tree/main/models"
    exit 1
else
    echo "[OK] Modelos SCRFD listos. Siguiente paso:"
    echo ""
    echo "  # Convertir a TensorRT:"
    echo "  /usr/src/tensorrt/bin/trtexec \\"
    echo "      --onnx=scrfd_2.5g_bnkps.onnx \\"
    echo "      --saveEngine=engines/scrfd_2.5g.engine \\"
    echo "      --fp16 \\"
    echo "      --workspace=3072"
    echo ""
    echo "  # O con script:"
    echo "  cd .. && ./run.sh convert models/scrfd_2.5g_bnkps.onnx models/engines/scrfd_2.5g.engine"
fi