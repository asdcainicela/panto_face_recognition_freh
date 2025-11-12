#!/bin/bash
# setup_models.sh - Script unificado para descargar modelos ONNX
# Usa buffalo_l.zip oficial de InsightFace

set -e
cd "$(dirname "$0")"

echo "╔═══════════════════════════════════════════════════╗"
echo "║   PANTO - Descarga de Modelos ONNX               ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ============================================
# 1. Descargar buffalo_l.zip (oficial)
# ============================================
echo "📦 Descargando buffalo_l.zip..."
echo "   Fuente: InsightFace GitHub Releases"
echo ""

BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_ZIP="buffalo_l.zip"

if [ -f "$BUFFALO_ZIP" ]; then
    echo "✓ $BUFFALO_ZIP ya existe ($(du -h "$BUFFALO_ZIP" | cut -f1))"
else
    echo "  Descargando desde:"
    echo "  $BUFFALO_URL"
    echo ""
    
    if wget -q --show-progress --timeout=60 "$BUFFALO_URL" -O "$BUFFALO_ZIP" 2>/dev/null; then
        echo "✓ Descarga completada"
    elif curl -L --progress-bar --max-time 120 -o "$BUFFALO_ZIP" "$BUFFALO_URL" 2>/dev/null; then
        echo "✓ Descarga completada"
    else
        echo "❌ Error al descargar buffalo_l.zip"
        echo ""
        echo "Descarga manual desde:"
        echo "  $BUFFALO_URL"
        echo ""
        echo "Luego colócalo en: $(pwd)/"
        echo "Y vuelve a ejecutar este script."
        exit 1
    fi
fi

echo ""

# ============================================
# 2. Verificar tamaño del ZIP
# ============================================
MIN_SIZE=100000000  # 100MB mínimo

if [ ! -f "$BUFFALO_ZIP" ]; then
    echo "❌ $BUFFALO_ZIP no encontrado"
    exit 1
fi

SIZE=$(stat -c%s "$BUFFALO_ZIP" 2>/dev/null || stat -f%z "$BUFFALO_ZIP" 2>/dev/null)

if [ "$SIZE" -lt "$MIN_SIZE" ]; then
    echo "❌ $BUFFALO_ZIP demasiado pequeño (${SIZE} bytes < ${MIN_SIZE})"
    echo "   El archivo puede estar corrupto."
    echo ""
    echo "Elimínalo y vuelve a ejecutar este script:"
    echo "  rm $BUFFALO_ZIP"
    echo "  ./setup_models.sh"
    exit 1
fi

echo "✓ buffalo_l.zip verificado ($(du -h "$BUFFALO_ZIP" | cut -f1))"
echo ""

# ============================================
# 3. Extraer modelos
# ============================================
echo "📂 Extrayendo modelos..."
echo ""

if [ -d "buffalo_l" ]; then
    echo "  Limpiando extracción anterior..."
    rm -rf buffalo_l
fi

unzip -q "$BUFFALO_ZIP"

if [ ! -d "buffalo_l" ]; then
    echo "❌ Error al extraer buffalo_l.zip"
    exit 1
fi

echo "✓ Archivos extraídos:"
ls -lh buffalo_l/ | tail -n +2

echo ""

# ============================================
# 4. Copiar modelos necesarios
# ============================================
echo "📋 Instalando modelos..."
echo ""

# Detector de rostros
if [ -f "buffalo_l/det_10g.onnx" ]; then
    cp buffalo_l/det_10g.onnx retinaface.onnx
    echo "✓ det_10g.onnx → retinaface.onnx ($(du -h retinaface.onnx | cut -f1))"
else
    echo "❌ det_10g.onnx no encontrado en buffalo_l/"
    exit 1
fi

# Reconocimiento facial
if [ -f "buffalo_l/w600k_r50.onnx" ]; then
    cp buffalo_l/w600k_r50.onnx arcface_r100.onnx
    echo "✓ w600k_r50.onnx → arcface_r100.onnx ($(du -h arcface_r100.onnx | cut -f1))"
else
    echo "❌ w600k_r50.onnx no encontrado en buffalo_l/"
    exit 1
fi

echo ""

# ============================================
# 5. Limpiar archivos temporales
# ============================================
echo "🧹 Limpiando archivos temporales..."
rm -rf buffalo_l
echo "✓ Carpeta buffalo_l/ eliminada"
echo ""

# ============================================
# 6. Verificar modelos instalados
# ============================================
echo "╔═══════════════════════════════════════════════════╗"
echo "║   MODELOS INSTALADOS                              ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

if [ -f "retinaface.onnx" ] && [ -f "arcface_r100.onnx" ]; then
    ls -lh retinaface.onnx arcface_r100.onnx
    echo ""
    echo "✅ INSTALACIÓN COMPLETA"
    echo ""
    echo "Modelos disponibles:"
    echo "  ✓ retinaface.onnx  - Detección de rostros"
    echo "  ✓ arcface_r100.onnx - Reconocimiento facial"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "SIGUIENTE PASO: Compilar el proyecto"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  cd .."
    echo "  ./build.sh"
    echo ""
    echo "Luego probar el detector:"
    echo "  ./run.sh test-img         # Probar con imagen"
    echo "  ./run.sh test-video       # Probar con video"
    echo ""
else
    echo "❌ INSTALACIÓN INCOMPLETA"
    echo ""
    echo "Archivos faltantes:"
    [ ! -f "retinaface.onnx" ] && echo "  ✗ retinaface.onnx"
    [ ! -f "arcface_r100.onnx" ] && echo "  ✗ arcface_r100.onnx"
    echo ""
    exit 1
fi