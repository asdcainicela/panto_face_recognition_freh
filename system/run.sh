#!/bin/bash
# run.sh - Script de ejecución rápida para PANTO

case "$1" in
    # ============================================
    # CAPTURA DE VIDEO
    # ============================================
    1|record)
        echo "=== Grabando Main Stream ==="
        ./build/bin/record main ${2:-0}
        ;;
    2|record-sub)
        echo "=== Grabando Sub Stream ==="
        ./build/bin/record sub ${2:-0}
        ;;
    3|view-both)
        echo "=== Visualizando Ambos Streams ==="
        ./build/bin/view both
        ;;
    4|view-main)
        echo "=== Visualizando Main Stream ==="
        ./build/bin/view main
        ;;
    5|view-sub)
        echo "=== Visualizando Sub Stream ==="
        ./build/bin/view sub
        ;;
    6|record-view)
        echo "=== Grabando + Visualizando Main ==="
        ./build/bin/record_view main ${2:-0}
        ;;
    7|record-view-sub)
        echo "=== Grabando + Visualizando Sub ==="
        ./build/bin/record_view sub ${2:-0}
        ;;
    
    # ============================================
    # DETECCIÓN DE ROSTROS
    # ============================================
    test-img)
        echo "=== Test Detector: Imagen ==="
        IMAGE=${2:-test/img/test1.png}
        THRESHOLD=${3:-0.5}
        
        if [ ! -f "$IMAGE" ]; then
            echo "Error: Imagen no encontrada: $IMAGE"
            echo ""
            echo "Uso: ./run.sh test-img [ruta/imagen.jpg] [threshold]"
            echo "Ejemplo: ./run.sh test-img test/img/test1.png 0.6"
            exit 1
        fi
        
        echo "Imagen: $IMAGE"
        echo "Threshold: $THRESHOLD"
        echo ""
        
        # Compilar si es necesario
        if [ ! -f "build/bin/test_detector" ]; then
            echo "Compilando..."
            ./build.sh
        fi
        
        ./build/bin/test_detector models/retinaface.onnx "$IMAGE"
        ;;
    
    test-video)
        echo "=== Test Detector: Video ==="
        
        # Si no se especifica video, usar el más reciente
        if [ -z "$2" ]; then
            VIDEO=$(ls -t videos/recording_*.mp4 2>/dev/null | head -1)
            if [ -z "$VIDEO" ]; then
                echo "Error: No se encontraron videos en videos/"
                echo ""
                echo "Uso: ./run.sh test-video [ruta/video.mp4] [threshold]"
                echo "Ejemplo: ./run.sh test-video videos/mi_video.mp4 0.6"
                exit 1
            fi
            echo "Usando video más reciente: $VIDEO"
        else
            VIDEO=$2
            if [ ! -f "$VIDEO" ]; then
                echo "Error: Video no encontrado: $VIDEO"
                exit 1
            fi
        fi
        
        THRESHOLD=${3:-0.5}
        
        echo "Video: $VIDEO"
        echo "Threshold: $THRESHOLD"
        echo ""
        echo "Controles:"
        echo "  SPACE - pausar/reanudar"
        echo "  ESC   - salir"
        echo "  +/-   - ajustar threshold"
        echo "  S     - guardar frame"
        echo "  F     - modo rápido"
        echo ""
        
        # Compilar si es necesario
        if [ ! -f "build/bin/test_detector_video" ]; then
            echo "Compilando..."
            ./build.sh
        fi
        
        ./build/bin/test_detector_video models/retinaface.onnx "$VIDEO" $THRESHOLD
        ;;
    
    test-webcam)
        echo "=== Test Detector: Webcam ==="
        echo "Presiona ESC para salir"
        echo ""
        
        if [ ! -f "build/bin/test_detector" ]; then
            echo "Compilando..."
            ./build.sh
        fi
        
        ./build/bin/test_detector models/retinaface.onnx
        ;;
    
    diagnose)
        echo "=== Diagnóstico de Modelo ==="
        IMAGE=${2:-test/img/test1.png}
        
        if [ ! -f "$IMAGE" ]; then
            echo "Imagen no encontrada: $IMAGE"
            exit 1
        fi
        
        if [ ! -f "build/bin/diagnose_retinaface" ]; then
            echo "Compilando..."
            ./build.sh
        fi
        
        ./build/bin/diagnose_retinaface models/retinaface.onnx "$IMAGE"
        ;;
    
    # ============================================
    # APLICACIÓN PRINCIPAL
    # ============================================
    8|panto)
        echo "=== PANTO (1080p ROI) ==="
        ./build/bin/panto --config configs/config_1080p_roi.toml
        ;;
    9|panto-720p)
        echo "=== PANTO (720p) ==="
        ./build/bin/panto --config configs/config_720p.toml
        ;;
    10|panto-4k)
        echo "=== PANTO (4K) ==="
        ./build/bin/panto --config configs/config_4k.toml
        ;;
    11|panto-full)
        echo "=== PANTO (1080p Full) ==="
        ./build/bin/panto --config configs/config_1080p_full.toml
        ;;
    
    # ============================================
    # UTILIDADES
    # ============================================
    build)
        echo "=== Compilando PANTO ==="
        ./build.sh
        ;;
    
    clean)
        echo "=== Limpiando Build ==="
        ./clean.sh
        ;;
    
    rebuild)
        echo "=== Limpieza y Recompilación ==="
        ./clean.sh
        ./build.sh
        ;;
    
    models)
        echo "=== Descargando Modelos ==="
        cd models
        ./setup_models.sh
        cd ..
        ;;
    
    help|--help|-h|"")
        cat << 'EOF'
╔════════════════════════════════════════════════════════════╗
║                   PANTO - Menú de Comandos                 ║
╚════════════════════════════════════════════════════════════╝

📹 CAPTURA DE VIDEO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  record            - Grabar main stream
  record-sub        - Grabar sub stream  
  view-main         - Ver main stream
  view-both         - Ver ambos streams
  record-view       - Grabar + ver main

  Ejemplos:
    ./run.sh record              # Hasta Ctrl+C
    ./run.sh record 60           # 60 segundos
    ./run.sh view-main

🎯 DETECCIÓN DE ROSTROS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  test-img          - Probar con imagen
  test-video        - Probar con video MP4
  test-webcam       - Probar con webcam
  diagnose          - Diagnóstico de modelo

  Ejemplos:
    ./run.sh test-img                          # Usa test/img/test1.png
    ./run.sh test-img foto.jpg                 # Imagen personalizada
    ./run.sh test-img foto.jpg 0.7             # Con threshold 0.7
    
    ./run.sh test-video                        # Video más reciente
    ./run.sh test-video videos/video.mp4       # Video específico
    ./run.sh test-video videos/video.mp4 0.6   # Con threshold 0.6

🚀 APLICACIÓN PRINCIPAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  panto             - Ejecutar PANTO (1080p ROI) - Recomendado
  panto-720p        - PANTO en 720p
  panto-4k          - PANTO en 4K
  panto-full        - PANTO 1080p sin ROI

🔧 UTILIDADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  build             - Compilar proyecto
  clean             - Limpiar build
  rebuild           - Limpiar y recompilar
  models            - Descargar modelos ONNX
  help              - Mostrar esta ayuda

📖 PRIMEROS PASOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Descargar modelos:  ./run.sh models
  2. Compilar:           ./run.sh build
  3. Probar detector:    ./run.sh test-img

⚙️  AJUSTAR THRESHOLD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  El threshold controla sensibilidad (0.0-1.0):
  
  • 0.7-0.8  → Muy estricto (pocos falsos positivos)
  • 0.5-0.6  → Balanceado (recomendado)
  • 0.3-0.4  → Permisivo (más detecciones)

  Durante video: Usa +/- para ajustar en tiempo real

📚 MÁS INFO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  README.md          - Documentación completa
  arquitecture.md    - Arquitectura del sistema

EOF
        ;;
    
    *)
        echo "Comando desconocido: $1"
        echo ""
        echo "Usa './run.sh help' para ver comandos disponibles"
        exit 1
        ;;
esac