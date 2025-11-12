#!/bin/bash

# Script de ejecucion rapida para PANTO

case "$1" in
    1|record)
        echo "=== Recording Main Stream ==="
        ./build/bin/record main ${2:-0}
        ;;
    2|record-sub)
        echo "=== Recording Sub Stream ==="
        ./build/bin/record sub ${2:-0}
        ;;
    3|view)
        echo "=== Viewing Streams ==="
        ./build/bin/view ${2:-both}
        ;;
    4|view-main)
        echo "=== Viewing Main Stream ==="
        ./build/bin/view main
        ;;
    5|view-sub)
        echo "=== Viewing Sub Stream ==="
        ./build/bin/view sub
        ;;
    6|record-view)
        echo "=== Recording + Viewing Main ==="
        ./build/bin/record_view main ${2:-0}
        ;;
    7|record-view-sub)
        echo "=== Recording + Viewing Sub ==="
        ./build/bin/record_view sub ${2:-0}
        ;;
    8|panto)
        echo "=== Running PANTO Main Application ==="
        ./build/bin/panto --config configs/config_1080p_roi.toml
        ;;
    9|panto-720p)
        echo "=== Running PANTO (720p) ==="
        ./build/bin/panto --config configs/config_720p.toml
        ;;
    10|panto-4k)
        echo "=== Running PANTO (4K) ==="
        ./build/bin/panto --config configs/config_4k.toml
        ;;
    *)
        echo "Usage: ./run.sh [option] [duration]"
        echo ""
        echo "Options:"
        echo "  1, record           - Record main stream"
        echo "  2, record-sub       - Record sub stream"
        echo "  3, view             - View both streams"
        echo "  4, view-main        - View main stream only"
        echo "  5, view-sub         - View sub stream only"
        echo "  6, record-view      - Record + view main"
        echo "  7, record-view-sub  - Record + view sub"
        echo "  8, panto            - Run PANTO (1080p ROI)"
        echo "  9, panto-720p       - Run PANTO (720p)"
        echo "  10, panto-4k        - Run PANTO (4K)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh 1          - Record main until Ctrl+C"
        echo "  ./run.sh 1 60       - Record main for 60 seconds"
        echo "  ./run.sh 3          - View both streams"
        echo "  ./run.sh 6 30       - Record+view main for 30s"
        echo ""
        exit 1
        ;;
esac