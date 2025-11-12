#!/bin/bash

echo "=== Cleaning PANTO build ==="

if [ -d "build" ]; then
    rm -rf build
    echo "Removed build directory"
fi

if [ -d "videos" ]; then
    echo "Found videos directory (not removed)"
    ls -lh videos/*.mp4 2>/dev/null | wc -l | xargs echo "  - MP4 files:"
fi

echo ""
echo "Clean complete. Run ./build.sh to rebuild."
echo ""