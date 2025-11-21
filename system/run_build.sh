#!/bin/bash
set -e

cores=6
build_type="${1:-Release}"

mkdir -p build
cd build

[ ! -f Makefile ] && cmake -DCMAKE_BUILD_TYPE=$build_type ..

make -j$cores

missing=0
[ ! -f bin/panto ] && missing=1
[ ! -f bin/benchmark ] && missing=1

[ $missing -eq 1 ] && echo "build incompleto" && exit 1

echo "ok"
