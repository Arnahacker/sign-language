#!/bin/bash
set -e

# ── Usage ──────────────────────────────────────────────────────────────────────
# ./build.sh /path/to/libtorch
# If no path given, assumes ./libtorch in the project root

LIBTORCH_PATH=${1:-"$(pwd)/libtorch"}

if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "LibTorch not found at: $LIBTORCH_PATH"
    echo "Download from: https://pytorch.org/get-started/locally/"
    echo "Usage: ./build.sh /path/to/libtorch"
    exit 1
fi

echo "Building with LibTorch at: $LIBTORCH_PATH"

mkdir -p build
cd build

cmake ../cpp \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j$(nproc)

# Copy .so to project root so Python can import it
cp sign_inference*.so ../
echo ""
echo "Build complete. sign_inference module is ready."
