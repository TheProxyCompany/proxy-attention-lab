#!/usr/bin/env bash
set -euo pipefail

OUT=build/attn.metallib
SRC=kernels/paged_attention.metal

mkdir -p build
# 1. Compile MSL → AIR
xcrun -sdk macosx metal -c "$SRC" -o build/paged.air
# 2. Link AIR → METALLIB
xcrun -sdk macosx metallib build/paged.air -o "$OUT"

echo "✅  Built $OUT"
