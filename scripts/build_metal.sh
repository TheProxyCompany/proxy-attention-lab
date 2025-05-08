#!/usr/bin/env bash
set -euo pipefail

# Define source, include, output directories and filenames
SRC_DIR="src/cpp"
SRC_METAL_FILE="${SRC_DIR}/paged_attention/paged_attention.metal"
INCLUDE_DIR="${SRC_DIR}" # Directory containing paged_attention.h
BUILD_DIR="build"       # Intermediate build files directory
LAB_DIR="src/python"    # Place the final .metallib next to Python code
OUT_METALLIB="${LAB_DIR}/pie_paged_attn_lab.metallib" # Final output library

# Temporary file for the intermediate AIR artifact
AIR_TMP=$(mktemp)

# Ensure required directories exist
mkdir -p "$BUILD_DIR"
mkdir -p "$LAB_DIR"

echo "Compiling $SRC_METAL_FILE to AIR..."
# Compile the .metal file to an AIR file (.air)
# - Add -I flag to specify the include directory for paged_attention.h
# - Use -o for the output AIR file (temporary)
xcrun -sdk macosx metal -I "$INCLUDE_DIR" -c "$SRC_METAL_FILE" -o "$AIR_TMP" || {
    echo "❌ Metal compilation to AIR failed"
    rm -f "$AIR_TMP" # Clean up temp file on failure
    exit 1
}
echo "✅ Compiled AIR artifact: $AIR_TMP"

echo "Linking AIR to $OUT_METALLIB..."
# Link the AIR file into the final Metal library (.metallib)
# - Use -o for the final output path
xcrun -sdk macosx metallib "$AIR_TMP" -o "$OUT_METALLIB" || {
    echo "❌ Metallib linking failed"
    rm -f "$AIR_TMP" # Clean up temp file on failure
    exit 1
}
echo "✅ Built Metal library: $OUT_METALLIB"

# Clean up the temporary AIR file
rm -f "$AIR_TMP"
echo "🗑️ Removed temporary AIR file."
