#!/usr/bin/env bash
# Script to generate and open the Xcode project for Proxy Attention Lab (PAL)

set -euo pipefail
trap 'echo "ERROR: Script failed at line $LINENO" >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
GENERATOR="Xcode"
CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=Release")

log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

hr() {
    printf "%80s\n" | tr ' ' '-'
}

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [--clean] [--cmake-arg <arg>]...

Options:
  --clean             Remove the build directory before generating the Xcode project.
  --cmake-arg <arg>   Pass additional arguments to CMake (can be repeated).
  --help              Show this help message.

Example:
  $(basename "$0") --clean --cmake-arg -DPAL_ENABLE_DEBUG_LOGGING=ON
EOF
}

# Parse arguments
CLEAN_BUILD=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --cmake-arg)
            if [[ $# -lt 2 ]]; then
                log "ERROR: --cmake-arg requires an argument"
                exit 1
            fi
            CMAKE_ARGS+=("$2")
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log "ERROR: Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

hr
log "PAL Xcode Project Generator"
hr

if [ "${CLEAN_BUILD}" = true ]; then
    log "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

log "Configuring project with CMake (Generator: ${GENERATOR})"
cmake -S . -B "${BUILD_DIR}" -G "${GENERATOR}" "${CMAKE_ARGS[@]}"

XCODEPROJ_PATH="${BUILD_DIR}/proxy_attention_lab.xcodeproj"

if [ ! -d "${XCODEPROJ_PATH}" ]; then
    log "ERROR: Xcode project not found at ${XCODEPROJ_PATH}"
    exit 1
fi

log "Opening Xcode project: ${XCODEPROJ_PATH}"
open "${XCODEPROJ_PATH}"
