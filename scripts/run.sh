#!/usr/bin/env bash
# Build and test the Proxy Attention Lab (PAL)

set -euo pipefail # Exit on error, unset var, pipe failure
trap 'echo "ERROR: Script failed at line $LINENO"' ERR

# --- Configuration ---
VENV_DIR=".venv" # Assumes virtualenv is in project root
BUILD_DIR="build" # CMake build directory (relative to project root)
CLEAN_BUILD=${CLEAN_BUILD:-true} # Set to true to force clean build: CLEAN_BUILD=true ./run.sh
UV_EXECUTABLE_PATH="$(which uv)" # Store the path to uv executable
PYTEST_EXE="${VENV_DIR}/bin/pytest"
DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

export DEVELOPER_DIR

# --- Helper Functions ---
log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

require_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "ERROR: Python virtual environment not found or not activated at $VENV_DIR"
        log "Please create and activate it first (e.g., python -m venv .venv && source .venv/bin/activate)"
        exit 1
    fi
    # Check if running within the activated venv (optional but good practice)
    if [ "${VIRTUAL_ENV:-}" != "$(pwd)/${VENV_DIR}" ]; then
         log "WARNING: Not running within the expected activated virtual environment ($VENV_DIR)."
         log "         Attempting to use executables directly from $VENV_DIR/bin/..."
    fi
}

# --- Main Script Logic ---
log "Starting PAL Build & Test run..."
require_venv

# Optional Clean Build
if [ "$CLEAN_BUILD" = true ]; then
    log "Performing clean build: Removing $BUILD_DIR and dist info..."
    rm -rf "$BUILD_DIR"
    rm -rf ".py-build-cmake_cache"
    rm -rf src/proxy_attention_lab/*.egg-info
fi

log "Installing/Updating PAL dependencies (including MLX, Nanobind from pyproject.toml)..."
export CMAKE_BUILD_PARALLEL_LEVEL=8 # Set parallel build level to 8
"$UV_EXECUTABLE_PATH" pip install --no-deps "git+https://github.com/TheProxyCompany/mlx.git" "nanobind==2.5.0"

"$UV_EXECUTABLE_PATH" pip install "." --force-reinstall --no-build-isolation --no-cache-dir

# Run Pytest
log "Running tests..."
"$PYTEST_EXE" tests/ "$@"

log "PAL Build & Test run finished successfully."
