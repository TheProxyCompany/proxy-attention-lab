#!/usr/bin/env bash
# Build and test the Proxy Attention Lab (PAL)

set -euo pipefail # Exit on error, unset var, pipe failure
trap 'echo "ERROR: Script failed at line $LINENO"' ERR

# --- Configuration ---
VENV_DIR=".venv" # Assumes virtualenv is in project root
BUILD_DIR="build" # CMake build directory (relative to project root)
CLEAN_BUILD=${CLEAN_BUILD:-false} # Set to true to force clean build: CLEAN_BUILD=true ./scripts/run.sh
PYTHON_EXE="${VENV_DIR}/bin/python"
UV_EXECUTABLE_PATH="$(which uv)" # Store the path to uv executable
PYTEST_EXE="${VENV_DIR}/bin/pytest"

# --- Helper Functions ---
log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

require_venv() {
    if [ ! -d "$VENV_DIR" ] || [ ! -f "$PYTHON_EXE" ]; then
        log "ERROR: Python virtual environment not found or not activated at $VENV_DIR"
        log "Please create and activate it first (e.g., python -m venv .venv && source .venv/bin/activate)"
        exit 1
    fi
    # Check if running within the activated venv (optional but good practice)
    if [ "${VIRTUAL_ENV:-}" != "$(pwd)/${VENV_DIR}" ]; then
         log "WARNING: Not running within the expected activated virtual environment ($VENV_DIR)."
         log "         Attempting to use executables directly from $VENV_DIR/bin/..."
    fi
    if ! command -v "$PYTEST_EXE" &>/dev/null; then
        log "ERROR: pytest not found in virtual environment. Please install dev dependencies (pip install -e '.[dev]')"
        exit 1
    fi
}

# --- Main Script Logic ---
log "Starting PAL Build & Test run..."
require_venv

# Optional Clean Build
if [ "$CLEAN_BUILD" = true ]; then
    log "Performing clean build: Removing $BUILD_DIR and dist info..."
    rm -rf "$BUILD_DIR"
    rm -rf src/python/*.egg-info # Remove previous build metadata if any
fi

# Build/Install the package in editable mode
# This invokes py-build-cmake and handles C++/Metal compilation
log "Building/Installing PAL in editable mode..."
"$UV_EXECUTABLE_PATH" pip install -e .

# Run Pytest
log "Running tests..."
"$PYTEST_EXE" tests/ "$@" # Pass any extra args to pytest

log "PAL Build & Test run finished successfully."
