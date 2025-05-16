#!/usr/bin/env bash
#
# Script to build and run all benchmarks for Proxy Attention Lab (PAL)
# - Updates dependencies and rebuilds PAL.
# - Runs Python pytest-benchmark tests.
# - Runs C++ Google Benchmark tests.

set -euo pipefail # Exit on error, unset var, pipe failure
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)" # Assumes script is in a 'scripts' subdir or similar
VENV_DIR="${PROJECT_ROOT}/.venv"
BUILD_DIR="${PROJECT_ROOT}/build" # CMake build directory
UV_EXECUTABLE_PATH=""             # Will be detected
PYTEST_BENCHMARK_FILE="${PROJECT_ROOT}/tests/paged_attention/benchmarks/python/test_pal_performance_pytest.py"
CPP_BENCHMARK_EXECUTABLE="${BUILD_DIR}/tests/paged_attention/benchmarks/cpp/pal_op_benchmarks"
CPP_BENCHMARK_OUTPUT_JSON="${BUILD_DIR}/pal_cpp_benchmark_results.json" # For C++ results

# --- Helper Functions ---
log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

hr() {
    printf "%80s\n" | tr ' ' '-'
}

detect_uv() {
    if command -v uv &>/dev/null; then
        UV_EXECUTABLE_PATH="$(command -v uv)"
        log "Using system uv: ${UV_EXECUTABLE_PATH}"
    elif [ -x "${VENV_DIR}/bin/uv" ]; then
        UV_EXECUTABLE_PATH="${VENV_DIR}/bin/uv"
        log "Using venv uv: ${UV_EXECUTABLE_PATH}"
    else
        log "ERROR: uv executable not found in PATH or venv. Please install uv." >&2
        exit 1
    fi
}

activate_venv() {
    if [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV}" != "${VENV_DIR}" ]; then
        if [ -f "${VENV_DIR}/bin/activate" ]; then
            log "Activating Python virtual environment: ${VENV_DIR}"
            # shellcheck source=/dev/null
            source "${VENV_DIR}/bin/activate"
        else
            log "ERROR: Virtual environment activation script not found at ${VENV_DIR}/bin/activate." >&2
            log "Please create and activate the venv first."
            exit 1
        fi
    else
        log "Virtual environment already active: ${VIRTUAL_ENV}"
    fi
}

# --- Main Script Logic ---
main() {
    cd "${PROJECT_ROOT}" # Ensure we are in the project root

    hr
    log "Starting PAL Benchmark Suite"
    hr

    detect_uv
    activate_venv

    # 1. Update Dependencies and Rebuild PAL
    #    (This ensures we're benchmarking the latest code with correct dependencies)
    log "Updating dependencies and rebuilding PAL..."
    # Assuming MLX and Nanobind might have updates, though --no-deps is used for PAL itself later
    "${UV_EXECUTABLE_PATH}" pip install --upgrade --no-deps "git+https://github.com/TheProxyCompany/mlx.git" "nanobind>=2.5.0" # Use >= for nanobind

    # Rebuild PAL C++ extension and Python package
    # The --force-reinstall and --no-cache-dir ensure a clean build of the extension
    # CMAKE_BUILD_PARALLEL_LEVEL can be set as an env var if desired
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
        log "Using CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"
    else
        log "CMAKE_BUILD_PARALLEL_LEVEL not set, CMake will use its default parallelism."
    fi
    "${UV_EXECUTABLE_PATH}" pip install . --force-reinstall --no-build-isolation --no-cache-dir
    log "PAL rebuilt successfully."

    # 2. Run Python Pytest Benchmarks
    hr
    log "Running Python pytest-benchmark tests from: ${PYTEST_BENCHMARK_FILE}"
    if [ ! -f "${PYTEST_BENCHMARK_FILE}" ]; then
        log "ERROR: Python benchmark file not found: ${PYTEST_BENCHMARK_FILE}" >&2
        exit 1
    fi
    # Command to run pytest benchmarks and save results
    # Adjust columns or output format as needed
    PYTEST_BENCHMARK_JSON_OUTPUT="${BUILD_DIR}/pal_python_benchmark_results.json"
    pytest "${PYTEST_BENCHMARK_FILE}" \
        --benchmark-only \
        --benchmark-columns="min,max,mean,stddev,rounds,iterations" \
        --benchmark-json="${PYTEST_BENCHMARK_JSON_OUTPUT}" \
        -v
    log "Python pytest-benchmark tests completed. Results potentially in ${PYTEST_BENCHMARK_JSON_OUTPUT}"
    hr

    # 3. Run C++ Google Benchmarks
    log "Running C++ Google Benchmark tests: ${CPP_BENCHMARK_EXECUTABLE}"
    if [ ! -x "${CPP_BENCHMARK_EXECUTABLE}" ]; then
        log "ERROR: C++ benchmark executable not found or not executable: ${CPP_BENCHMARK_EXECUTABLE}" >&2
        log "Ensure PAL was built correctly (e.g., run 'cmake --build ${BUILD_DIR} --target ${CPP_BENCHMARK_EXECUTABLE##*/}' or full build)."
        exit 1
    fi
    # Run C++ benchmarks and output to JSON
    "${CPP_BENCHMARK_EXECUTABLE}" \
        --benchmark_format=json \
        --benchmark_out="${CPP_BENCHMARK_OUTPUT_JSON}" \
        --benchmark_repetitions=3 # Example: add repetitions
        # Add other Google Benchmark flags as needed, e.g., --benchmark_filter=BM_PAL_LatencyVsSeqLen

    log "C++ Google Benchmark tests completed. Results saved to ${CPP_BENCHMARK_OUTPUT_JSON}"
    hr

    log "PAL Benchmark Suite Finished."
    log "Python results are in terminal output and potentially: ${PYTEST_BENCHMARK_JSON_OUTPUT}"
    log "C++ results are in: ${CPP_BENCHMARK_OUTPUT_JSON}"
}

# --- Script Execution ---
main "$@"

exit 0
