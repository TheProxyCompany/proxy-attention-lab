#!/usr/bin/env bash
#
# Script to build and run all benchmarks for the project.
# - Updates dependencies and rebuilds the project.
# - Discovers and runs Python pytest-benchmark tests.
# - Discovers and runs C++ Google Benchmark executables.

set -euo pipefail # Exit on error, unset var, pipe failure
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

# --- Configuration ---
VENV_DIR=".venv"
BUILD_DIR="build" # CMake build directory
UV_EXECUTABLE_PATH=""             # Will be detected

# Directories for benchmark discovery
PYTHON_BENCHMARK_ROOT_DIR="tests" # Root for Python benchmark discovery
CPP_BENCHMARK_BUILD_ROOT_DIR="build/tests" # Root for C++ benchmark executable discovery

# Naming conventions for discovery
PYTHON_BENCHMARK_PATTERN="test_*performance*.py" # Pattern for Python benchmark files

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
    hr
    log "Starting Benchmark Suite Runner"
    hr

    detect_uv
    activate_venv

    # 1. Update Dependencies and Rebuild Project
    log "Updating dependencies and rebuilding project..."
    "${UV_EXECUTABLE_PATH}" pip install --upgrade --no-deps "git+https://github.com/TheProxyCompany/mlx.git" "nanobind>=2.5.0"

    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
        log "Using CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"
    else
        log "CMAKE_BUILD_PARALLEL_LEVEL not set, CMake will use its default parallelism."
    fi
    "${UV_EXECUTABLE_PATH}" pip install . --force-reinstall --no-build-isolation --no-cache-dir
    log "Project rebuilt successfully."

    # Ensure the main build/output directory for benchmarks exists
    log "Ensuring benchmark output directory exists: ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # Explicitly configure and build C++ benchmarks
    hr
    log "Configuring C++ benchmarks separately..."
    # Use the CMAKE_BUILD_PARALLEL_LEVEL if set, otherwise let CMake decide or use a default
    local cmake_parallel_args=""
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
        cmake_parallel_args="-j${CMAKE_BUILD_PARALLEL_LEVEL}"
    elif command -v nproc &>/dev/null; then
        cmake_parallel_args="-j$(nproc)"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cmake_parallel_args="-j$(sysctl -n hw.ncpu)"
    else
        cmake_parallel_args="-j2" # Default to 2 jobs if nproc/sysctl not available
    fi

    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release # Configure all targets from root
    log "C++ benchmark CMake configuration complete."

    log "Building C++ benchmarks..."
    cmake --build "${BUILD_DIR}" ${cmake_parallel_args} # Build all targets, including benchmarks
    # Alternatively, to build only the specific benchmark target if known and desired:
    # cmake --build "${BUILD_DIR}" --target pal_op_benchmarks ${cmake_parallel_args}
    log "C++ benchmarks built."
    hr

    # 2. Discover and Run Python Pytest Benchmarks
    hr
    log "Discovering and Running Python pytest-benchmark tests..."

    # Use find to locate Python benchmark files
    # Adjust depth or path if benchmark files are nested differently
    python_benchmark_files=()
    while IFS= read -r -d $'\0' file; do
        python_benchmark_files+=("$file")
    done < <(find "${PYTHON_BENCHMARK_ROOT_DIR}" -type f -name "${PYTHON_BENCHMARK_PATTERN}" -print0)

    # if [ ${#python_benchmark_files[@]} -eq 0 ]; then
    #     log "No Python benchmark files found matching pattern '${PYTHON_BENCHMARK_PATTERN}' in '${PYTHON_BENCHMARK_ROOT_DIR}'."
    # else
    #     for benchmark_file in "${python_benchmark_files[@]}"; do
    #         hr
    #         log "Running Python benchmarks in: ${benchmark_file}"
    #         # Generate a unique JSON output name based on the benchmark file name
    #         local benchmark_basename
    #         benchmark_basename=$(basename "${benchmark_file}" .py)
    #         local python_json_output="${BUILD_DIR}/${benchmark_basename}_results.json"

    #         pytest "${benchmark_file}" \
    #             --benchmark-only \
    #             --benchmark-columns="min,max,mean,stddev,rounds,iterations" \
    #             --benchmark-json="${python_json_output}" \
    #             -v
    #         log "Python benchmarks from ${benchmark_file} completed. Results potentially in ${python_json_output}"
    #     done
    # fi
    hr

    # 3. Discover and Run C++ Google Benchmarks
    log "Discovering and Running C++ Google Benchmark tests..."

    # Use find to locate C++ benchmark executables in the build directory
    cpp_benchmark_executables=()
    while IFS= read -r -d $'\0' file; do
        cpp_benchmark_executables+=("$file")
    done < <(find "${CPP_BENCHMARK_BUILD_ROOT_DIR}" -type f -perm -u+x -print0)

    if [ ${#cpp_benchmark_executables[@]} -eq 0 ]; then
        log "No C++ benchmark executables found in '${CPP_BENCHMARK_BUILD_ROOT_DIR}'."
        log "Ensure project was built correctly and executables are in the expected locations."
    else
        for benchmark_exe in "${cpp_benchmark_executables[@]}"; do
            hr
            log "Running C++ benchmarks: ${benchmark_exe}"
            local benchmark_exename
            benchmark_exename=$(basename "${benchmark_exe}")
            local cpp_json_output="${BUILD_DIR}/${benchmark_exename}_results.json"

            SPDLOG_LEVEL=warn "${benchmark_exe}" \
                --benchmark_format=json \
                --benchmark_out="${cpp_json_output}" \
                --benchmark_repetitions=3 # Default repetitions, can be overridden
                # Add other common Google Benchmark flags if desired
            log "C++ benchmarks from ${benchmark_exe} completed. Results saved to ${cpp_json_output}"
        done
    fi
    hr

    log "Benchmark Suite Finished."
    log "Review individual JSON output files in ${BUILD_DIR}/ for detailed results."
}

# --- Script Execution ---
main "$@"

exit 0
