#!/usr/bin/env bash
#
# Script to build and run benchmarks for the project with flexible control.
# - Can update dependencies and rebuild the project.
# - Can discover and run Python pytest-benchmark tests (all or filtered).
# - Can discover and run C++ Google Benchmark executables (all or filtered).
# - Can analyze benchmark results from previous runs.

set -euo pipefail
# Trap any error and print a useful message
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

# Minimum required Python version
REQUIRED_PYTHON="3.11"

# --- Configuration ---
VENV_DIR=".venv"
BUILD_DIR="build" # CMake build directory
UV_EXECUTABLE_PATH=""  # Will be detected

# Directories for benchmark discovery
BENCHMARK_ROOT_DIR="benchmarks" # Root for benchmark discovery
BENCHMARK_OUTPUT_ROOT=".benchmarks" # Output directory for benchmark results

# Naming conventions for discovery (unused but kept for reference)
PYTHON_BENCHMARK_PATTERN="*/benchmarks/python/*.py"

# --- Helper Functions ---
log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

hr() {
    printf "%80s\n" | tr ' ' '-'
}

# Verify that a required command exists in PATH
check_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log "ERROR: Required command '$cmd' not found in PATH" >&2
        exit 1
    fi
}

# Ensure the active python meets the required version
check_python_version() {
    local pyver
    pyver=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '')
    if [[ -z "$pyver" ]]; then
        log "ERROR: python executable not found" >&2
        exit 1
    fi
    if [[ $(printf '%s\n' "$REQUIRED_PYTHON" "$pyver" | sort -V | head -n1) != "$REQUIRED_PYTHON" ]]; then
        log "ERROR: Python $REQUIRED_PYTHON or higher required, found $pyver" >&2
        exit 1
    fi
}

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --run [all|py|cpp] [kernel]  Run benchmarks. Language defaults to 'all'.
                               If a kernel name is provided it limits the run
                               to that kernel.
  --analyze                    Only analyze existing benchmark results
  --rebuild-only               Only update dependencies and rebuild the project
  --reset                      Clear all existing benchmark results before running
  --help                       Show this help message

Combined Options:
  --run ... --analyze          Run the specified benchmarks and then analyze the results
  --run ... --reset            Clear all benchmarks before running new ones

Examples:
  $(basename "$0") --run                  # Rebuild, run all benchmarks
  $(basename "$0") --run py paged_attention   # Run Python benchmarks for paged_attention
  $(basename "$0") --run cpp                 # Run all C++ benchmarks
  $(basename "$0") --analyze                # Only analyze existing data
EOF
}

# --- Core Action Functions ---

setup_environment() {
    log "Setting up environment..."
    detect_uv
    activate_venv
    # Verify required tools are available
    check_command pytest
    check_command cmake
    check_python_version
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

update_and_rebuild_project() {
    local cmake_parallel_args=""

    log "Updating dependencies and rebuilding project..."

    check_command cmake

    # Update dependencies
    "${UV_EXECUTABLE_PATH}" pip install --upgrade --no-deps "git+https://github.com/TheProxyCompany/mlx.git" "nanobind>=2.5.0"

    # Configure build parallelism
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
        log "Using CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"
    else
        log "CMAKE_BUILD_PARALLEL_LEVEL not set, CMake will use its default parallelism."
    fi

    # Install the project
    "${UV_EXECUTABLE_PATH}" pip install . --force-reinstall --no-build-isolation --no-cache-dir
    log "Project rebuilt successfully."

    # Ensure the main build directory exists
    log "Ensuring build directory exists: ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # Explicitly configure and build C++ benchmarks
    hr
    log "Configuring C++ benchmarks separately..."

    # Determine build parallelism
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
        cmake_parallel_args="-j${CMAKE_BUILD_PARALLEL_LEVEL}"
    elif command -v nproc &>/dev/null; then
        cmake_parallel_args="-j$(nproc)"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cmake_parallel_args="-j$(sysctl -n hw.ncpu)"
    else
        cmake_parallel_args="-j2" # Default to 2 jobs if nproc/sysctl not available
    fi

    # Configure and build
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    log "C++ benchmark CMake configuration complete."

    log "Building C++ benchmarks..."
    cmake --build "${BUILD_DIR}" ${cmake_parallel_args}
    log "C++ benchmarks built."
    hr
}

setup_benchmark_output_dir() {
    local reset="${1:-false}"

    log "Setting up benchmark output directory: ${BENCHMARK_OUTPUT_ROOT}"

    if [ "${reset}" = "true" ]; then
        log "Hard reset requested. Clearing all benchmark results."
        rm -rf "${BENCHMARK_OUTPUT_ROOT}"
    fi

    mkdir -p "${BENCHMARK_OUTPUT_ROOT}"
}

run_python_benchmarks() {
    local kernel="${1:-}"
    local benchmark_start benchmark_end benchmark_duration

    hr
    if [ -n "${kernel}" ]; then
        log "Running Python benchmarks for kernel: ${kernel}"
    else
        log "Running all Python benchmarks"
    fi

    benchmark_start=$(date +%s)

    python_benchmark_files=()
    if [ -n "${kernel}" ]; then
        if [ -d "${BENCHMARK_ROOT_DIR}/${kernel}/python" ]; then
            while IFS= read -r -d $'\0' file; do
                python_benchmark_files+=("$file")
            done < <(find "${BENCHMARK_ROOT_DIR}/${kernel}/python" -maxdepth 1 -type f -name "*.py" -print0 | sort -z -u)
        else
            log "WARNING: No Python benchmarks found for kernel '${kernel}'"
        fi
    else
        while IFS= read -r -d $'\0' file; do
            python_benchmark_files+=("$file")
        done < <(find "${BENCHMARK_ROOT_DIR}" -type f -path "*/python/*.py" -print0 | sort -z -u)
    fi

    if [ ${#python_benchmark_files[@]} -eq 0 ]; then
        log "No Python benchmark files found in any '*/python' subfolder under '${BENCHMARK_ROOT_DIR}'."
    else
        for benchmark_file in "${python_benchmark_files[@]}"; do
            hr
            log "Running Python benchmarks in: ${benchmark_file}"

            # Generate a unique JSON output name based on the benchmark file name and timestamp
            local benchmark_basename
            benchmark_basename=$(basename "${benchmark_file}" .py)
            local timestamp=$(date +"%Y%m%d_%H%M%S")
            local test_type="py_all"
            if [ -n "${kernel}" ]; then
                test_type="py_${kernel}"
            fi
            local python_json_output="${BENCHMARK_OUTPUT_ROOT}/${test_type}_${benchmark_basename}_${timestamp}.json"

            # Run pytest with appropriate filters
            pytest "${benchmark_file}" \
                --benchmark-only \
                --benchmark-columns="min,max,mean,rounds,iterations" \
                --benchmark-json="${python_json_output}" \
                --benchmark-min-time=0.001 \
                --benchmark-warmup=on \
                --benchmark-warmup-iterations=10 \
                -v

            log "Python benchmarks from ${benchmark_file} completed. Results saved to ${python_json_output}"
        done
    fi

    benchmark_end=$(date +%s)
    benchmark_duration=$((benchmark_end - benchmark_start))
    log "Python benchmarks completed in ${benchmark_duration}s"
    hr
}

run_cpp_benchmarks() {
    local kernel="${1:-}"
    local filter_option=""
    local benchmark_start benchmark_end benchmark_duration

    hr
    if [ -n "${kernel}" ]; then
        log "Running C++ benchmarks for kernel: ${kernel}"
        filter_option="--benchmark_filter=${kernel}"
    else
        log "Running all C++ benchmarks"
    fi

    benchmark_start=$(date +%s)

    cpp_benchmark_executables=()
    if [ -n "${kernel}" ]; then
        while IFS= read -r -d $'\0' file; do
            cpp_benchmark_executables+=("$file")
        done < <(find "${BUILD_DIR}/${BENCHMARK_ROOT_DIR}/${kernel}/cpp" -type f -perm -u+x -print0 | sort -z -u)
    else
        while IFS= read -r -d $'\0' file; do
            cpp_benchmark_executables+=("$file")
        done < <(find "${BUILD_DIR}/${BENCHMARK_ROOT_DIR}" -type f -path "*/cpp/*" -perm -u+x -print0 | sort -z -u)
    fi

    if [ ${#cpp_benchmark_executables[@]} -eq 0 ]; then
        log "No C++ benchmark executables found in '${BUILD_DIR}/${BENCHMARK_ROOT_DIR}'."
        log "Ensure project was built correctly and executables are in the expected locations."
    else
        for benchmark_exe in "${cpp_benchmark_executables[@]}"; do
            hr
            log "Running C++ benchmarks: ${benchmark_exe}"

            local benchmark_exename
            benchmark_exename=$(basename "${benchmark_exe}")
            local timestamp=$(date +"%Y%m%d_%H%M%S")
            local test_type="cpp_all"
            if [ -n "${kernel}" ]; then
                test_type="cpp_${kernel}"
            fi
            local cpp_json_output="${BENCHMARK_OUTPUT_ROOT}/${test_type}_${benchmark_exename}_${timestamp}.json"

            SPDLOG_LEVEL=debug "${benchmark_exe}" \
                --benchmark_format=json \
                --benchmark_out="${cpp_json_output}" \
                --benchmark_repetitions=1 \
                ${filter_option}

            log "C++ benchmarks from ${benchmark_exe} completed. Results saved to ${cpp_json_output}"
        done
    fi

    benchmark_end=$(date +%s)
    benchmark_duration=$((benchmark_end - benchmark_start))
    log "C++ benchmarks completed in ${benchmark_duration}s"
    hr
}

analyze_results() {
    local kernel="${1:-}"
    local analyzer_args=""

    hr
    log "Analyzing benchmark results..."

    # Check if there are benchmark results to analyze
    if [ ! -d "${BENCHMARK_OUTPUT_ROOT}" ] || [ -z "$(ls -A "${BENCHMARK_OUTPUT_ROOT}" 2>/dev/null)" ]; then
        log "WARNING: No benchmark results found in ${BENCHMARK_OUTPUT_ROOT}"
        log "Run benchmarks first or ensure benchmark JSON files exist"
        return 1
    fi

    # Add kernel filter if provided
    if [ -n "${kernel}" ]; then
        log "Filtering analysis for kernel: ${kernel}"
        analyzer_args="--kernel ${kernel}"
    fi

    # Run analysis script
    log "Running analysis script on results in ${BENCHMARK_OUTPUT_ROOT}..."
    python -m benchmarks.analyzer "${BENCHMARK_OUTPUT_ROOT}" "${BENCHMARK_OUTPUT_ROOT}" ${analyzer_args}

    log "Analysis complete. Results saved to ${BENCHMARK_OUTPUT_ROOT}"
    hr
}

# --- Main Script Logic ---
main() {
    local script_start=$(date +%s)
    local script_end total_duration
    local RUN_REQUESTED=false
    local ANALYZE_ONLY=false
    local REBUILD_ONLY=false
    local RESET_BENCHMARKS=true
    local RUN_LANGUAGE="all"
    local RUN_KERNEL=""

    hr
    log "Starting Benchmark Suite Runner"
    hr

    # Process command-line arguments
    # First check for help flag as it should take precedence
    for arg in "$@"; do
        if [ "$arg" == "--help" ]; then
            print_usage
            exit 0
        fi
    done

    if [ $# -eq 0 ]; then
        # Default behavior: run everything
        setup_environment
        setup_benchmark_output_dir "true"
        update_and_rebuild_project
        run_python_benchmarks ""
        run_cpp_benchmarks ""
        analyze_results ""
    else
        # Process flags
        local RUN_AND_ANALYZE=false

        while [ $# -gt 0 ]; do
            case "$1" in
                --run)
                    RUN_REQUESTED=true
                    if [ $# -gt 1 ] && [[ ! "$2" == --* ]]; then
                        RUN_LANGUAGE="$2"
                        shift
                        if [ $# -gt 1 ] && [[ ! "$2" == --* ]]; then
                            RUN_KERNEL="$2"
                            shift
                        fi
                    else
                        RUN_LANGUAGE="all"
                    fi
                    ;;
                --analyze)
                    if [ "${RUN_REQUESTED}" = true ]; then
                        RUN_AND_ANALYZE=true
                    else
                        ANALYZE_ONLY=true
                    fi
                    ;;
                --rebuild-only)
                    REBUILD_ONLY=true
                    ;;
                --reset)
                    RESET_BENCHMARKS=true
                    ;;
                --help)
                    print_usage
                    exit 0
                    ;;
                *)
                    log "ERROR: Unknown option: $1"
                    print_usage
                    exit 1
                    ;;
            esac
            shift
        done

        # Validate run language
        if [ "${RUN_REQUESTED}" = true ] && ! [[ "${RUN_LANGUAGE}" =~ ^(all|py|cpp)$ ]]; then
            log "ERROR: Invalid run language: ${RUN_LANGUAGE}"
            log "Valid values: all, py, cpp"
            exit 1
        fi

        # Setup environment for all operations
        setup_environment

        # Conditional execution based on flags
        if [ "${REBUILD_ONLY}" = true ]; then
            log "Rebuild-only mode"
            update_and_rebuild_project
            exit 0
        fi

        if [ "${ANALYZE_ONLY}" = true ]; then
            log "Analyze-only mode"
            analyze_results "${RUN_KERNEL}"
            exit 0
        fi

        if [ "${RUN_REQUESTED}" = true ]; then
            log "Run mode: language=${RUN_LANGUAGE} kernel=${RUN_KERNEL:-all}"
            setup_benchmark_output_dir "${RESET_BENCHMARKS}"
            update_and_rebuild_project

            # Run appropriate benchmark suites
            case "${RUN_LANGUAGE}" in
                "all")
                    run_python_benchmarks "${RUN_KERNEL}"
                    run_cpp_benchmarks "${RUN_KERNEL}"
                    ;;
                "py")
                    run_python_benchmarks "${RUN_KERNEL}"
                    ;;
                "cpp")
                    run_cpp_benchmarks "${RUN_KERNEL}"
                    ;;
            esac

            # If requested, analyze after running benchmarks
            if [ "${RUN_AND_ANALYZE}" = true ]; then
                log "Running analysis after benchmark execution"
                analyze_results "${RUN_KERNEL}"
            fi
        fi
    fi

    script_end=$(date +%s)
    total_duration=$((script_end - script_start))

    log "Benchmark Suite Finished in ${total_duration}s."
    hr

    # run unit tests just to be sure
    pytest tests/
}

# --- Script Execution ---
main "$@"

exit 0
