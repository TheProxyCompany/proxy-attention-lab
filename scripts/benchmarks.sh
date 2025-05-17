#!/usr/bin/env bash
#
# Script to build and run benchmarks for the project with flexible control.
# - Can update dependencies and rebuild the project.
# - Can discover and run Python pytest-benchmark tests (all or filtered).
# - Can discover and run C++ Google Benchmark executables (all or filtered).
# - Can analyze benchmark results from previous runs.

set -euo pipefail # Exit on error, unset var, pipe failure
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

# --- Configuration ---
VENV_DIR=".venv"
BUILD_DIR="build" # CMake build directory
UV_EXECUTABLE_PATH=""  # Will be detected

# Directories for benchmark discovery
PYTHON_BENCHMARK_ROOT_DIR="tests" # Root for Python benchmark discovery
CPP_BENCHMARK_BUILD_ROOT_DIR="build/tests" # Root for C++ benchmark executable discovery
BENCHMARK_OUTPUT_ROOT=".benchmarks" # Output directory for benchmark results

# Naming conventions for discovery
PYTHON_BENCHMARK_PATTERN="paged_attention/benchmarks/python/*.py" # Pattern for Python benchmark files

# --- Helper Functions ---
log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

hr() {
    printf "%80s\n" | tr ' ' '-'
}

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --run [suite_target]     Run benchmark suite(s). If suite_target is omitted, runs all.
                           Valid targets: all, py, cpp, py_pal, py_sdpa, cpp_pal, cpp_sdpa
  --analyze                Only analyze existing benchmark results
  --rebuild-only           Only update dependencies and rebuild the project
  --help                   Show this help message

Combined Options:
  --run ... --analyze      Run the specified benchmarks and then analyze the results

Examples:
  $(basename "$0")                             # Default: rebuild, run all, analyze
  $(basename "$0") --run                       # Rebuild, run all, NO analysis
  $(basename "$0") --run py_pal                # Rebuild, run only Python PAL benchmarks
  $(basename "$0") --run cpp_sdpa              # Rebuild, run only C++ SDPA benchmarks
  $(basename "$0") --run py --analyze          # Run Python benchmarks and analyze results
  $(basename "$0") --analyze                   # Only analyze existing data
  $(basename "$0") --rebuild-only              # Only update deps and rebuild
EOF
}

# --- Core Action Functions ---

setup_environment() {
    log "Setting up environment..."
    detect_uv
    activate_venv
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
    log "Setting up benchmark output directory: ${BENCHMARK_OUTPUT_ROOT}"
    rm -rf "${BENCHMARK_OUTPUT_ROOT}"
    mkdir -p "${BENCHMARK_OUTPUT_ROOT}"
}

run_python_benchmarks() {
    local target="${1:-all}"
    local filter_option=""
    local benchmark_start benchmark_end benchmark_duration

    hr
    log "Running Python benchmarks (target: ${target})..."

    benchmark_start=$(date +%s)

    # Set filter based on target
    case "${target}" in
        "py_pal"|"pal")
            filter_option="-k test_pal_"
            log "Filtering for PAL Python benchmarks only"
            ;;
        "py_sdpa"|"sdpa")
            filter_option="-k test_sdpa_"
            log "Filtering for SDPA Python benchmarks only"
            ;;
        "all"|"py"|*)
            filter_option="" # Run all Python benchmarks
            log "Running all Python benchmarks"
            ;;
    esac

    # Find all .py files in any benchmarks/python subfolder under tests/
    python_benchmark_files=()
    while IFS= read -r -d $'\0' file; do
        python_benchmark_files+=("$file")
    done < <(find "${PYTHON_BENCHMARK_ROOT_DIR}" -type d -path "*/benchmarks/python" -print0 | while IFS= read -r -d $'\0' dir; do
        find "$dir" -maxdepth 1 -type f -name "*.py" -print0
    done)

    if [ ${#python_benchmark_files[@]} -eq 0 ]; then
        log "No Python benchmark files found in any 'benchmark/python' subfolder under '${PYTHON_BENCHMARK_ROOT_DIR}'."
    else
        for benchmark_file in "${python_benchmark_files[@]}"; do
            hr
            log "Running Python benchmarks in: ${benchmark_file}"

            # Generate a unique JSON output name based on the benchmark file name
            local benchmark_basename
            benchmark_basename=$(basename "${benchmark_file}" .py)
            local python_json_output="${BENCHMARK_OUTPUT_ROOT}/${benchmark_basename}_results.json"

            # Run pytest with appropriate filters
            pytest "${benchmark_file}" \
                --benchmark-only \
                --benchmark-columns="min,max,mean,stddev,rounds,iterations" \
                --benchmark-json="${python_json_output}" \
                ${filter_option} \
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
    local target="${1:-all}"
    local filter_option=""
    local benchmark_start benchmark_end benchmark_duration

    hr
    log "Running C++ benchmarks (target: ${target})..."

    benchmark_start=$(date +%s)

    # Set filter based on target
    case "${target}" in
        "cpp_pal"|"pal")
            filter_option="--benchmark_filter=BM_PAL_"
            log "Filtering for PAL C++ benchmarks only"
            ;;
        "cpp_sdpa"|"sdpa")
            filter_option="--benchmark_filter=BM_SDPA_"
            log "Filtering for SDPA C++ benchmarks only"
            ;;
        "all"|"cpp"|*)
            filter_option="" # Run all C++ benchmarks
            log "Running all C++ benchmarks"
            ;;
    esac

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
            local cpp_json_output="${BENCHMARK_OUTPUT_ROOT}/${benchmark_exename}_results.json"

            SPDLOG_LEVEL=warn "${benchmark_exe}" \
                --benchmark_format=json \
                --benchmark_out="${cpp_json_output}" \
                --benchmark_repetitions=3 \
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
    hr
    log "Analyzing benchmark results..."

    # Get the directory where the script is located
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local analyze_script="${script_dir}/analyze_benchmarks.py"

    # Check if analysis script exists
    if [ ! -f "${analyze_script}" ]; then
        log "ERROR: Analysis script not found at ${analyze_script}"
        return 1
    fi

    # Check if there are benchmark results to analyze
    if [ ! -d "${BENCHMARK_OUTPUT_ROOT}" ] || [ -z "$(ls -A "${BENCHMARK_OUTPUT_ROOT}" 2>/dev/null)" ]; then
        log "WARNING: No benchmark results found in ${BENCHMARK_OUTPUT_ROOT}"
        log "Run benchmarks first or ensure benchmark JSON files exist"
        return 1
    fi

    # Run analysis script
    log "Running analysis script on results in ${BENCHMARK_OUTPUT_ROOT}..."
    python "${analyze_script}" --results-dir "${BENCHMARK_OUTPUT_ROOT}" --output-dir "${BENCHMARK_OUTPUT_ROOT}"

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
    local TARGET_SUITE="all"

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
        setup_benchmark_output_dir
        update_and_rebuild_project
        run_python_benchmarks "all"
        run_cpp_benchmarks "all"
        analyze_results
    else
        # Process flags
        local RUN_AND_ANALYZE=false

        while [ $# -gt 0 ]; do
            case "$1" in
                --run)
                    RUN_REQUESTED=true
                    if [ $# -gt 1 ] && [[ ! "$2" == --* ]]; then
                        TARGET_SUITE="$2"
                        shift
                    else
                        TARGET_SUITE="all"
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

        # Validate target suite if specified
        if [ "${RUN_REQUESTED}" = true ] && ! [[ "${TARGET_SUITE}" =~ ^(all|py|cpp|py_pal|py_sdpa|cpp_pal|cpp_sdpa)$ ]]; then
            log "ERROR: Invalid target suite: ${TARGET_SUITE}"
            log "Valid values: all, py, cpp, py_pal, py_sdpa, cpp_pal, cpp_sdpa"
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
            analyze_results
            exit 0
        fi

        if [ "${RUN_REQUESTED}" = true ]; then
            log "Run mode: target=${TARGET_SUITE}"
            setup_benchmark_output_dir
            update_and_rebuild_project

            # Run appropriate benchmark suites
            case "${TARGET_SUITE}" in
                "all")
                    run_python_benchmarks "all"
                    run_cpp_benchmarks "all"
                    ;;
                "py")
                    run_python_benchmarks "all"
                    ;;
                "cpp")
                    run_cpp_benchmarks "all"
                    ;;
                "py_pal")
                    run_python_benchmarks "pal"
                    ;;
                "py_sdpa")
                    run_python_benchmarks "sdpa"
                    ;;
                "cpp_pal")
                    run_cpp_benchmarks "pal"
                    ;;
                "cpp_sdpa")
                    run_cpp_benchmarks "sdpa"
                    ;;
            esac

            # If requested, analyze after running benchmarks
            if [ "${RUN_AND_ANALYZE}" = true ]; then
                log "Running analysis after benchmark execution"
                analyze_results
            fi
        fi
    fi

    script_end=$(date +%s)
    total_duration=$((script_end - script_start))

    log "Benchmark Suite Finished in ${total_duration}s."
    hr
}

# --- Script Execution ---
main "$@"

exit 0
