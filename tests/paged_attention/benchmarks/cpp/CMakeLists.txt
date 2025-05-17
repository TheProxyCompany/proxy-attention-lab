cmake_minimum_required(VERSION 3.20)
project(pal_op_benchmarks CXX)

# --- Check if PAL_METALLIB_PATH is set ---
if(NOT PAL_METALLIB_PATH)
    message(FATAL_ERROR "PAL_METALLIB_PATH is not set. Ensure src/pal_core/CMakeLists.txt sets it.")
else()
    message(STATUS "PAL_METALLIB_PATH: ${PAL_METALLIB_PATH}")
endif()

# --- Find Google Test ---
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG    v1.17.0
)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Find Google Benchmark ---
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG    v1.9.3
)
FetchContent_MakeAvailable(benchmark)
FetchContent_GetProperties(benchmark SOURCE_DIR)

# --- Define Benchmark Executable ---
set(BENCHMARK_EXE_NAME pal_op_benchmarks)
file(GLOB_RECURSE BENCHMARK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${BENCHMARK_EXE_NAME} ${BENCHMARK_SOURCES})

target_include_directories(${BENCHMARK_EXE_NAME}
    PRIVATE
        ${MLX_INCLUDE_DIRS}
        ${GOOGLETEST_INCLUDE_DIRS}
        ${benchmark_SOURCE_DIR}/include
)

target_link_libraries(${BENCHMARK_EXE_NAME}
    PRIVATE
        pal_core_lib
        benchmark
)

include(GoogleTest)
add_test(NAME PalOpBenchmarks COMMAND ${BENCHMARK_EXE_NAME} --benchmark_filter=all)

add_custom_command(
    TARGET ${BENCHMARK_EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PAL_METALLIB_PATH}" # Source
            "$<TARGET_FILE_DIR:${BENCHMARK_EXE_NAME}>/pal.metallib" # Destination
    COMMENT "Copying pal.metallib to benchmark executable directory"
)
