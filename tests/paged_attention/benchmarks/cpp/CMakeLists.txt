cmake_minimum_required(VERSION 3.20)
project(pal_op_benchmarks CXX)

# --- C++ Settings (can inherit or be simpler than main lib) ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Typically OFF for executables unless needed

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Lighter debug flags for benchmarks might be preferable for speed
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
else()
    # Release flags are important for benchmark performance
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=thin") # ThinLTO for faster link times
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    endif()
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
add_executable(${BENCHMARK_EXE_NAME} pal_op_benchmarks.cpp)

target_include_directories(${BENCHMARK_EXE_NAME}
    PRIVATE
        ${MLX_INCLUDE_DIRS}
        ${GOOGLETEST_INCLUDE_DIRS}
        ${benchmark_SOURCE_DIR}/include
)

target_link_libraries(${BENCHMARK_EXE_NAME}
    PRIVATE
        pal_core_lib
        mlx
        benchmark
)

install(TARGETS ${BENCHMARK_EXE_NAME} DESTINATION bin)

include(GoogleTest)
add_test(NAME PalOpBenchmarks COMMAND ${BENCHMARK_EXE_NAME} --benchmark_filter=all)
