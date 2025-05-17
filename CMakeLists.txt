cmake_minimum_required(VERSION 3.20)
project(proxy_attention_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  if (APPLE)
    set(CMAKE_OBJC_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

if(APPLE)
  execute_process(
    COMMAND sysctl -n hw.physicalcpu
    OUTPUT_VARIABLE CMAKE_BUILD_PARALLEL_LEVEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(UNIX)
  execute_process(
    COMMAND nproc
    OUTPUT_VARIABLE CMAKE_BUILD_PARALLEL_LEVEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(WIN32)
  set(CMAKE_BUILD_PARALLEL_LEVEL $ENV{NUMBER_OF_PROCESSORS})
else()
  set(CMAKE_BUILD_PARALLEL_LEVEL 4)
endif()

add_subdirectory(src/pal_core)

# Recursively find benchmark directories by locating their CMakeLists.txt files.
file(GLOB_RECURSE _BENCHMARK_CMAKE_FILES
     CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/benchmarks/*/cpp/CMakeLists.txt"
)

set(BENCHMARK_DIRS "") # Initialize list of benchmark directories
foreach(_benchmark_cmake_file ${_BENCHMARK_CMAKE_FILES})
    get_filename_component(_benchmark_dir ${_benchmark_cmake_file} DIRECTORY)
    list(APPEND BENCHMARK_DIRS ${_benchmark_dir})
endforeach()

message(STATUS "BENCHMARK_DIRS: ${BENCHMARK_DIRS}")
foreach(BENCHMARK_DIR ${BENCHMARK_DIRS})
    message(STATUS "Adding benchmark directory: ${BENCHMARK_DIR}")
    add_subdirectory(${BENCHMARK_DIR})
endforeach()
