cmake_minimum_required(VERSION 3.20)
project(pal_cpp LANGUAGES CXX)
cmake_policy(SET CMP0135 NEW)

set(PROJECT_NAME proxy_attention_lab)  # Top-level project name
set(EXTENSION_NAME pal_core)           # Python module name
set(CORE_LIB_NAME pal_core_lib)        # C++ Static Library name

# --- C++ Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

# --- Python ---
set(Python_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python site-packages (Python_SITELIB): ${Python_SITELIB}")

# --- Find Nanobind ---
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
message(STATUS "Nanobind CMake directory: ${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# --- Download Metal-CPP library ---
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  include(FetchContent)
  set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip)
  FetchContent_Declare(metal_cpp URL ${METAL_CPP_URL})
  FetchContent_MakeAvailable(metal_cpp)
  message(STATUS "Metal-CPP library found in: ${metal_cpp_SOURCE_DIR}")
else()
  message(FATAL_ERROR "Metal is not supported on this platform - Only macOS is supported for now.")
endif()

find_package(fmt QUIET)
if(NOT fmt_FOUND)
    message(STATUS "PAL: fmt not found via find_package. Attempting FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG "11.2.0"
    )
    FetchContent_MakeAvailable(fmt)
    FetchContent_GetProperties(fmt)
    if(NOT fmt_POPULATED)
      FetchContent_Populate(fmt)
    endif()
else()
    message(STATUS "PAL: Found fmt via find_package. Version: ${fmt_VERSION}")
endif()

# --- Find MLX ---
find_package(MLX CONFIG QUIET)
if(NOT MLX_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE MLX_ROOT)
    message(STATUS "MLX CMake directory: ${MLX_ROOT}")
    find_package(MLX CONFIG REQUIRED)
endif()

# --- Metal Library Build (Generalized) ---
set(METALLIB_TARGET pal_metallib)
set(METALLIB_TITLE "pal")
set(PAL_METALLIB_FILENAME "pal.metallib")

# --- Core C++ Static Library ---
file(GLOB_RECURSE CORE_LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/paged_attention_primitive.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal_loader_posix.cpp"
)

message(STATUS "Core library sources: ${CORE_LIB_SOURCES}")
add_library(${CORE_LIB_NAME} STATIC ${CORE_LIB_SOURCES})

target_include_directories(
    ${CORE_LIB_NAME}
    PUBLIC
        # PAL's own headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        # Headers from dependencies that might be exposed through PAL's headers
        $<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/metal_cpp>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${MLX_INCLUDE_DIRS}
        ${nanobind_INCLUDE_DIRS}
)

target_link_libraries(
    ${CORE_LIB_NAME}
    PUBLIC
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework MetalKit"
    PRIVATE
        mlx
)

file(
    GLOB_RECURSE METAL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.metal"
)
message(STATUS "Configuring Metal library build for PAL with sources: ${METAL_SOURCES}")
mlx_build_metallib(
    TARGET ${METALLIB_TARGET}
    TITLE ${METALLIB_TITLE}
    SOURCES ${METAL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MLX_INCLUDE_DIRS}
        ${metal_cpp_SOURCE_DIR}/include
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    OUTPUT_FILENAME ${PAL_METALLIB_FILENAME}
)

add_dependencies(${CORE_LIB_NAME} ${METALLIB_TARGET})

# Define the Nanobind module
nanobind_add_module(
    ${EXTENSION_NAME}
    NB_DOMAIN mlx
    STABLE_ABI NOMINSIZE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

target_include_directories(${EXTENSION_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MLX_INCLUDE_DIRS}
        ${nanobind_INCLUDE_DIRS}
)

target_link_libraries(
    ${EXTENSION_NAME}
    PUBLIC
        mlx
    PRIVATE
        ${CORE_LIB_NAME}
)

target_compile_definitions(
    ${EXTENSION_NAME}
    PRIVATE
        MODULE_NAME=$<TARGET_FILE_BASE_NAME:${EXTENSION_NAME}>
        VERSION_INFO="${PY_FULL_VERSION}"
        $<$<CONFIG:Debug>:ENABLE_ASSERTIONS>
)

install(
  DIRECTORY ${metal_cpp_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metal_cpp
)

# --- Installation ---
set(PYTHON_MODULE_INSTALL_DESTINATION "${PROJECT_NAME}")

install(
    TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT python_modules
)

if(TARGET ${METALLIB_TARGET})
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${PAL_METALLIB_FILENAME}"
        DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
        COMPONENT python_modules
    )
endif()

# --- Automatic Stub Generation ---
nanobind_add_stub(
    pal_stubgen_target
    MODULE ${EXTENSION_NAME}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXTENSION_NAME}.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:${EXTENSION_NAME}>
    DEPENDS ${EXTENSION_NAME}
    MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/py.typed
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${EXTENSION_NAME}.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT python_modules
)
