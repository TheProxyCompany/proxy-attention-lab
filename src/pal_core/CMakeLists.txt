cmake_minimum_required(VERSION 3.20)
project(pal_cpp LANGUAGES CXX)

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

set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# --- Nanobind ---
add_subdirectory(${PROJECT_ROOT_DIR}/external/nanobind ${CMAKE_CURRENT_BINARY_DIR}/nanobind EXCLUDE_FROM_ALL)
message(STATUS "Added nanobind subdirectory")

# --- MLX ---
option(MLX_BUILD_TESTS "Build MLX tests" OFF)
option(MLX_BUILD_METAL "Build Metal library" ON)
add_subdirectory(${PROJECT_ROOT_DIR}/external/mlx ${CMAKE_CURRENT_BINARY_DIR}/mlx EXCLUDE_FROM_ALL)

target_compile_options(mlx PRIVATE -w)
find_package(MLX CONFIG REQUIRED)
message(STATUS "Added mlx subdirectory")

# --- Metal Library Build (Generalized) ---
set(METALLIB_TARGET pal_metallib)
set(METALLIB_TITLE "pal")
file(GLOB_RECURSE METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*/*.metal")
set(METAL_HEADERS "")
set(METAL_INCLUDE_DIRS "")

foreach(metal_file ${METAL_SOURCES})
    message(STATUS "Found metal file: ${metal_file}")
    cmake_path(GET metal_file PARENT_PATH dir)
    cmake_path(GET metal_file STEM name)
    set(header_file "${dir}/${name}.h")
    if(EXISTS "${header_file}")
        list(APPEND METAL_HEADERS "${header_file}")
        message(STATUS "Added metal header: ${header_file}")
    endif()
    list(APPEND METAL_INCLUDE_DIRS "${dir}")
    message(STATUS "Added metal include directory: ${dir}")
endforeach()
list(REMOVE_DUPLICATES METAL_INCLUDE_DIRS)

message(STATUS "Configuring Metal library build for PAL with sources: ${METAL_SOURCES}")
mlx_build_metallib(
    TARGET ${METALLIB_TARGET}
    TITLE ${METALLIB_TITLE}
    SOURCES ${METAL_SOURCES}
    INCLUDE_DIRS ${METAL_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${METAL_HEADERS}
)

# --- Core C++ Static Library ---
file(
    GLOB_RECURSE CORE_LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp"
)
file(
    GLOB_RECURSE CORE_LIB_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp"
)
add_library(${CORE_LIB_NAME} STATIC ${CORE_LIB_SOURCES})

target_link_libraries(
    ${CORE_LIB_NAME}
    PUBLIC
        mlx
)

target_include_directories(
    ${CORE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Define the Nanobind module
nanobind_add_module(
    ${EXTENSION_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp
)

target_link_libraries(
    ${EXTENSION_NAME}
    PRIVATE
        ${CORE_LIB_NAME}
        mlx
)

target_include_directories(
    ${EXTENSION_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure the Metal library is built before the binding module
if(MLX_BUILD_METAL AND TARGET ${METALLIB_TARGET})
    add_dependencies(${EXTENSION_NAME} ${METALLIB_TARGET})
endif()

# --- Installation ---
set(PYTHON_MODULE_INSTALL_DESTINATION "${PROJECT_NAME}")

install(
    TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT python_modules
)

if(MLX_BUILD_METAL AND TARGET ${METALLIB_TARGET})
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${METALLIB_TITLE}.metallib
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
