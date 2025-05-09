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
message(STATUS "Using Python site-packages (Python_SITELIB): ${Python_SITELIB}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

set(MLX_BUILD_TESTS OFF "Build MLX tests")
set(MLX_BUILD_PYTHON_BINDINGS ON "Build Python bindings for MLX from this source tree")
set(MLX_BUILD_METAL ON "Build Metal library for MLX from this source tree")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/mlx)

include_directories(
    SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mlx/mlx
)

find_package(MLX CONFIG REQUIRED)

# --- Metal Library Build (Generalized) ---
set(METALLIB_TARGET pal_metallib)
set(METALLIB_TITLE "pal")

# --- Core C++ Static Library ---
file(GLOB_RECURSE CORE_LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(${CORE_LIB_NAME} STATIC ${CORE_LIB_SOURCES})

target_include_directories(${CORE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${CORE_LIB_NAME} PUBLIC mlx)

target_include_directories(
    ${CORE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

file(GLOB_RECURSE METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.metal")
message(STATUS "Configuring Metal library build for PAL with sources: ${METAL_SOURCES}")
mlx_build_metallib(
    TARGET ${METALLIB_TARGET}
    TITLE ${METALLIB_TITLE}
    SOURCES ${METAL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(${CORE_LIB_NAME} ${METALLIB_TARGET})

# Define the Nanobind module
nanobind_add_module(
    ${EXTENSION_NAME}
    STABLE_ABI NOMINSIZE
    NB_DOMAIN mlx
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

target_link_libraries(${EXTENSION_NAME} PRIVATE ${CORE_LIB_NAME})

target_compile_definitions(
    ${EXTENSION_NAME}
    PRIVATE
        MODULE_NAME=$<TARGET_FILE_BASE_NAME:${EXTENSION_NAME}>
        VERSION_INFO="${PY_FULL_VERSION}"
        $<$<CONFIG:Debug>:ENABLE_ASSERTIONS>
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
