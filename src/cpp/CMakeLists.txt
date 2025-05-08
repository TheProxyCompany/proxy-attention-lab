cmake_minimum_required(VERSION 3.20)
project(pal_cpp_binding LANGUAGES CXX)

# --- C++ Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Set compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

# --- Find Dependencies ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find MLX
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}")
message(STATUS "Found MLX at: ${MLX_ROOT}")

# Find Nanobind
find_package(nanobind CONFIG REQUIRED)
message(STATUS "Found nanobind version: ${nanobind_VERSION}")

# --- Metal Library Build ---
set(PAL_METALLIB_TARGET pal_metallib)      # CMake target name for the metallib
set(PAL_METALLIB_TITLE "pal")              # Title used for registration (matches python package)
set(PAL_METAL_SOURCE "paged_attention.metal") # Source file in this directory
set(PAL_METAL_HEADERS "paged_attention.h")     # Header file in this directory

# Use mlx_build_metallib
if(MLX_BUILD_METAL)
    message(STATUS "Configuring Metal library build for PAL")
    mlx_build_metallib(
        TARGET ${PAL_METALLIB_TARGET}
        TITLE ${PAL_METALLIB_TITLE}
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${PAL_METAL_SOURCE}
        INCLUDE_DIRS
             ${CMAKE_CURRENT_SOURCE_DIR}
             ${MLX_INCLUDE_DIRS}
        OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPS ${PAL_METAL_HEADERS}
    )
else()
    message(WARNING "MLX_BUILD_METAL is OFF. Metal library will not be built.")
endif()

# --- Python Binding Module ---
set(PAL_BINDING_MODULE_NAME "_pal_cpp_binding") # Python import name (internal)
set(PAL_BINDING_SOURCE "binding.cpp")           # C++ binding source file

# Define the Nanobind module
nanobind_add_module(
    ${PAL_BINDING_MODULE_NAME}
    # NB_STATIC # Avoid static linking for simpler dev builds if possible
    ${PAL_BINDING_SOURCE}
)

# Link the binding against MLX and nanobind
target_link_libraries(
    ${PAL_BINDING_MODULE_NAME}
    PRIVATE
        mlx          # Link against the MLX library
        nanobind::nanobind # Link against nanobind
)

# Include directories for the binding
target_include_directories(
    ${PAL_BINDING_MODULE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For paged_attention.h etc.
        ${MLX_INCLUDE_DIRS}
        ${nanobind_INCLUDE_DIRS}
)

# Ensure the Metal library is built before the binding module
if(MLX_BUILD_METAL AND TARGET ${PAL_METALLIB_TARGET})
    add_dependencies(${PAL_BINDING_MODULE_NAME} ${PAL_METALLIB_TARGET})
endif()

# --- Installation ---
set(PYTHON_MODULE_INSTALL_DESTINATION "pal") # Install into 'pal' package dir
set(INSTALL_COMPONENT_NAME "pal_python_module") # Must match pyproject.toml

# Install the compiled C++ extension module (.so/.dylib)
install(
    TARGETS ${PAL_BINDING_MODULE_NAME}
    LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT_NAME} # Assign to component
)

# Install the Metal library alongside the Python module
if(MLX_BUILD_METAL AND TARGET ${PAL_METALLIB_TARGET})
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${PAL_METALLIB_TITLE}.metallib
        DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
        COMPONENT ${INSTALL_COMPONENT_NAME} # Assign to component
    )
endif()

# --- Automatic Stub Generation ---
# (Optional but recommended for type hinting)
nanobind_add_stub(
    pal_stubgen_target # Unique target name for stub generation
    MODULE ${PAL_BINDING_MODULE_NAME}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_pal_cpp_binding.pyi # Output stub file
    PYTHON_PATH $<TARGET_FILE_DIR:${PAL_BINDING_MODULE_NAME}> # Path to the built module
    DEPENDS ${PAL_BINDING_MODULE_NAME} # Depends on the module being built
    MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/py.typed # Creates py.typed marker
)

# Install the generated stub file and py.typed marker
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/_pal_cpp_binding.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT_NAME} # Assign to component
)
