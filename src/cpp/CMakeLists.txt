cmake_minimum_required(VERSION 3.20)
project(pal_cpp_core LANGUAGES CXX)

# --- C++ Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra")
else() # Release, RelWithDebInfo
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

# --- Find Dependencies via Submodules ---
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
message(STATUS "Project root directory: ${PROJECT_ROOT_DIR}")

# --- Python ---
set(Python_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# --- Nanobind ---
add_subdirectory(${PROJECT_ROOT_DIR}/external/nanobind ${CMAKE_CURRENT_BINARY_DIR}/nanobind EXCLUDE_FROM_ALL)
message(STATUS "Added nanobind subdirectory")

# --- MLX ---
option(MLX_BUILD_TESTS "Build MLX tests" OFF)
add_subdirectory(${PROJECT_ROOT_DIR}/external/mlx ${CMAKE_CURRENT_BINARY_DIR}/mlx EXCLUDE_FROM_ALL)
find_package(MLX CONFIG REQUIRED)
target_compile_options(mlx PRIVATE -w)
message(STATUS "Added mlx subdirectory")

# --- Metal Library Build (Generalized) ---
set(PAL_METALLIB_TARGET pal_metallib)
set(PAL_METALLIB_TITLE "pal")
file(GLOB_RECURSE PAL_METAL_SOURCES
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/*/*.metal")
list(TRANSFORM PAL_METAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(PAL_METAL_HEADERS "")
set(PAL_METAL_INCLUDE_DIRS "")

foreach(metal_file ${PAL_METAL_SOURCES})
    message(STATUS "Found metal file: ${metal_file}")
    cmake_path(GET metal_file PARENT_PATH dir)
    cmake_path(GET metal_file STEM name)
    set(header_file "${dir}/${name}.h")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}")
        list(APPEND PAL_METAL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}")
        message(STATUS "Added metal header: ${header_file}")
    endif()
    list(APPEND PAL_METAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    message(STATUS "Added metal include directory: ${dir}")
endforeach()
list(REMOVE_DUPLICATES PAL_METAL_INCLUDE_DIRS)
list(APPEND PAL_METAL_INCLUDE_DIRS ${MLX_INCLUDE_DIRS}) # Add MLX includes

if(PAL_METAL_SOURCES AND MLX_BUILD_METAL)
    message(STATUS "Configuring Metal library build for PAL with sources: ${PAL_METAL_SOURCES}")
    mlx_build_metallib(
        TARGET ${PAL_METALLIB_TARGET}
        TITLE ${PAL_METALLIB_TITLE}
        SOURCES ${PAL_METAL_SOURCES}
        INCLUDE_DIRS ${PAL_METAL_INCLUDE_DIRS}
        OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PAL_METAL_HEADERS}
    )
elseif(NOT MLX_BUILD_METAL)
    message(WARNING "MLX_BUILD_METAL is OFF. Metal library will not be built.")
else()
    message(WARNING "No .metal source files found in subdirectories of src/cpp.")
endif()

# --- Core C++ Static Library ---
set(PAL_CORE_LIB_NAME pal_core_lib)

file(GLOB_RECURSE PAL_CORE_LIB_SOURCES
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp"
)
if(NOT PAL_CORE_LIB_SOURCES)
    message(FATAL_ERROR "No core C++ source files found in subdirectories of src/cpp. Check paths.")
endif()
message(STATUS "Found core library sources: ${PAL_CORE_LIB_SOURCES}")

add_library(${PAL_CORE_LIB_NAME} STATIC ${PAL_CORE_LIB_SOURCES})

# Link the core lib against MLX
target_link_libraries(
    ${PAL_CORE_LIB_NAME}
    PUBLIC
        mlx
)

# Automatically add include directories based on where sources were found
set(PAL_CORE_INCLUDE_DIRS "")
foreach(core_source ${PAL_CORE_LIB_SOURCES})
    cmake_path(GET core_source PARENT_PATH dir)
    list(APPEND PAL_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
endforeach()
list(REMOVE_DUPLICATES PAL_CORE_INCLUDE_DIRS)
list(APPEND PAL_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}") # Add src/cpp itself

message(STATUS "Core library include directories: ${PAL_CORE_INCLUDE_DIRS}")

target_include_directories(
    ${PAL_CORE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PAL_CORE_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # MLX includes handled by target_link_libraries PUBLIC
)

# --- Python Binding Module ---
set(PAL_BINDING_MODULE_NAME "_pal_cpp_binding")
set(PAL_BINDING_SOURCE "binding.cpp") # Only the binding code

# Define the Nanobind module
nanobind_add_module(
    ${PAL_BINDING_MODULE_NAME}
    ${PAL_BINDING_SOURCE}
)

target_link_libraries(
    ${PAL_BINDING_MODULE_NAME}
    PRIVATE
        ${PAL_CORE_LIB_NAME}
)

target_include_directories(
    ${PAL_BINDING_MODULE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure the Metal library is built before the binding module
if(MLX_BUILD_METAL AND TARGET ${PAL_METALLIB_TARGET})
    add_dependencies(${PAL_BINDING_MODULE_NAME} ${PAL_METALLIB_TARGET})
endif()

# --- Installation ---
set(PYTHON_MODULE_INSTALL_DESTINATION "pal")
set(INSTALL_COMPONENT_NAME "pal_python_module")

install(
    TARGETS ${PAL_BINDING_MODULE_NAME}
    LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT_NAME}
)

if(MLX_BUILD_METAL AND TARGET ${PAL_METALLIB_TARGET})
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${PAL_METALLIB_TITLE}.metallib
        DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
        COMPONENT ${INSTALL_COMPONENT_NAME}
    )
endif()

# --- Automatic Stub Generation ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
nanobind_add_stub(
    pal_stubgen_target
    MODULE ${PAL_BINDING_MODULE_NAME}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_pal_cpp_binding.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:${PAL_BINDING_MODULE_NAME}>
    DEPENDS ${PAL_BINDING_MODULE_NAME}
    MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/py.typed
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/_pal_cpp_binding.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${PYTHON_MODULE_INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT_NAME}
)
