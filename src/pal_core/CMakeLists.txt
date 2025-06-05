cmake_minimum_required(VERSION 3.20)
project(pal_cpp LANGUAGES CXX)
cmake_policy(SET CMP0135 NEW)

set(PROJECT_NAME proxy_attention_lab)  # Top-level project name
set(EXTENSION_NAME pal_core)           # Python module name
set(CORE_LIB_NAME pal_core_lib)        # C++ Static Library name
# pal_core can contain multiple primitives/kernels. New sources should be added
# under src/ and kernels/ and listed here as the project grows.

# --- C++ Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_MODE ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 \
        -fno-omit-frame-pointer \
        -Wall -Wextra -Wpedantic \
        -Wformat=2 -Wformat-security \
        -Wcast-align -Wcast-qual \
        -Wconversion -Wdouble-promotion \
        -Wmissing-include-dirs \
        -Wnull-dereference -Wredundant-decls \
        -Wshadow -Wstrict-aliasing=2 \
        -Wuninitialized \
        -D_GLIBCXX_DEBUG \
        -D_GLIBCXX_DEBUG_PEDANTIC \
        -D_FORTIFY_SOURCE=2"
    )
else()
    set(DEBUG_MODE OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -O3 \
        -mtune=native \
        -flto=auto \
        -DNDEBUG \
        -ffunction-sections \
        -fdata-sections \
        -ftree-vectorize \
        -fstack-protector-strong \
        -D_FORTIFY_SOURCE=2 \
        -funroll-loops \
        -finline-functions"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -fmerge-all-constants \
            -fvectorize \
            -fslp-vectorize")
    endif()
endif()

# --- Find or Fetch spdlog ---
find_package(spdlog QUIET CONFIG) # Attempt to find an installed/exported spdlog

if(NOT spdlog_FOUND) # If not found by find_package
    message(STATUS "PAL: spdlog not found via find_package. Attempting FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        spdlog # Standard FetchContent name
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
        EXCLUDE_FROM_ALL
    )
    # Make spdlog available. This typically defines targets like spdlog::spdlog_header_only or spdlog::spdlog
    FetchContent_MakeAvailable(spdlog)
else()
    message(STATUS "PAL: Found spdlog via find_package.")
endif()

# Link against spdlog - prefer header-only if available from either path
if(TARGET spdlog::spdlog_header_only)
    set(PAL_SPDLOG_LINK_TARGET spdlog::spdlog_header_only)
elseif(TARGET spdlog) # Check for compiled spdlog target
    set(PAL_SPDLOG_LINK_TARGET spdlog)
else()
    # This might occur if find_package found it but didn't create a modern target,
    # or FetchContent failed to create expected targets.
    message(WARNING "PAL: Could not determine a CMake target for spdlog. Build might fail or use system headers if lucky.")
    # As a last resort, if spdlog_INCLUDE_DIRS and spdlog_LIBRARIES were set by find_package (older style)
    if(spdlog_INCLUDE_DIRS)
        target_include_directories(${CORE_LIB_NAME} PRIVATE ${spdlog_INCLUDE_DIRS})
    endif()
    if(spdlog_LIBRARIES)
        set(PAL_SPDLOG_LINK_TARGET ${spdlog_LIBRARIES}) # This could be a list
    else()
        set(PAL_SPDLOG_LINK_TARGET "") # Avoid CMake error if var is empty
    endif()
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

# --- Metal-CPP library ---
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Check if metal_cpp_SOURCE_DIR is already defined (e.g., by a parent like MLX)
    if(DEFINED ENV{METAL_CPP_DIR} AND IS_DIRECTORY "$ENV{METAL_CPP_DIR}")
         set(PAL_METAL_CPP_INCLUDE_DIR "$ENV{METAL_CPP_DIR}")
         message(STATUS "PAL: Using Metal-CPP provided by environment METAL_CPP_DIR: ${PAL_METAL_CPP_INCLUDE_DIR}")
    elif(DEFINED metal_cpp_SOURCE_DIR AND IS_DIRECTORY "${metal_cpp_SOURCE_DIR}")
        set(PAL_METAL_CPP_INCLUDE_DIR ${metal_cpp_SOURCE_DIR})
        message(STATUS "PAL: Using Metal-CPP provided by parent project (metal_cpp_SOURCE_DIR): ${PAL_METAL_CPP_INCLUDE_DIR}")
    else()
        message(STATUS "PAL: Metal-CPP not found from parent. Fetching Metal-CPP.")
        include(FetchContent)
        set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip)
        FetchContent_Declare(
            metal_cpp # Standard name for this FetchContent
            URL ${METAL_CPP_URL}
        )
        FetchContent_MakeAvailable(metal_cpp)
        set(PAL_METAL_CPP_INCLUDE_DIR ${metal_cpp_SOURCE_DIR}) # metal_cpp_SOURCE_DIR is set by FetchContent_MakeAvailable for target 'metal_cpp'
        message(STATUS "PAL: Using Metal-CPP from FetchContent directory: ${PAL_METAL_CPP_INCLUDE_DIR}")
    endif()
    # Ensure PAL_METAL_CPP_INCLUDE_DIR is used in target_include_directories and mlx_build_metallib
else()
    message(FATAL_ERROR "Metal is not supported on this platform - Only macOS is supported for now.")
endif()

# --- Find MLX ---
if(NOT TARGET mlx)
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
    find_package(MLX CONFIG REQUIRED PATHS ${Python_SITELIB} NO_DEFAULT_PATH)
else()
    message(STATUS "PAL: MLX already found as a target")
endif()
message(STATUS "PAL: Found MLX. Include Dirs: ${MLX_INCLUDE_DIRS}. Libraries: ${MLX_LIBRARIES}")

include_directories(
    SYSTEM
    ${MLX_INCLUDE_DIRS}
    ${nanobind_INCLUDE_DIRS}
)

# --- Metal Library Build (Generalized) ---
set(METALLIB_TARGET pal_metallib)
set(METALLIB_TITLE "pal")
set(PAL_METALLIB_FILENAME "pal.metallib")

# --- Core C++ Static Library ---
file(GLOB_RECURSE CORE_LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_utils/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/debug/*.cpp"
)

add_library(${CORE_LIB_NAME} STATIC ${CORE_LIB_SOURCES})

target_include_directories(
    ${CORE_LIB_NAME}
    PUBLIC
        # PAL's own headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/>
        # Headers from dependencies that might be exposed through PAL's headers
        $<BUILD_INTERFACE:${PAL_METAL_CPP_INCLUDE_DIR}>
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
        mlx
        ${PAL_SPDLOG_LINK_TARGET}
)


file(
    GLOB_RECURSE METAL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*/*.metal"
)

set(PAL_METALLIB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Output directory for pal.metallib")
message(STATUS "Configuring Metal library build for PAL with sources: ${METAL_SOURCES}")
mlx_build_metallib(
    TARGET ${METALLIB_TARGET}
    TITLE ${METALLIB_TITLE}
    SOURCES ${METAL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
        ${PAL_METAL_CPP_INCLUDE_DIR}/
        ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${PAL_METALLIB_OUTPUT_DIRECTORY}
    OUTPUT_FILENAME ${PAL_METALLIB_FILENAME}
    # DEBUG ON
)

set(PAL_METALLIB_PATH ${PAL_METALLIB_OUTPUT_DIRECTORY}/${PAL_METALLIB_FILENAME} CACHE INTERNAL "Path to built pal.metallib")
message(STATUS "PAL Core: pal.metallib built at ${PAL_METALLIB_PATH}")

add_dependencies(${CORE_LIB_NAME} ${METALLIB_TARGET})

# Define the Nanobind module
nanobind_add_module(
    ${EXTENSION_NAME}
    STABLE_ABI NOMINSIZE
    NB_DOMAIN mlx
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

target_include_directories(
    ${EXTENSION_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${MLX_INCLUDE_DIRS}
)

target_link_libraries(
    ${EXTENSION_NAME}
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
