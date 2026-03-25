cmake_minimum_required(VERSION 3.21)
project(llama_inference_hip LANGUAGES CXX)

# ---------------------------------------------------------------------------
# ROCm / HIP toolchain discovery
# ---------------------------------------------------------------------------
find_package(hip REQUIRED CONFIG
    PATHS /opt/rocm/lib/cmake/hip
          $ENV{ROCM_PATH}/lib/cmake/hip)

if(NOT DEFINED AMDGPU_TARGETS)
    set(AMDGPU_TARGETS "gfx900;gfx906;gfx908;gfx1030;gfx1100" CACHE STRING
        "AMD GPU targets to compile for")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(
    -O3
    -Wall
    -Wextra
    -ffast-math
)

# ---------------------------------------------------------------------------
# Source files
# hip_kernels.cpp must be compiled as a HIP translation unit
# All other files are plain C++
# ---------------------------------------------------------------------------
set_source_files_properties(
    src/hip_kernels.cpp
    PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE
)

add_executable(llama_hip
    src/hip_kernels.cpp
    src/transformer.cpp
    src/tokenizer.cpp
    src/main.cpp
)

target_include_directories(llama_hip PRIVATE include)

target_link_libraries(llama_hip
    hip::host
    hip::device
    m
)

# compile for each requested GPU target
set_target_properties(llama_hip PROPERTIES
    HIP_ARCHITECTURES "${AMDGPU_TARGETS}"
)

# ---------------------------------------------------------------------------
# Install step: copy binary to soc-analyzer workloads/ for UI pickup
# ---------------------------------------------------------------------------
install(TARGETS llama_hip DESTINATION ${CMAKE_SOURCE_DIR}/../../../workloads)

message(STATUS "llama_hip configured for GPU targets: ${AMDGPU_TARGETS}")
