cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Use hipcc as the C++ compiler for the whole project.
# Set this BEFORE project() so CMake picks up the right compiler.
# If ROCM_PATH is not set, default to /opt/rocm.
# ---------------------------------------------------------------------------
if(NOT DEFINED ENV{ROCM_PATH})
    set(ENV{ROCM_PATH} "/opt/rocm")
endif()

set(CMAKE_CXX_COMPILER "$ENV{ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP C++ compiler")

project(llama_inference_hip LANGUAGES CXX)

if(NOT DEFINED AMDGPU_TARGETS)
    # gfx906 = Vega20 (MI50/MI60), gfx900 = Vega10 (Fiji/Vega)
    # gem5's VEGA_X86 models gfx801 (Fiji) by default - keep gfx900 for safety
    set(AMDGPU_TARGETS "gfx900;gfx906" CACHE STRING
        "AMD GPU ISA targets (use gfx801 for gem5 VEGA_X86, gfx90a for MI200)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Pass GPU target flags to hipcc
string(REPLACE ";" "," AMDGPU_TARGETS_COMMA "${AMDGPU_TARGETS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=${AMDGPU_TARGETS_COMMA} -O3 -ffast-math")

add_executable(llama_hip
    src/hip_kernels.cpp
    src/transformer.cpp
    src/tokenizer.cpp
    src/main.cpp
)

target_include_directories(llama_hip PRIVATE include)
target_link_libraries(llama_hip m)

# Install binary to soc-analyzer/workloads/ so the UI can pick it up
install(TARGETS llama_hip DESTINATION ${CMAKE_SOURCE_DIR}/../../../workloads)

message(STATUS "Building llama_hip with hipcc: ${CMAKE_CXX_COMPILER}")
message(STATUS "GPU targets: ${AMDGPU_TARGETS}")
