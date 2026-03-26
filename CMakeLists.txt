cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# hipcc compiler setup.
# Detect from ROCM_PATH env var, /opt/rocm, or system PATH in that order.
# Must be set BEFORE the project() call.
# ---------------------------------------------------------------------------
if(DEFINED ENV{ROCM_PATH})
    set(_HIPCC "$ENV{ROCM_PATH}/bin/hipcc")
elseif(EXISTS "/opt/rocm/bin/hipcc")
    set(_HIPCC "/opt/rocm/bin/hipcc")
else()
    find_program(_HIPCC hipcc DOC "AMD HIP compiler")
endif()

if(NOT _HIPCC OR NOT EXISTS "${_HIPCC}")
    message(FATAL_ERROR
        "hipcc not found. Install it with:\n"
        "  sudo apt-get update && sudo apt-get install -y hipcc\n"
        "Or set the ROCM_PATH environment variable to your ROCm install root.")
endif()

set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP C++ compiler" FORCE)

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

# Hardcode Ubuntu 22.04 gcc-11 paths so hipcc clang doesn't get lost
target_include_directories(llama_hip SYSTEM PRIVATE 
    /usr/include/c++/11 
    /usr/include/x86_64-linux-gnu/c++/11
)
target_link_directories(llama_hip PRIVATE /usr/lib/gcc/x86_64-linux-gnu/11)

target_include_directories(llama_hip PRIVATE include)
target_link_libraries(llama_hip m)

# Install binary to soc-analyzer/workloads/ so the UI can pick it up
install(TARGETS llama_hip DESTINATION ${CMAKE_SOURCE_DIR}/../../../workloads)

message(STATUS "Building llama_hip with hipcc: ${CMAKE_CXX_COMPILER}")
message(STATUS "GPU targets: ${AMDGPU_TARGETS}")
