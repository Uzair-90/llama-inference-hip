# llama-inference-hip

A HIP port of Andrej Karpathy's llama2.c transformer inference engine, structured as a proper CMake project targeting AMD GPU architectures (GCN3 / Vega / RDNA) via the ROCm software stack.

Built to run inside the SoC Analyzer gem5 APU simulation environment.

---

## Project Structure

```
llama-inference-hip/
├── CMakeLists.txt          - CMake build system
├── include/
│   ├── model.h             - All struct definitions (Config, Weights, Tokenizer, Sampler)
│   └── hip_kernels.h       - HIP kernel declarations
├── src/
│   ├── hip_kernels.cpp     - HIP __global__ kernel implementations
│   ├── transformer.cpp     - GPU-backed forward pass (weight upload + kernel dispatch)
│   ├── tokenizer.cpp       - BPE tokenizer, sampler, generation loop
│   └── main.cpp            - CLI entrypoint
├── models/                 - Place .bin model checkpoints here
└── run.c                   - Original karpathy/llama2.c reference (unmodified)
```

---

## How it Works

The original llama2.c runs entirely on the CPU. This port replaces the performance-critical sections with HIP GPU kernels while keeping the tokenizer and sampler on the host.

| Component | Original (CPU) | This port (GPU) |
|---|---|---|
| matmul W@x | OMP for loop | `kernel_matmul`: one thread per output row |
| RMSNorm | Sequential loop | `kernel_rmsnorm`: shared-memory tree reduction |
| Softmax | Sequential loop | `kernel_softmax`: two-pass shared-memory reduction |
| RoPE | Sequential loop | `kernel_rope`: one thread per complex pair |
| Attention scores + accumulation | OMP for loop over heads | `kernel_attention`: one block per head |
| SwiGLU | Sequential loop | `kernel_swiglu`: element-wise parallel |
| Residual adds | Sequential loop | `kernel_add`: element-wise parallel |
| Tokenizer / Sampler | CPU | CPU (unchanged) |

All transformer weights are uploaded to device memory (HBM) on startup. Activation buffers live on device for the entire generation run. Only final logits are copied back to host per step for sampling.

---

## Prerequisites

- ROCm >= 5.x (`sudo apt install rocm-hip-sdk`)
- CMake >= 3.21
- A llama2 compatible `.bin` checkpoint and `tokenizer.bin`

To obtain the tiny 15M parameter test model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -P models/
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -P models/
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DAMDGPU_TARGETS="gfx900;gfx906"
make -j$(nproc)
```

The binary is placed in `build/bin/llama_hip`. Running `cmake --install .` will also copy it to `workloads/` for direct use in the SoC Analyzer UI.

---

## Run

```bash
./build/bin/llama_hip models/stories15M.bin \
    -z models/tokenizer.bin \
    -i "Once upon a time" \
    -t 0.8 \
    -n 200
```

---

## Running inside gem5 APU Simulation

1. Build the binary as shown above.
2. Open the SoC Analyzer web UI at `http://localhost:8000`.
3. Navigate to **Sim Config**.
4. Set **Architecture** to `X86 + APU (CPU & GPU / Ruby Memory)`.
5. Upload or select `llama_hip` from the **Workload Binary** dropdown.
6. Set **Command Line Arguments** to:
   ```
   models/stories15M.bin -z models/tokenizer.bin -i "Once upon a time" -n 50
   ```
7. Click **Run gem5 Simulation** and observe the generation trace in the Simulation terminal.

---

## Notes

- The gem5 VEGA_X86 binary uses the GFX801 (Fiji) GPU model by default.
- The `kernel_attention` kernel places one block per attention head; block dimensions are set to 128 threads with shared memory for the softmax reduction.
- For small models (15M, 42M) this binary runs entirely within the simulated DDR + HBM address space without address translation issues.
