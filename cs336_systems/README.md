# cs336_systems — benchmarking & profiling

<!-- CO AUTHORED BY CLAUDE.
Maintenance rule (see AGENTS.md): any change that can affect benchmarking — a new flag, a different way to run part of
the model (FlashAttention-2, torch.compile, mixed precision, checkpointing), new NVTX ranges, profiler options —
must update this file in the same change: the relevant section, §5 "How settings change the profiles", and §11 Changelog. -->

Tools for assignment 2 (systems): end-to-end timing of the assignment 1 `TransformerLM`, plus how to profile it
with NVIDIA Nsight Systems (where time goes) and PyTorch's memory snapshot (what holds GPU memory).

**Contents**

1. [Files & quick start](#1-files--quick-start)
2. [Model sizes: classes, architecture, parameter counts](#2-model-sizes)
3. [The benchmark: functions, return types, what one step does](#3-the-benchmark)
4. [Flags](#4-flags)
5. [How settings change the profiles](#5-how-settings-change-the-profiles)
6. [Output & baseline results](#6-output--baseline-results)
7. [Profiling with Nsight Systems](#7-profiling-with-nsight-systems)
8. [Memory profiling](#8-memory-profiling)
9. [Nsight app over SSH](#9-nsight-app-over-ssh)
10. [Pod workflow](#10-pod-workflow)
11. [Changelog](#11-changelog)

---

## 1. Files & quick start

| File | What it is |
|---|---|
| `benchmark.py` | End-to-end benchmark. Builds a model, runs warm-up + timed steps, times each phase, reports peak memory; optionally writes a memory snapshot. |
| `model_configs.py` | Table 1 model sizes, shared defaults, analytic parameter count and memory estimate. |
| `README.md` | This file. |

Run everything from the **repo root**:

```bash
uv run python -m cs336_systems.benchmark --size small                 # one full training step, small model
uv run python -m cs336_systems.benchmark --size all --mode forward    # inference, every size
uv run python -m cs336_systems.benchmark --size small --warmup 1 --steps 2 \
  --memory-snapshot profiles/mem_small_train.pickle                   # memory snapshot (CUDA only, §8)
uv run python -m cs336_systems.benchmark --help                       # every flag
```

When your shell config isn't loaded (the Nsight app, a one-off `ssh host 'cmd'`), call the project Python directly:
`/workspace/assignment1-basics/.venv/bin/python -m cs336_systems.benchmark …`

---

## 2. Model sizes

### 2.1 What's in `model_configs.py`

| Name | Kind | Value / signature | Purpose |
|---|---|---|---|
| `VOCAB_SIZE` | `int` | `10_000` | Vocab for every benchmark (matches the TinyStories tokenizer) |
| `BATCH_SIZE` | `int` | `4` | Default `--batch-size` |
| `CONTEXT_LENGTH` | `int` | `512` | Default `--context-length` |
| `ROPE_THETA` | `float` | `10_000.0` | RoPE base passed to every model |
| `ModelSize` | `@dataclass(frozen=True)` | fields `name: str, d_model: int, d_ff: int, num_layers: int, num_heads: int` | One row of Table 1. Frozen (immutable); overrides make a copy via `dataclasses.replace`. |
| `MODEL_SIZES` | `dict[str, ModelSize]` | keys `"small"`, `"medium"`, `"large"`, `"xl"`, `"10B"` (insertion order = size order) | Lookup used by `--size` |
| `count_parameters(size, vocab_size=VOCAB_SIZE)` | `-> int` | | Exact parameter count computed on paper (no allocation). Verified equal to `sum(p.numel() for p in model.parameters())`. |
| `training_state_gb(size, vocab_size=VOCAB_SIZE)` | `-> float` | `count_parameters(size) * 16 / 1e9` | fp32 weights + grads + AdamW `m` + `v` (4 copies × 4 bytes), **before activations** |

A size becomes a model in `benchmark.build_model`:
`TransformerLM(VOCAB_SIZE, context_length, num_layers, d_model, num_heads, d_ff, theta=ROPE_THETA, device=device)`.
`context_length` also sets RoPE's precomputed table length (`max_seq_len`).

### 2.2 The model being benchmarked (`cs336_basics.model`)

Shapes use **B** = batch, **T** = context length, **d** = d_model, **h** = heads, **d_k** = d/h, **V** = vocab, **f** = d_ff.

```
TransformerLM                                        in: (B, T) int64 token ids
├── token_embeddings   Embedding  weight (V, d)      → (B, T, d)        row lookup: weight[x]
├── layers             ModuleList of num_layers × TransformerBlock      (B, T, d) → (B, T, d)
│   └── TransformerBlock (pre-norm)
│       ├── ln1        RMSNorm    weight (d)
│       ├── attn       multihead_self_attention
│       │   ├── q_proj, k_proj, v_proj   Linear weight (d, d) each      → (B, T, d) → rearranged (B, h, T, d_k)
│       │   ├── rope   RotaryPositionEmbedding  buffers cos/sin (T, d_k/2), non-persistent, no params
│       │   ├── scaled_dot_product_attention (functional)               scores (B, h, T, T) → out (B, h, T, d_k)
│       │   └── output_proj               Linear weight (d, d)          → (B, T, d)
│       ├── ln2        RMSNorm    weight (d)
│       └── ffn        SwiGLU: w1 (f, d), w3 (f, d), w2 (d, f)          (B, T, d) → (B, T, f) → (B, T, d)
├── ln_final           RMSNorm    weight (d)
└── lm_head            Linear     weight (V, d)      → logits (B, T, V)
```

Block math: `z = x + attn(ln1(x))`, `y = z + ffn(ln2(z))`. No biases anywhere; embeddings and `lm_head` are **not** tied.

Parameters: per block `4d² + 3·d·f + 2d`; plus `2·V·d` (embeddings + lm_head) `+ d` (ln_final).

| Size | d_model | d_ff | layers | heads | d_k | Params/block | Blocks total | Emb + lm_head | **Total** | fp32 training state |
|---|---|---|---|---|---|---|---|---|---|---|
| small | 768 | 3072 | 12 | 12 | 64 | 9.44M | 113.3M | 15.4M | **128.6M** | ~2.1 GB |
| medium | 1024 | 4096 | 24 | 16 | 64 | 16.78M | 402.7M | 20.5M | **423.2M** | ~6.8 GB |
| large | 1280 | 5120 | 36 | 20 | 64 | 26.22M | 943.8M | 25.6M | **969.4M** | ~15.5 GB |
| xl | 2560 | 10240 | 32 | 32 | 80 | 104.86M | 3,355.6M | 51.2M | **3,406.8M** | ~54.5 GB |
| 10B | 4608 | 12288 | 50 | 36 | 128 | 254.81M | 12,740.7M | 92.2M | **12,832.8M** | ~205 GB |

### 2.3 Which kernels each module launches (what you'll see in Nsight)

Your code is written op-by-op in PyTorch, so almost every line becomes its own GPU kernel:

| Module | Ops → kernels (forward) |
|---|---|
| `Linear` | `einsum` → `aten::bmm` → one GEMM (`ampere_sgemm_*` / `cutlass…sgemm` in fp32) |
| `Embedding` | indexing `weight[x]` → gather kernel. Backward = `index_put` (scatter-add), notably slow on the first step |
| `RMSNorm` | cast to fp32, square, mean, add eps, sqrt, divide, multiply by gain, cast back — ~6 small elementwise/reduction kernels |
| attention mask | **rebuilt every forward, in every layer**: `ones(T, T)`, `triu`, `== 0` |
| RoPE (×2: Q and K) | strided slices, two table gathers, 4 multiplies + add/sub, `stack`, `flatten` — ~10 small kernels each |
| `scaled_dot_product_attention` | `einsum` Q·Kᵀ (GEMM) → `/√d_k` → `torch.where(mask, …, -inf)` → softmax (`amax`, subtract, `exp`, `sum`, divide) → `einsum` with V (GEMM). Every step after the first GEMM produces a full **(B, h, T, T)** tensor |
| `SwiGLU` | 3 GEMMs (w1, w3, w2) + SiLU (`sigmoid`, multiply) + gating multiply |
| loss (`cross_entropy_loss`) | `amax`, subtract, `gather`, `exp`, `sum`, `log`, add, `mean` over (B, T, V) |
| `AdamW.step` | per parameter tensor, ~10 elementwise kernels (`m`, `v` updates, bias-corrected step, weight decay) — thousands of tiny launches per step |

Backward launches roughly 2× the forward's GEMM work (gradients w.r.t. both inputs and weights) plus the backward
of every elementwise op.

---

## 3. The benchmark

### 3.1 Functions in `benchmark.py`

| Function | Signature → returns | What it does |
|---|---|---|
| `default_device()` | `-> str` | `"cuda"` if available, else `"mps"`, else `"cpu"` |
| `synchronize(device)` | `(str) -> None` | Blocks until every queued GPU kernel has finished (`torch.cuda.synchronize` / `torch.mps.synchronize`; no-op on CPU) |
| `build_model(size, context_length, device)` | `(ModelSize, int, str) -> TransformerLM` | Random-init model, allocated directly on `device` |
| `random_batch(batch_size, context_length, device)` | `(int, int, str) -> tuple[Tensor, Tensor]` | `x`, `y`: each **(B, T) int64** uniform in `[0, VOCAB_SIZE)`. `y` is independent random targets (fine for speed; meaningless for loss) |
| `run_step(model, optimizer, x, y, mode, device)` | `-> dict[str, float]` | One step. Returns **seconds per phase that ran**: `{"forward"}` (forward), `{"forward", "backward"}` (forward_backward), `{"forward", "backward", "optimizer"}` (train) |
| `mean_std_ms(values)` | `(list[float]) -> tuple[float, float]` | `(mean_ms, std_ms)`; std is `0.0` for a single value |
| `snapshot_path(base, size_name, multiple_sizes)` | `(str, str, bool) -> str` | Snapshot file for one config: `base` itself, or `<stem>_<size><suffix>` when several sizes run (`*` in overridden names becomes `_custom`). Creates the parent folder. |
| `dump_memory_snapshot(path)` | `(str) -> None` | `torch.cuda.memory._dump_snapshot(path)` and prints where it went |
| `benchmark(size, context_length, batch_size, mode, warmup, steps, device, memory_snapshot=None)` | `-> dict` | Full loop for one config (§3.3); returns a result dict (§3.4). `memory_snapshot` is a file path or `None` |
| `format_table(results)` | `(list[dict]) -> str` | Markdown table, one row per result |
| `main()` | `-> None` | Parses flags, resolves sizes/overrides, runs `benchmark` for each size, prints progress lines then the table |

### 3.2 One step (`run_step`)

```
if mode != "forward": model.zero_grad(set_to_none=True)      # not timed; grads become None (freed)
synchronize                                                  # start from an idle GPU
t0
with torch.set_grad_enabled(mode != "forward"):              # forward mode = no_grad (inference)
    loss = cross_entropy_loss(model(x), y)                   # logits (B, T, V) → loss: 0-dim float32 tensor
synchronize; t1                                              # times["forward"] = t1 - t0
if mode == "forward": return
loss.backward()                                              # fills p.grad for every parameter (same shapes as params)
synchronize; t2                                              # times["backward"] = t2 - t1
if mode == "forward_backward": return
optimizer.step()                                             # your AdamW; updates weights, m, v
synchronize; t3                                              # times["optimizer"] = t3 - t2
```

Tensor shapes during a step:

| Tensor | Shape | dtype | Size for small, B=4, T=512 (fp32) |
|---|---|---|---|
| `x`, `y` | (B, T) | int64 | 16 KB each |
| hidden state between blocks | (B, T, d) | float32 | 6.3 MB |
| Q, K, V after rearrange | (B, h, T, d_k) | float32 | 6.3 MB each |
| attention scores / masked / softmax | (B, h, T, T) | float32 | **50.3 MB each, per layer** |
| FFN inner activations | (B, T, f) | float32 | 25.2 MB each |
| logits | (B, T, V) | float32 | 81.9 MB |
| loss | () | float32 | scalar |

### 3.3 The full loop (`benchmark`)

1. `torch.cuda.reset_peak_memory_stats()` (CUDA).
2. *(snapshot only)* `torch.cuda.memory._record_memory_history(max_entries=1_000_000)` — **before** the model exists,
   so weight allocations are recorded.
3. `build_model(...)`; `AdamW(model.parameters())` with default hyperparameters **only in `train` mode** (else `None`).
4. One `random_batch(...)` — the **same batch is reused** every step.
5. `warmup` × `run_step(...)` — results discarded.
6. `steps` × `run_step(...)` — collected as `list[dict[str, float]]`.
7. Per phase present: `mean_std_ms`; whole step: `mean_std_ms` of `sum(times.values())` per step.
8. `torch.cuda.max_memory_allocated()` → `peak_mem_gb` (covers build, warm-up and measured steps).
9. *(snapshot only)* `dump_memory_snapshot(path)`.
10. `except torch.OutOfMemoryError` → `{"oom": True}`, and *(snapshot only)* dump **before cleanup**, capturing what
    filled the GPU at the moment of OOM.
11. `finally` → *(snapshot only)* stop recording (`_record_memory_history(enabled=None)`); delete model/optimizer,
    `gc.collect()`, `torch.cuda.empty_cache()` so the next size starts clean.

### 3.4 Result dict (one per config)

| Key | Type | When present |
|---|---|---|
| `size` | `str` | always (`"small"`, or `"small*"` if overridden) |
| `params` | `int` | always (analytic) |
| `context_length` | `int` | always |
| `mode` | `str` | always |
| `forward` | `tuple[float, float]` (mean ms, std ms) | if it ran |
| `backward` | `tuple[float, float]` | `forward_backward`, `train` |
| `optimizer` | `tuple[float, float]` | `train` |
| `step` | `tuple[float, float]` | if it ran |
| `peak_mem_gb` | `float` | CUDA only |
| `memory_snapshot` | `str` (file path) | only with `--memory-snapshot` (also set when the config OOMs — the snapshot is still written) |
| `oom` | `bool` (`True`) | only if that config ran out of memory (then no timing keys) |

`main()` collects these into a `list[dict]` and prints `format_table(results)`.

---

## 4. Flags

### What to run

| Flag | Default | What it does |
|---|---|---|
| `--size {small,medium,large,xl,10B,all}` | `small` | Table 1 dimensions. `all` runs every size in order and prints one combined table. |
| `--mode {forward,forward_backward,train}` | `train` | Which phases each step runs (§3.2). |
| `--context-length N` | `512` | Tokens per sequence (T). Also sizes RoPE's tables. |
| `--batch-size N` | `4` | Sequences per batch (B). |

### How to measure

| Flag | Default | What it does |
|---|---|---|
| `--warmup N` | `5` | Untimed steps first. Early steps pay one-time costs (CUDA context, allocator growth, cuBLAS kernel selection, library loading). |
| `--steps N` | `10` | Timed steps; results are mean ± std over these. |
| `--device DEV` | `cuda` → `mps` → `cpu` | Where to run. Peak memory is reported on CUDA only. |

### Memory profiling

| Flag | Default | What it does |
|---|---|---|
| `--memory-snapshot PATH` | off | Records every GPU allocation/free (with stack traces) from **before the model is built** to the end of the run, then writes a `.pickle` for pytorch.org/memory_viz (§8). With `--size all`, writes one file per size: `PATH_<size>.pickle`. Also written if the config OOMs. **CUDA only** (errors otherwise). Timings from these runs include recording overhead — don't compare them with plain runs. Use few steps (e.g. `--warmup 1 --steps 2`). |

### Override a size's dimensions

Applied to every selected size; rows are labelled with `*` (e.g. `small*`).

| Flag | Overrides | Constraint |
|---|---|---|
| `--d-model N` | d | divisible by `num_heads`; d_k = d/h should be even (RoPE pairs dimensions) |
| `--d-ff N` | f | ≈ 8/3·d rounded to a multiple of 64 is standard |
| `--num-layers N` | number of blocks | |
| `--num-heads N` | h | must divide d |

### Planned (not implemented yet)

Each will default to the current baseline so existing commands keep their meaning. Move rows up when implemented.

| Flag | Topic | Intended behaviour |
|---|---|---|
| `--attention {naive,flash_pytorch,flash_triton}` | FlashAttention-2 | Which attention implementation the model calls |
| `--dtype {fp32,bf16,fp16}` | Mixed precision | Forward/backward under `torch.autocast`; matmuls on Tensor Cores |
| `--compile` | Kernel fusion | Wrap the model in `torch.compile` |
| `--checkpoint {none,block,…}` | Gradient checkpointing | Save fewer activations, recompute in backward |
| (NVTX ranges) | Annotation | `warmup` / `measured` / `forward` / `backward` / `optimizer` ranges for `--nvtx-capture` |

---

## 5. How settings change the profiles

What to expect in **Nsight Systems** (timeline/kernels) and in the **memory profiler** (§8) when you change a setting.

| Setting | Nsight Systems | Memory |
|---|---|---|
| `--size` (bigger) | Same kernel pattern repeated `num_layers` times per pass; GEMMs get wider (longer blocks), elementwise kernels scale with d and f. Bigger models are more GPU-bound (fewer gaps). | Weights/grads/optimizer state scale with params (§2.2); per-layer activations scale with d, f and layer count. |
| `--context-length` ↑ | GEMMs scale ~linearly in T; **attention scores, mask, softmax and their backward scale with T²** and come to dominate attention time. The mask (`T×T`) is rebuilt every layer. | Activations with T for most tensors, **T² for each (B, h, T, T) attention tensor** — the usual cause of OOM at long context. |
| `--batch-size` ↑ | Same kernels, each longer; fewer relative gaps (better GPU utilization). Number of launches unchanged. | Activations scale linearly with B; weights/optimizer state unchanged. |
| `--mode forward` | Only forward kernels; `pt_autograd_0` stays idle; no `AdamW.step`. | `no_grad`: activations freed as soon as used → tiny peak (small: 0.81 GB vs 6.03 GB in train). |
| `--mode forward_backward` | Adds the backward pass, launched from the `pt_autograd_0` thread, ~2× forward time. | Activations held until backward consumes them; gradients allocated (one per parameter). |
| `--mode train` | Adds `AdamW.step`: thousands of tiny elementwise launches with gaps (launch-bound). | Adds `m` and `v` (2× params) — allocated **lazily on the first step**, so step 1 shows a jump. Your AdamW builds new tensors each step (`data = data - …`, `p.data = data`) rather than updating in place, adding temporary allocations per parameter. |
| `--warmup 0` | The first step appears much wider: CUDA context setup, cuBLAS heuristics, allocator growth, `IndexBackward` on embeddings especially slow. | First step shows the allocator reserving memory; the optimizer-state jump lands in the measured window. |
| `--steps` ↑ | Longer report, more repeats of the same step pattern. Use `--nvtx-capture` (once ranges exist) to keep reports small. | More repeats of the same sawtooth; peak unchanged. |
| `--device mps/cpu` | Nsight not applicable. | Peak memory not reported; `--memory-snapshot` refused. |
| `--memory-snapshot` | Extra CPU time per allocation (stack-trace capture) → wider CPU ranges, more GPU gaps; don't profile and snapshot in the same run. | Produces the snapshot (§8). Recording itself uses host RAM, not GPU memory; peak GPU memory is unchanged. |
| Profiling itself | Adds overhead per CUDA call and NVTX range: small train step 174.60 → 206.70 ms; optimizer 21.9 → 36.6 ms. Launch-heavy phases distort most. | Snapshot recording adds CPU overhead; doesn't change what's allocated. |
| Per-phase `synchronize` (built in) | Short idle GPU gaps between forward/backward/optimizer, plus a long `cudaDeviceSynchronize` when the CPU was ahead. Real training doesn't have them. | None. |

**Planned settings — expected effects** (update with real measurements when implemented):

| Setting | Nsight Systems | Memory |
|---|---|---|
| FlashAttention-2 | Mask/softmax/scores kernels replaced by one fused attention kernel per layer (forward and backward); far fewer launches in `attn`. | No (B, h, T, T) tensors stored → attention memory linear in T; much longer contexts fit. |
| bf16 / fp16 autocast | GEMM kernels switch from `sgemm` (CUDA cores) to Tensor Core kernels and get much shorter; casting kernels appear; elementwise work becomes a larger fraction. | Activations roughly halve; weights and AdamW state stay fp32. |
| `torch.compile` | Chains of elementwise ops (RMSNorm, RoPE, SiLU, softmax) fused into Triton kernels → far fewer, larger kernels; first step includes compile time. | Fewer intermediate tensors saved. |
| Gradient checkpointing | Forward kernels re-run inside the backward pass (backward gets longer). | Activations stored only at checkpoints → lower peak, traded for recompute time. |
| NVTX ranges | New labelled rows (your domain) and projected ranges over the GPU; enables `--capture-range=nvtx`. | None. |

---

## 6. Output & baseline results

Printed: one progress line per size (`params`, fp32 training-state estimate), then:

| Column | Meaning |
|---|---|
| `params` | Analytic parameter count |
| `forward / backward / optimizer (ms)` | Mean ± std per phase; `-` if the mode doesn't run it |
| `step (ms)` | Mean ± std of the whole step |
| `peak mem (GB)` | `torch.cuda.max_memory_allocated()` for that config |
| `OOM` | That config ran out of GPU memory |

**RTX 3090 (24 GB), fp32, ctx 512, batch 4, warmup 5, steps 10** — numbers only compare on the same GPU.

| Size | Mode | Forward (ms) | Backward (ms) | Optimizer (ms) | Step (ms) | Peak mem |
|---|---|---|---|---|---|---|
| small | train | 47.21 ± 0.25 | 105.48 ± 0.23 | 21.91 ± 0.19 | 174.60 ± 0.50 | 6.03 GB |
| medium | train | 143.97 ± 2.04 | 305.71 ± 0.36 | 68.80 ± 0.24 | 518.48 ± 2.42 | 16.37 GB |
| large / xl / 10B | train | OOM | | | | |
| small | forward | 48.49 ± 0.12 | | | | 0.81 GB |
| medium | forward | 147.01 ± 1.25 | | | | 2.08 GB |
| large | forward | 297.50 ± 1.87 | | | | 4.48 GB |
| xl | forward | 891.31 ± 2.77 | | | | 14.55 GB |
| 10B | forward | OOM | | | | |

---

## 7. Profiling with Nsight Systems

### 7.1 Recipes (run on the GPU machine)

```bash
# lean: CUDA + your NVTX ranges only — use for numbers you report
nsys profile -o profiles/NAME --force-overwrite true --trace=cuda,nvtx \
  uv run python -m cs336_systems.benchmark --size small --mode train

# exploration: PyTorch's automatic labels for every module and op
nsys profile -o profiles/NAME --force-overwrite true --trace=cuda,nvtx,osrt \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  uv run python -m cs336_systems.benchmark --size small --mode train

# record only inside your NVTX range "measured" (skips warm-up; needs the range to exist)
nsys profile --capture-range=nvtx --nvtx-capture=measured --trace=cuda,nvtx -o profiles/NAME \
  uv run python -m cs336_systems.benchmark --size small

# text summaries
nsys stats --report cuda_gpu_kern_sum profiles/NAME.nsys-rep   # GPU time per kernel
nsys stats --report nvtx_sum          profiles/NAME.nsys-rep   # time per NVTX range
nsys stats --report cuda_api_sum      profiles/NAME.nsys-rep   # CPU-side CUDA calls

# to the Mac (run on the Mac), then open in the Nsight Systems app
scp -P PORT -i ~/.ssh/id_ed25519 root@HOST:/workspace/assignment1-basics/profiles/NAME.nsys-rep profiles/
open profiles/NAME.nsys-rep
```

`profiles/`, `*.nsys-rep`, `*.sqlite` are git-ignored.

### 7.2 Profiling options

| Option | Adds | Use when |
|---|---|---|
| `--trace=cuda` | Every GPU kernel / memcpy / memset (CUDA HW rows) and CPU-side CUDA calls (`cudaLaunchKernel`, `cudaDeviceSynchronize`) | Always |
| `--trace=nvtx` | Your `torch.cuda.nvtx` ranges | Always |
| `--trace=osrt` | OS runtime calls (`pthread_cond_wait`, sleeps, file I/O) | Explaining CPU-side gaps |
| `--trace=cublas,cudnn` | Library calls | Rarely |
| `--pytorch=functions-trace` | Range per module forward (`PyTorch:TransformerLM.layers.0.attn`) plus `Step`, `Forward`, `AdamW.step` | Mapping kernels to modules |
| `--pytorch=autograd-shapes-nvtx` | Range per op and backward function (`aten::bmm`, `…BmmBackward0`) **with shapes** | Identifying ops, esp. backward |
| `--pytorch=autograd-nvtx` | Same without shapes (per nsys docs; cheaper; untested here) | |
| `--capture-range=nvtx --nvtx-capture=NAME` | Records only inside range `NAME` | Skipping warm-up |
| `--capture-range=cudaProfilerApi` | Records only between `torch.cuda.profiler.start()` / `stop()` | Same, from code |
| In code: `torch.autograd.profiler.emit_nvtx(record_shapes=True)` | Same op labels as `--pytorch=autograd-shapes-nvtx`, for a block of code | When the profiler UI lacks the PyTorch option |

**Blocked on Runpod containers** (no hardware-counter permissions): `--gpu-metrics-devices` (ERR_NVGPUCTRPERM),
CPU sampling / context switches (`perf_event_open` fails), and very likely Nsight Compute (`ncu`).

### 7.3 Reading the timeline

```
CUDA HW (GPU)                      ← what actually ran on the GPU
  [All Streams]
    Kernels                        ← one block per kernel; hover for name/duration
    Memory                         ← memcpy / memset
    NVTX / PyTorch (projected)     ← CPU ranges re-drawn over the kernels they launched
Threads                            ← every row below is a CPU thread
  [pid] python                     ← main thread: launches forward + optimizer
    OS runtime libraries
    PyTorch                        ← --pytorch=functions-trace labels
    NVTX                           ← your ranges + --pytorch op labels
    CUDA API                       ← cudaLaunchKernel, cudaDeviceSynchronize, …
  [pid] pt_autograd_0              ← autograd worker: launches the BACKWARD pass (main thread waits in pthread_cond_wait)
```

- **CPU rows show launching; only CUDA HW shows execution.** A CPU range ends when its kernels are *queued*.
- **Projection follows the launching thread.** The main thread's `Step` only claims forward + optimizer kernels;
  backward kernels sit under `pt_autograd_0`'s op labels (`…Backward0`). Backward has no module labels.
- **Patterns:** solid back-to-back GPU blocks → GPU-bound (faster kernels help: mixed precision, fusion,
  FlashAttention). Thin blocks with gaps while the CPU launches → launch-bound (fewer, bigger kernels help).
  A long `cudaDeviceSynchronize` → the CPU was ahead, waiting on the GPU.
- The CPU runs slightly ahead of the GPU: the GPU work directly above a CPU range is usually *earlier* work.

### 7.4 NVTX domains (the separate NVTX rows)

| Row / domain | Comes from |
|---|---|
| `NVTX` / `[Default]` | Your `torch.cuda.nvtx.range(...)` (and `--pytorch` op labels) |
| `PyTorch` | `--pytorch=functions-trace` module/step labels |
| `CCCL` (e.g. `cub::DeviceRadixSort`) | NVIDIA's CUDA C++ libraries inside PyTorch — not your code |

How NVTX reaches Nsight: NVTX calls are no-op stubs until `nsys` sets `NVTX_INJECTION64_PATH`, which loads its
library; each push/pop is recorded with a CPU timestamp and thread id. Kernels are attached to ranges through
CUPTI correlation ids (launch call ↔ kernel) on the same thread.

### 7.5 Useful views

- **Zoom:** drag across the timeline → right-click → *Zoom into selection* (or Cmd/Ctrl + scroll).
- **Stats for one range:** right-click a range → filter to selection → **Stats System View → CUDA GPU Kernel Summary**.
- **CPU → GPU link:** click a `cudaLaunchKernel` in a CUDA API row; its kernel is highlighted in CUDA HW.

### 7.6 Writing NVTX ranges

```python
import torch.cuda.nvtx as nvtx
with nvtx.range("forward"): ...                       # context manager
@nvtx.range("attention")                              # decorator
def f(...): ...
nvtx.range_push("measured"); ...; nvtx.range_pop()    # manual (same thread)
```

Near-zero cost without a profiler. Gotcha: `layers.py` does `from .functional import scaled_dot_product_attention`,
so replacing `functional.scaled_dot_product_attention` from outside does **not** reach the attention layer.

---

## 8. Memory profiling

### 8.1 Running it

```bash
# on the GPU machine
uv run python -m cs336_systems.benchmark --size small --mode train --warmup 1 --steps 2 \
  --memory-snapshot profiles/mem_small_train.pickle

# several sizes → profiles/mem_small.pickle, profiles/mem_medium.pickle, ...
uv run python -m cs336_systems.benchmark --size all --mode forward --warmup 1 --steps 1 \
  --memory-snapshot profiles/mem.pickle

# to the Mac (run on the Mac), then drag the file onto https://pytorch.org/memory_viz
scp -P PORT -i ~/.ssh/id_ed25519 root@HOST:/workspace/assignment1-basics/profiles/mem_small_train.pickle profiles/
```

`*.pickle` is git-ignored. memory_viz runs entirely in your browser; the file is not uploaded anywhere.

### 8.2 How it works (and how it differs from NVTX / Nsight)

- It hooks **PyTorch's caching allocator**, not NVTX. Every tensor allocation and free is logged with its size,
  address, CUDA stream and a **stack trace** (Python + C++ frames). Hovering a block in memory_viz shows that stack,
  e.g. `layers.py:172 forward` → `rope.py … stack` — your file and line, not an NVTX label.
- Your NVTX ranges do **not** appear. Identify phases by stack: model files = forward, `torch/autograd/…` frames =
  backward, `optim.py` = your AdamW.
- The timeline's x-axis is **allocation events, not seconds**: a burst of tiny allocations looks wide; a slow GEMM
  that allocates nothing looks like nothing happened. It cannot be lined up with an Nsight timeline.
- The snapshot also contains allocator **segments** (memory reserved from CUDA, including cached-but-unused blocks),
  which explains `nvidia-smi` showing more memory in use than `max_memory_allocated`.
- Recording is capped at 1,000,000 events (`MEMORY_HISTORY_MAX_ENTRIES`); older events drop off beyond that.

### 8.3 Views in memory_viz

| View | Shows | Use it for |
|---|---|---|
| Active Memory Timeline | Every live tensor as a band (height = size, width = lifetime) | The handout's "active memory timeline" images; seeing the step pattern |
| Allocator State History | Segments reserved from CUDA and how tensors are packed into them | Fragmentation, reserved vs allocated |
| Active Cached Segment Timeline | Reserved segments over time | Why reserved memory stays high |
| Detail slider | Hides the smallest allocations | Finding the largest allocations (handout memory question) |

### 8.4 What a recording from the very beginning contains

Expected shape of a `train` run (fp32):

| Phase | What you'll see |
|---|---|
| Model build | Base rises to the weights (4 bytes × params; small ≈ 0.51 GB), plus RoPE tables per layer. Recording starts before this, so it's visible |
| Random batch | Two tiny (B, T) int64 tensors |
| Forward | Staircase up: activations saved for backward, layer by layer; attention's (B, h, T, T) tensors are the tallest steps at long T |
| Backward | Activations freed in reverse layer order while gradients (another 4 bytes × params) appear |
| Optimizer | First step only: `m` and `v` appear (8 bytes × params). Every step: short spikes from AdamW's out-of-place updates |
| Next step | `zero_grad(set_to_none=True)` frees gradients; the sawtooth repeats |
| OOM | The file ends at the failing allocation: the top of the stack is what filled the GPU |

By mode: `forward` (no_grad) shows small short-lived bumps per layer and no gradients; `forward_backward` shows the
staircase and gradients but no `m`/`v`; `train` shows everything above. Longer `--context-length` makes the
(B, h, T, T) attention bands dominate (they grow with T²).

The handout's xl memory questions need ~54.5 GB of training state alone — too big for 24 GB; use small/medium
(same patterns) or a larger GPU.

---

## 9. Nsight app over SSH

*Manage targets → SSH*: host, port, user `root` from the pod's `ssh.direct` block. The key has a passphrase:
run `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` first. Run settings: command
`/workspace/assignment1-basics/.venv/bin/python -m cs336_systems.benchmark …`, working directory
`/workspace/assignment1-basics`; enable CUDA, NVTX, OS runtime (+ PyTorch tracing if offered); disable CPU sampling,
context switches, GPU metrics. First connect uploads ~350 MB of profiler files (minutes; again after a pod reset).

Version note: Mac app 2026.5.1 > pod `nsys` 2026.3.2. Pod reports open on the Mac; to summarize app-made reports
on the pod use the app's copy: `/root/.local/share/nsight_systems/nsys stats …`.

---

## 10. Pod workflow

```bash
git add -A && git commit -m "..." && git push            # Mac
cd /workspace/assignment1-basics && git pull             # pod (fast-forward only; the pod cannot push)
/workspace/pod_setup.sh                                  # pod, once after any restart (reinstalls nsys)
```

The SSH port changes on every pod start. Stop (don't terminate) the pod when idle; `/workspace` survives a stop.

---

## 11. Changelog

Benchmark-affecting changes, newest first. Add an entry for every change that can alter timings, kernels, memory
or profiles.

| Date | Change | Effect on benchmarks / profiles |
|---|---|---|
| 2026-10-14 | `--memory-snapshot PATH` added (records from before model build; dumps on success and on OOM; one file per size); `*.pickle` git-ignored | No effect on plain runs. Snapshot runs carry CPU-side recording overhead — timings not comparable. Not yet run on a GPU (pod stopped) |
| 2026-10-14 | README expanded (model structure, loop internals, return types, settings → profile effects) | Docs only |
| 2026-10-13 | `benchmark.py`, `model_configs.py` added (fp32, per-phase sync, naive attention) | Baseline; 3090 results in §6 |
