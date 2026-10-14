# cs336_systems — benchmarking & profiling

<!-- CO AUTHORED BY CLAUDE.
Maintenance rule (see AGENTS.md): any change that can affect benchmarking — a new flag, a different way to run part of
the model (FlashAttention-2, torch.compile, mixed precision, checkpointing), new NVTX ranges, profiler options —
must update this file in the same change: the relevant section, §5 "How settings change the profiles", and §13 Changelog. -->

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
9. [HBM traffic per op (`--op-traffic`)](#9-hbm-traffic-per-op---op-traffic)
10. [Saved tensors & activation checkpointing (`--saved-tensors`, `--checkpoint`)](#10-saved-tensors--activation-checkpointing---saved-tensors---checkpoint)
11. [Nsight app over SSH](#11-nsight-app-over-ssh)
12. [Pod workflow](#12-pod-workflow)
13. [Changelog](#13-changelog)

---

## 1. Files & quick start

| File | What it is |
|---|---|
| `benchmark.py` | End-to-end benchmark. Builds a model, runs warm-up + timed steps, times each phase, reports peak memory; optionally writes a memory snapshot. |
| `model_configs.py` | Table 1 model sizes, shared defaults, analytic parameter count and memory estimate. |
| `op_traffic.py` | `OpTrafficMode` (a `TorchDispatchMode`) that estimates bytes read from / written to HBM per op, plus report/JSON helpers (§9). |
| `activation_memory.py` | Activation checkpointing (`apply_checkpointing`) and saved-for-backward measurement (`SavedTensorsTracker`, `measure_saved_tensors`, report) (§10). |
| `README.md` | This file. |

### Implementation status

| Feature | Status | How | Section |
|---|---|---|---|
| End-to-end timing (forward / backward / optimizer, warm-up) | ✅ implemented | `--mode`, `--warmup`, `--steps` | §3, §4 |
| Model sizes (Table 1) + dimension overrides | ✅ implemented | `--size`, `--d-model` … | §2, §4 |
| Peak GPU memory | ✅ implemented | always (CUDA) | §6 |
| Nsight Systems profiling (time, CPU/GPU threads) | ✅ documented (no code needed) | `nsys profile …` | §7 |
| Mixed precision (autocast bf16/fp16, GradScaler) | ✅ implemented | `--dtype` | §3.2, §4, §6 |
| Memory snapshot for memory_viz | ✅ implemented | `--memory-snapshot` | §8 |
| Nsight memory tracking (per-tensor) | ✅ implemented | `--no-allocator-cache` + `nsys --cuda-memory-usage=true` | §7.7 |
| HBM traffic estimate per op | ✅ implemented | `--op-traffic`, `--op-traffic-json` | §9 |
| Saved-for-backward measurement per block / attention / FFN | ✅ implemented | `--saved-tensors` | §10 |
| Activation (gradient) checkpointing | ✅ implemented | `--checkpoint`, `--checkpoint-group` | §10 |
| Your own NVTX ranges (warm-up / measured / phases / attention stages) | ⏳ not yet — **you're writing these** | `torch.cuda.nvtx` | §7.6 |
| Kernel fusion with `torch.compile` | ⏳ not yet | planned `--compile` | §4 Planned |
| FlashAttention-2 (PyTorch, Triton) | ⏳ not yet | planned `--attention` | §4 Planned |

Run everything from the **repo root**:

```bash
uv run python -m cs336_systems.benchmark --size small                 # one full training step, small model
uv run python -m cs336_systems.benchmark --size all --mode forward    # inference, every size
uv run python -m cs336_systems.benchmark --size small --warmup 1 --steps 2 \
  --memory-snapshot profiles/mem_small_train.pickle                   # memory snapshot (CUDA only, §8)
uv run python -m cs336_systems.benchmark --size all --dtype bf16      # mixed precision (§4, §6)
uv run python -m cs336_systems.benchmark --size small --op-traffic    # per-op HBM traffic estimate (§9)
uv run python -m cs336_systems.benchmark --size small --saved-tensors # memory kept for backward, per block (§10)
uv run python -m cs336_systems.benchmark --size small --checkpoint block --saved-tensors   # checkpointing (§10)
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
| `autocast_context(device, dtype)` | `(str, str) -> context manager` | `torch.autocast(device_type, dtype=bfloat16/float16)` for `bf16`/`fp16`; `contextlib.nullcontext()` for `fp32` |
| `run_step(model, optimizer, x, y, mode, device, dtype="fp32", scaler=None, tracer=None)` | `-> dict[str, float]` | One step. Returns **seconds per phase that ran**: `{"forward"}` (forward), `{"forward", "backward"}` (forward_backward), `{"forward", "backward", "optimizer"}` (train) |
| `mean_std_ms(values)` | `(list[float]) -> tuple[float, float]` | `(mean_ms, std_ms)`; std is `0.0` for a single value |
| `snapshot_path(base, size_name, multiple_sizes)` | `(str, str, bool) -> str` | Snapshot file for one config: `base` itself, or `<stem>_<size><suffix>` when several sizes run (`*` in overridden names becomes `_custom`). Creates the parent folder. |
| `dump_memory_snapshot(path)` | `(str) -> None` | `torch.cuda.memory._dump_snapshot(path)` and prints where it went |
| `benchmark(size, context_length, batch_size, mode, warmup, steps, device, memory_snapshot=None, dtype="fp32", op_traffic=False, op_traffic_top=15, checkpoint="none", checkpoint_group=1, saved_tensors=False, saved_tensors_top=10)` | `-> dict` | Full loop for one config (§3.3); returns a result dict (§3.4). `memory_snapshot` is a file path or `None`; `dtype` is a key of `DTYPES`; `op_traffic` adds one traced, untimed step (§9); `checkpoint`/`checkpoint_group` apply activation checkpointing to every step and `saved_tensors` adds one measured, untimed forward (§10) |
| `format_table(results)` | `(list[dict]) -> str` | Markdown table, one row per result |
| `main()` | `-> None` | Parses flags; with `--no-allocator-cache` sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1` before any CUDA allocation; resolves sizes/overrides, runs `benchmark` for each size, prints progress lines then the table |

When `tracer` (an `OpTrafficMode`) is passed, `run_step` sets `tracer.phase` to `"forward"`, `"backward"`,
`"optimizer"` as it enters each phase so every op is labelled.

Functions from other modules used by `benchmark()`: `apply_checkpointing`, `measure_saved_tensors`,
`format_saved_report` (`activation_memory.py`, §10.4); `OpTrafficMode`, `format_report`, `write_json`
(`op_traffic.py`, §9).

Module constants: `MODES`, `PHASES`, `MEMORY_HISTORY_MAX_ENTRIES = 1_000_000`,
`DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}`.

### 3.2 One step (`run_step`)

```
if mode != "forward": model.zero_grad(set_to_none=True)      # not timed; grads become None (freed)
synchronize                                                  # start from an idle GPU
t0
with torch.set_grad_enabled(mode != "forward"), autocast_context(device, dtype):   # no_grad in forward mode
    loss = cross_entropy_loss(model(x), y)                   # logits (B, T, V) → loss: 0-dim tensor
synchronize; t1                                              # times["forward"] = t1 - t0
if mode == "forward": return
scaler.scale(loss).backward() if fp16 else loss.backward()   # fills p.grad (fp32, same shapes as params)
synchronize; t2                                              # times["backward"] = t2 - t1
if mode == "forward_backward": return
scaler.step(optimizer); scaler.update() if fp16 else optimizer.step()   # your AdamW; updates weights, m, v
synchronize; t3                                              # times["optimizer"] = t3 - t2
```

Only forward + loss run inside autocast (PyTorch's recommended pattern): backward ops automatically run in the
dtype their forward op used. Under autocast, matmuls (`einsum` → `bmm`) run in bf16/fp16, ops on autocast's
"fp32 list" (e.g. `exp`, `log`, `sum`, `softmax`-like reductions) are promoted to fp32, and your RMSNorm explicitly
upcasts to fp32 internally. Parameters, gradients and AdamW state remain **fp32**.

Tensor shapes during a step (fp32 sizes; under bf16/fp16 most activations are half these):

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
3. `build_model(...)`; `apply_checkpointing(model, checkpoint, checkpoint_group)` (no-op for `none`; otherwise wraps
   block / attn / ffn forwards with `torch.utils.checkpoint`, §10); `AdamW(model.parameters())` with default
   hyperparameters **only in `train` mode** (else `None`).
4. One `random_batch(...)` — the **same batch is reused** every step.
   *(fp16 with backward only)* `torch.amp.GradScaler(device_type)` — scales the loss so fp16 gradients don't
   underflow; bf16 has fp32's exponent range and doesn't need one.
5. `warmup` × `run_step(...)` — results discarded.
6. `steps` × `run_step(...)` — collected as `list[dict[str, float]]`.
7. Per phase present: `mean_std_ms`; whole step: `mean_std_ms` of `sum(times.values())` per step.
8. `torch.cuda.max_memory_allocated()` → `peak_mem_gb` (covers build, warm-up and measured steps). Skipped with
   `--no-allocator-cache`, where PyTorch's counters stay at 0.
9. *(snapshot only)* `dump_memory_snapshot(path)`.
   *(op traffic only)* one **extra, untimed** `run_step(..., tracer=OpTrafficMode(model))` → `result["op_traffic_rows"]`,
   report printed (§9). It runs after timing and peak memory were recorded, so neither is affected. In `train` mode it
   performs one more optimizer update (harmless: weights are random).
   *(saved tensors only)* one **extra, untimed forward pass** (grad enabled, same autocast) inside `SavedTensorsTracker`
   → `result["saved_tensors"]`, report printed (§10). Measured after everything above.
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
| `dtype` | `str` (`"fp32"`, `"bf16"`, `"fp16"`) | always |
| `checkpoint` | `str` (`"none"`, `"block"`, `"block xN"`, `"attn"`, `"ffn"`) | always |
| `forward` | `tuple[float, float]` (mean ms, std ms) | if it ran |
| `backward` | `tuple[float, float]` | `forward_backward`, `train` |
| `optimizer` | `tuple[float, float]` | `train` |
| `step` | `tuple[float, float]` | if it ran |
| `peak_mem_gb` | `float` | CUDA only, and not with `--no-allocator-cache` |
| `op_traffic_rows` | `list[dict]` — `{"phase", "module", "op", "calls", "bytes_read", "bytes_written"}` per (phase, module, op) | only with `--op-traffic` / `--op-traffic-json` |
| `saved_tensors` | `dict` — `{"per_save_total", "unique_total", "retained_total", "layers": [...], "other": {...}, "tensors": [...]}` (bytes; schema in §10.4) | only with `--saved-tensors` |
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

### Precision

| Flag | Default | What it does |
|---|---|---|
| `--dtype {fp32,bf16,fp16}` | `fp32` | `fp32`: no autocast (baseline). `bf16`/`fp16`: forward + loss under `torch.autocast` — matmuls on Tensor Cores; weights, grads, AdamW state stay fp32. `fp16` adds a `GradScaler` (loss scaling) whenever backward runs. Shown in the table's `dtype` column. |

### Memory profiling

| Flag | Default | What it does |
|---|---|---|
| `--no-allocator-cache` | off | Disables PyTorch's caching allocator (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) so every tensor is a real `cudaMalloc`/`cudaFree` that `nsys --cuda-memory-usage=true` can see (§7.7). Much slower (small train step ~1.7 s vs 0.17 s); `peak mem` shows `-`. Use only for Nsight memory profiles. |
| `--memory-snapshot PATH` | off | Records every GPU allocation/free (with stack traces) from **before the model is built** to the end of the run, then writes a `.pickle` for pytorch.org/memory_viz (§8). With `--size all`, writes one file per size: `PATH_<size>.pickle`. Also written if the config OOMs. **CUDA only** (errors otherwise). Timings from these runs include recording overhead — don't compare them with plain runs. Use few steps (e.g. `--warmup 1 --steps 2`). |

### HBM traffic estimate

| Flag | Default | What it does |
|---|---|---|
| `--op-traffic` | off | After the measured steps, run **one extra untimed step** under `OpTrafficMode` and print estimated bytes read/written per phase, per forward module, and the top ops (§9). Timings and peak memory are unaffected. Works on any device. |
| `--op-traffic-json PATH` | off | Also save the per-op rows (plus the config) as JSON for before/after comparisons. Implies `--op-traffic`. With `--size all`: `PATH_<size>.json`. |
| `--op-traffic-top N` | `15` | How many ops to list per phase in the printed report. |

### Activation checkpointing & saved tensors

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint {none,block,attn,ffn}` | `none` | Wraps modules with `torch.utils.checkpoint` (`use_reentrant=False`) for **every** step, so timings include recomputation. `block`: each TransformerBlock keeps only its input; `attn` / `ffn`: only that sub-module is checkpointed (§10). Shown in the table's `ckpt` column. |
| `--checkpoint-group N` | `1` | With `--checkpoint block`: checkpoint groups of N consecutive blocks (one saved input per group). |
| `--saved-tensors` | off | After the measured steps, one extra untimed forward pass reporting memory saved for backward: totals, per TransformerBlock (attention / FFN / norms), outside the blocks, and the largest saved tensors (§10). Retained-memory columns need CUDA. |
| `--saved-tensors-top N` | `10` | How many of the largest saved tensors to list. |

### Override a size's dimensions

Applied to every selected size; rows are labelled with `*` (e.g. `small*`).

| Flag | Overrides | Constraint |
|---|---|---|
| `--d-model N` | d | divisible by `num_heads`; d_k = d/h should be even (RoPE pairs dimensions) |
| `--d-ff N` | f | ≈ 8/3·d rounded to a multiple of 64 is standard |
| `--num-layers N` | number of blocks | |
| `--num-heads N` | h | must divide d |

### Planned (not implemented yet)

Everything else in this section is implemented. These will default to the current baseline so existing commands keep
their meaning; move rows up when implemented.

| Flag | Topic | Intended behaviour |
|---|---|---|
| `--attention {naive,flash_pytorch,flash_triton}` | FlashAttention-2 | Which attention implementation the model calls |
| `--compile` | Kernel fusion | Wrap the model in `torch.compile` |
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
| `--dtype bf16` / `fp16` | GEMMs switch from `ampere_sgemm` (CUDA cores) to Tensor Core kernels and get much shorter; cast kernels (`to`/copy) appear; elementwise and optimizer kernels become a bigger share. Measured on the 3090: small train step 172 → 114 ms (forward 46.5 → 29.5, backward 104 → 63), optimizer unchanged (fp32). Speedup grows with size (forward: small 1.6×, xl 2.6×). `fp16` adds GradScaler kernels (inf checks, unscale) to backward/optimizer. | **Training**: activations about halve (small 6.03 → 4.76 GB, medium 16.37 → 13.18 GB). **Forward-only**: peak goes **up** (small 0.81 → 1.05 GB, xl 14.55 → 20.89 GB) — autocast caches a bf16 copy of each weight on top of the fp32 one, and no_grad has few activations to save. Weights, grads, `m`, `v` stay fp32, so large/xl training still doesn't fit. |
| `--no-allocator-cache` | Every tensor becomes a `cudaMalloc`/`cudaFree` (small train step: 3,600 / 3,299 calls vs 154 / 75 with the cache); those calls synchronize, so the timeline is much slower and gappier. Use only to read memory, never timings. | `nsys --cuda-memory-usage=true` now tracks per-tensor usage. PyTorch's own counters (and `--memory-snapshot`) stop being meaningful. |
| `--op-traffic` | None for the measured steps (tracing happens in an extra step afterwards). If you profile with Nsight at the same time, the traced step appears at the end, much slower (Python runs for every op). | None for the reported peak (measured before the traced step). A memory snapshot taken in the same run includes the traced step. |
| `--checkpoint block` | Forward unchanged; **backward re-runs each block's forward first** (the forward kernel pattern appears again inside the backward pass, launched from `pt_autograd_0`). Small: backward 104.8 → 142.7 ms, step 174.0 → 212.4 ms (+22%). | Saved for backward drops to one block input per block: retained after forward 4,150.8 → 326.2 MiB; **peak 6.03 → 2.46 GB**. During backward, one block's activations exist at a time (recomputed). |
| `--checkpoint block --checkpoint-group N` | Same recompute, in groups of N blocks. | Fewer saved inputs, but a whole group's activations are recomputed at once, so **peak rises** with N (small: x1 2.46 GB, x3 3.07 GB) while retained-after-forward falls (326 → 278 MiB). The handout's memory-optimal choice balances these (≈ √num_layers groups). |
| `--checkpoint attn` | Attention's kernels re-run in backward. Step 174.0 → 194.0 ms. | Removes the (B, h, T, T) residuals: retained 4,150.8 → 2,126.0 MiB, peak 6.03 → 3.91 GB. Best memory saved per ms of recompute at long context. |
| `--checkpoint ffn` | FFN GEMMs re-run in backward. Step 174.0 → 187.9 ms. | Retained 4,150.8 → 2,711.2 MiB, peak 6.03 → 4.53 GB. |
| `--saved-tensors` | None for the measured steps (one extra forward afterwards). | None for the reported peak (measured before). |
| `--memory-snapshot` | Extra CPU time per allocation (stack-trace capture) → wider CPU ranges, more GPU gaps; don't profile and snapshot in the same run. | Produces the snapshot (§8). Recording itself uses host RAM, not GPU memory; peak GPU memory is unchanged. |
| Profiling itself | Adds overhead per CUDA call and NVTX range: small train step 174.60 → 206.70 ms; optimizer 21.9 → 36.6 ms. Launch-heavy phases distort most. | Snapshot recording adds CPU overhead; doesn't change what's allocated. |
| Per-phase `synchronize` (built in) | Short idle GPU gaps between forward/backward/optimizer, plus a long `cudaDeviceSynchronize` when the CPU was ahead. Real training doesn't have them. | None. |

**Planned settings — expected effects** (update with real measurements when implemented):

| Setting | Nsight Systems | Memory |
|---|---|---|
| FlashAttention-2 | Mask/softmax/scores kernels replaced by one fused attention kernel per layer (forward and backward); far fewer launches in `attn`. | No (B, h, T, T) tensors stored → attention memory linear in T; much longer contexts fit. |
| `torch.compile` | Chains of elementwise ops (RMSNorm, RoPE, SiLU, softmax) fused into Triton kernels → far fewer, larger kernels; first step includes compile time. | Fewer intermediate tensors saved. |
| NVTX ranges | New labelled rows (your domain) and projected ranges over the GPU; enables `--capture-range=nvtx`. | None. |

---

## 6. Output & baseline results

Printed: one progress line per size (`params`, fp32 training-state estimate), then:

| Column | Meaning |
|---|---|
| `params` | Analytic parameter count |
| `dtype` | `fp32`, `bf16` or `fp16` (`--dtype`) |
| `ckpt` | Activation checkpointing (`--checkpoint`): `none`, `block`, `block xN`, `attn`, `ffn` |
| `forward / backward / optimizer (ms)` | Mean ± std per phase; `-` if the mode doesn't run it |
| `step (ms)` | Mean ± std of the whole step |
| `peak mem (GB)` | `torch.cuda.max_memory_allocated()` for that config; `-` off CUDA or with `--no-allocator-cache` |
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

**Mixed precision — RTX 3090, ctx 512, batch 4, warmup 5, steps 10** (same pod run, 2026-10-14):

| Size | Mode | dtype | Forward (ms) | Backward (ms) | Optimizer (ms) | Step (ms) | Peak mem |
|---|---|---|---|---|---|---|---|
| small | train | fp32 | 46.50 ± 0.17 | 104.11 ± 0.19 | 21.63 ± 0.05 | 172.23 ± 0.28 | 6.03 GB |
| small | train | bf16 | 29.50 ± 0.08 | 63.29 ± 0.20 | 21.42 ± 0.15 | 114.22 ± 0.16 | 4.76 GB |
| small | train | fp16 | 28.00 ± 0.05 | 65.19 ± 0.10 | 22.93 ± 0.02 | 116.12 ± 0.10 | 4.76 GB |
| medium | train | fp32 | 140.42 ± 0.36 | 301.67 ± 0.72 | 68.09 ± 0.18 | 510.18 ± 0.70 | 16.37 GB |
| medium | train | bf16 | 75.62 ± 0.17 | 177.94 ± 0.24 | 67.88 ± 0.18 | 321.44 ± 0.38 | 13.18 GB |
| medium | train | fp16 | 76.64 ± 0.26 | 177.30 ± 0.28 | 72.79 ± 0.37 | 326.74 ± 0.70 | 13.18 GB |
| large | train | all | OOM | | | | |
| small | forward | fp32 / bf16 | 47.78 → 29.38 | | | | 0.81 → 1.05 GB |
| medium | forward | fp32 / bf16 | 144.95 → 76.32 | | | | 2.08 → 2.78 GB |
| large | forward | fp32 / bf16 | 295.78 → 147.06 | | | | 4.48 → 6.21 GB |
| xl | forward | fp32 / bf16 | 877.55 → 341.54 | | | | 14.55 → 20.89 GB |

Trends: bigger models gain more (more of their time is GEMM); the optimizer never speeds up (fp32 state,
launch-bound); fp16 ≈ bf16 here, with slightly slower optimizer steps from the GradScaler.

**Activation checkpointing — small, train, fp32, ctx 512, batch 4, RTX 3090** (2026-10-14; details in §10.3):

| ckpt | Forward (ms) | Backward (ms) | Step (ms) | Peak mem | Retained after forward |
|---|---|---|---|---|---|
| none | 47.09 ± 0.23 | 104.79 ± 0.37 | 173.95 ± 0.54 | 6.03 GB | 4,150.8 MiB |
| ffn | 46.96 ± 0.54 | 119.20 ± 0.26 | 187.94 ± 0.65 | 4.53 GB | 2,711.2 MiB |
| attn | 47.29 ± 0.48 | 124.89 ± 0.26 | 194.01 ± 0.71 | 3.91 GB | 2,126.0 MiB |
| block | 48.06 ± 2.74 | 142.67 ± 0.33 | 212.41 ± 2.89 | 2.46 GB | 326.2 MiB |
| block x3 | 47.34 ± 0.52 | 147.21 ± 0.30 | 216.29 ± 0.82 | 3.07 GB | 278.2 MiB |

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

# memory: GPU memory usage over time, aligned with PyTorch's per-module labels (§7.7)
nsys profile -o profiles/NAME --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  .venv/bin/python -m cs336_systems.benchmark --size small --mode train --warmup 0 --steps 1 --no-allocator-cache

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
| `--cuda-memory-usage=true` | A GPU memory-usage graph under CUDA HW, built from `cudaMalloc`/`cudaFree` (pair with `--no-allocator-cache`, §7.7) | Per-layer memory saved for backward |
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

### 7.7 Memory profiling in Nsight

Use this when you need memory **aligned with model structure** (e.g. "how much does one `TransformerBlock` save for
backward?") — memory_viz (§8) can't do that because its x-axis is allocation events and it has no layer boundaries.

**Why `--no-allocator-cache` is needed.** `--cuda-memory-usage=true` only sees CUDA-level `cudaMalloc`/`cudaFree`.
PyTorch's caching allocator reuses blocks it already reserved, so with the cache on Nsight sees only ~154 mallocs per
small train step and the graph mostly shows the cache *growing*. With the cache off every tensor is its own
malloc/free (3,600 / 3,299), so the graph follows actual tensor lifetimes. The env var must be set before the first
CUDA allocation; the flag does this at the top of `main()` (verified on the pod: reserved memory after freeing a
1 GiB tensor is 1024 MiB with the cache, 0 MiB without).

**Reading it.** Expand CUDA HW to find the memory-usage graph. Line it up with the projected `PyTorch:` ranges:
- **Forward:** usage climbs as each `TransformerLM.layers.N` runs; the rise across one layer ≈ that block's saved
  activations ("residuals"). Hover the graph at the layer's start and end.
- **Backward** (`pt_autograd_0`, op labels `…Backward0`): per block, residuals are freed while gradient tensors are
  allocated; the net change per block, combined with the forward rise, gives the gradient memory per block
  (≈ 4 bytes × that block's parameters in fp32).
- `--pytorch=autograd-shapes-nvtx` labels show which op allocated what (with tensor shapes) — use them to find the
  largest contributors.

Keep it to `--warmup 0 --steps 1` (or a capture range): the run is ~10× slower and only the memory is meaningful.

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

For "how much does each block keep for backward?" as numbers, use `--saved-tensors` (§10) instead.

Two tools, two questions: **memory_viz** (this section) answers *what* holds memory and *which line* allocated it;
**Nsight + `--cuda-memory-usage`** (§7.7) answers *how memory changes across layers/phases over time*. Don't combine
`--memory-snapshot` with `--no-allocator-cache`: the snapshot records the caching allocator, which is then off.

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

Under `--dtype bf16`: most activation bands are half as tall, plus bf16 weight copies created by autocast (forward
only). By mode: `forward` (no_grad) shows small short-lived bumps per layer and no gradients; `forward_backward` shows the
staircase and gradients but no `m`/`v`; `train` shows everything above. Longer `--context-length` makes the
(B, h, T, T) attention bands dominate (they grow with T²).

The handout's xl memory questions need ~54.5 GB of training state alone — too big for 24 GB; use small/medium
(same patterns) or a larger GPU.

---

## 9. HBM traffic per op (`--op-traffic`)

### 9.1 What it measures (and doesn't)

`OpTrafficMode` is a `torch.utils._python_dispatch.TorchDispatchMode` — the same hook `FlopCounterMode` uses. Every
aten op dispatched during the traced step (forward, backward on the autograd thread, optimizer) passes through it with
its real input and output tensors. In eager PyTorch each op is its own kernel that reads its inputs from GPU memory
(HBM) and writes its outputs back, so per op:

| Quantity | Estimate |
|---|---|
| bytes read | sum of **unique** input tensors' bytes; a broadcast/expanded view counts its storage, not `numel` |
| bytes written | sum of output tensors' bytes (for in-place ops, the mutated tensor) |
| `copy_(dst, src)` | reads `src` only |
| view/metadata ops (`view`, `reshape`, `transpose`, `slice`, `expand`, `detach`, …) and `empty*` | 0 — no kernel |

**Not visible:** L2 cache hits, traffic *inside* a kernel (SRAM / shared memory / registers — needs Nsight Compute
hardware counters, blocked on Runpod), and cuBLAS internals (a GEMM is counted as reading A and B once and writing C
once). It's an **estimate of HBM traffic for unfused eager code** — which is exactly the baseline FlashAttention and
`torch.compile` improve on: a fused kernel shows up as **one op** whose bytes are only its real inputs/outputs, so
the intermediate tensors of the chain it replaced disappear from the totals.

**Attribution.** Each op is tagged with the phase (`run_step` sets it) and, in the forward pass, the innermost module
running (forward pre/post hooks on every submodule), with layer indices collapsed (`layers.3.attn` → `layers.*.attn`).
Ops outside any submodule (the loss) are `(outside modules)`; backward/optimizer ops are `(backward)`/`(optimizer)`
because autograd and AdamW have no module structure.

### 9.2 Output

```
Estimated HBM traffic for one step ...
| phase | ops | read (GB) | written (GB) | total (GB) |          ← per phase + total
Forward pass by module (layers.* = summed over all layers)       ← attn vs ffn vs norms vs loss
Top N forward / backward / optimizer ops by traffic              ← which aten ops move the bytes
```

JSON (`--op-traffic-json`): `{"config": {size, params, context_length, mode, dtype, batch_size}, "rows": [{phase,
module, op, calls, bytes_read, bytes_written}, ...]}` — one row per (phase, module, op), modules collapsed across layers.

### 9.3 Baseline — small, train, ctx 512, batch 4 (RTX 3090, 2026-10-14)

| Phase | ops | fp32 read / written (GB) | fp32 total | bf16 total |
|---|---|---|---|---|
| forward | 2,231 | 10.132 / 7.251 | **17.383** | 15.072 |
| backward | 3,264 | 25.945 / 15.980 | **41.925** | 34.349 |
| optimizer | 1,778 | 9.776 / 7.203 | **16.979** | 16.979 |
| all | 7,273 | 45.852 / 30.434 | **76.287** | 66.400 |

Forward by module (fp32): `layers.*.attn` **11.21 GB** (64% of forward), `layers.*.ffn` 3.89 GB, norms ~0.53 GB each,
loss 0.57 GB. The top attention ops each move one `(B, h, T, T)` tensor per layer — 4·12·512²·4 B = 50.3 MB × 12 layers
= **0.604 GB**: `div` (scale by √d_k + softmax normalize), `where` (mask), `sub`, `exp`, `amax`, `sum`. The two GEMMs
(`bmm`, Q·Kᵀ and P·V) move only 1.5 GB together.

Takeaways:
- Attention's HBM traffic is dominated by **materializing and re-reading the T×T scores**, not by its matmuls —
  that's the traffic FlashAttention-2 eliminates (scores stay in SRAM). Re-run with the flash implementation and
  compare `layers.*.attn` and the backward total.
- **bf16 barely helps attention** (11.21 → 10.23 GB): softmax's ops run in fp32 under autocast, and the casts
  (`_to_copy`, 1.06 GB read) add traffic. It mostly helps the FFN (3.89 → 2.68 GB) and backward.
- The optimizer moves ~17 GB (≈33× the 0.51 GB of fp32 weights) in 1,778 small ops — a fused/foreach AdamW would
  read and write each tensor once.

### 9.4 Comparing implementations

```bash
uv run python -m cs336_systems.benchmark --size small --op-traffic-json profiles/traffic_small_naive.json
# … after switching attention implementation (planned --attention flag) …
uv run python -m cs336_systems.benchmark --size small --op-traffic-json profiles/traffic_small_flash.json
```

Diff the two JSON files by (phase, module, op) — or just compare the printed per-phase and per-module tables.
Use the same `--size`, `--context-length`, `--batch-size`, `--dtype` and `--mode` for both.

---

## 10. Saved tensors & activation checkpointing (`--saved-tensors`, `--checkpoint`)

### 10.1 What `--saved-tensors` measures

One extra, untimed forward pass (grad enabled, same `--dtype` autocast, same checkpointing) runs inside
`SavedTensorsTracker`, which measures what the forward pass keeps for backward ("residuals") in two ways:

| Measure | How | Notes |
|---|---|---|
| **hook-counted, per save** | `torch.autograd.graph.saved_tensors_hooks` — `pack_hook` adds `numel × element_size` for every tensor autograd saves (skipping `nn.Parameter`s), attributed to the innermost module running | Exactly the handout's §3.2 method/number. A tensor saved by several ops counts each time. |
| **hook-counted, unique** | Same hook, but each underlying storage counted once | What those saves actually cost in memory. |
| **retained** (CUDA) | `torch.cuda.memory_allocated()` at each module's forward entry and exit; the difference is what that module leaves allocated | Ground truth: saved tensors **plus the module's output** (which the next module consumes). The only measure that reflects checkpointing correctly (see below). |

Why two kinds: inside a `torch.utils.checkpoint` region, checkpoint installs its **own** saved-tensor hooks, so our
`pack_hook` never sees those saves — checkpointed modules read ≈ 0 by hook count. Retained memory still shows what
really stays allocated (the region's input).

Report sections: totals (per save, unique, retained after forward — the last includes logits and the loss graph);
a per-TransformerBlock table (block / attn / ffn / norms by hook, block / attn / ffn retained); modules outside the
blocks (`token_embeddings`, `ln_final`, `lm_head`, `(outside modules)` = the loss); and the largest saved tensors
summed over layers with shape, dtype and the grad_fn that saved them.

### 10.2 What `--checkpoint` does

`apply_checkpointing(model, mode, group)` replaces module instances' `forward` with
`lambda *a, **k: torch.utils.checkpoint.checkpoint(original, *a, use_reentrant=False, **k)` — your model code is not
edited. During forward the wrapped region saves only its inputs; during backward it **re-runs its forward**, then
backpropagates through the recomputed activations. Autocast state is preserved for the recompute.

| Mode | Wrapped | Saved per block (forward) | Recompute in backward |
|---|---|---|---|
| `none` | — | everything (§10.3) | none |
| `block` | each `TransformerBlock` | its input (B, T, d) | the whole block |
| `block` + `--checkpoint-group N` | first block of each group runs N blocks inside one checkpoint; the others pass `x` through | one input per group | N blocks at once |
| `attn` | each block's `attn` | attention input; FFN/norm residuals as usual | attention only |
| `ffn` | each block's `ffn` (SwiGLU) | FFN input; attention/norm residuals as usual | FFN only |

With groups, per-block attribution lands on each group's first block (the others show ≈ 0).

### 10.3 Results — small, train, fp32, ctx 512, batch 4 (RTX 3090, 2026-10-14)

No checkpointing, per TransformerBlock (all 12 identical):

| | per save | unique | retained |
|---|---|---|---|
| **block** | **463.0 MiB** | 360.7 MiB | 325.2 MiB |
| attention | 243.9 MiB | | 174.7 MiB |
| FFN | 183.0 MiB | | 126.0 MiB |
| ln1 + ln2 | 36.0 MiB | | |

Totals: 5,843.4 MiB per save, 4,610.2 MiB unique, **4,150.8 MiB retained** after forward. Largest saved tensors:
attention's (4, 12, 512, 512) fp32 tensors — `ExpBackward0` 1,152 MiB (24 saves), `WhereBackward0` 576 MiB,
the P·V `bmm` input 576 MiB — then the FFN's (4, 512, 3072) activations (SiLU/sigmoid, w1/w3 outputs).

| ckpt | retained after forward | per-block retained | peak | step |
|---|---|---|---|---|
| none | 4,150.8 MiB | 325.2 MiB | 6.03 GB | 173.95 ms |
| ffn | 2,711.2 MiB | 205.2 MiB | 4.53 GB | 187.94 ms |
| attn | 2,126.0 MiB | 156.8 MiB | 3.91 GB | 194.01 ms |
| block | 326.2 MiB | 6.0 MiB (= its (4, 512, 768) fp32 input) | 2.46 GB | 212.41 ms |
| block x3 | 278.2 MiB | 6.0 MiB per group | 3.07 GB | 216.29 ms |

Takeaways:
- Per-block checkpointing cuts what's kept for backward ~13× and peak memory 2.5×, for +22% step time (one extra
  forward of each block during backward: backward 104.8 → 142.7 ms).
- Attention's T×T residuals are the single biggest item — `attn` checkpointing alone recovers half of it for +12%;
  FlashAttention-2 (which never stores them) should make that cheaper still. Re-measure with `--saved-tensors` then.
- Bigger groups save fewer inputs but recompute more at once, so peak goes **up** (2.46 → 3.07 GB) — the
  memory-optimal group size balances stored inputs (∝ num_groups) against one group's activations (∝ group size).
- The handout's single xl block at ctx 2048 saves 3,651 MiB; ours is small at ctx 512 — same method, same pattern.

### 10.4 Functions (`activation_memory.py`)

| Name | Signature → returns | What it does |
|---|---|---|
| `CHECKPOINT_MODES` | `("none", "block", "attn", "ffn")` | `--checkpoint` choices |
| `apply_checkpointing(model, mode, group=1)` | `-> None` | Wraps forwards in place (§10.2) |
| `SavedTensorsTracker(model, cuda)` | context manager | Installs module pre/post hooks + `saved_tensors_hooks`; records per-module per-save bytes, unique bytes, retained bytes, and a tensor table |
| `measure_saved_tensors(model, run_forward, cuda)` | `(Module, () -> Tensor, bool) -> dict` | `zero_grad`, runs `run_forward()` (returns the loss) under the tracker with grad enabled, summarizes, frees the graph |
| `summarize(tracker, model)` | `-> dict` | `{"per_save_total": int, "unique_total": int, "retained_total": int \| None, "layers": [{"layer", "per_save": {part: bytes}, "per_save_total", "unique": {...}, "unique_total", "retained": {attn, ffn, ln1, ln2}, "retained_total"}], "other": {"per_save", "unique", "retained"}, "tensors": [{"module", "shape", "dtype", "grad_fn", "count", "bytes"}]}` (bytes; tensors sorted largest first) |
| `format_saved_report(summary, top=10, checkpoint_desc="none")` | `-> str` | The printed tables (MiB) |

---

## 11. Nsight app over SSH

*Manage targets → SSH*: host, port, user `root` from the pod's `ssh.direct` block. The key has a passphrase:
run `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` first. Run settings: command
`/workspace/assignment1-basics/.venv/bin/python -m cs336_systems.benchmark …`, working directory
`/workspace/assignment1-basics`; enable CUDA, NVTX, OS runtime (+ PyTorch tracing if offered); disable CPU sampling,
context switches, GPU metrics. First connect uploads ~350 MB of profiler files (minutes; again after a pod reset).

Version note: Mac app 2026.5.1 > pod `nsys` 2026.3.2. Pod reports open on the Mac; to summarize app-made reports
on the pod use the app's copy: `/root/.local/share/nsight_systems/nsys stats …`.

---

## 12. Pod workflow

```bash
git add -A && git commit -m "..." && git push            # Mac
cd /workspace/assignment1-basics && git pull             # pod (fast-forward only; the pod cannot push)
/workspace/pod_setup.sh                                  # pod, once after any restart (reinstalls nsys)
```

The SSH port changes on every pod start. Stop (don't terminate) the pod when idle; `/workspace` survives a stop.

---

## 13. Changelog

Benchmark-affecting changes, newest first. Add an entry for every change that can alter timings, kernels, memory
or profiles.

| Date | Change | Effect on benchmarks / profiles |
|---|---|---|
| 2026-10-14 | `--checkpoint {none,block,attn,ffn}`, `--checkpoint-group N`, `--saved-tensors`, `--saved-tensors-top` + `activation_memory.py`; `ckpt` table column; README implementation-status table | Plain runs unchanged. Checkpointing changes timings and memory for every step (small: block 174 → 212 ms, peak 6.03 → 2.46 GB; §10.3) |
| 2026-10-14 | `--op-traffic`, `--op-traffic-json`, `--op-traffic-top` + `op_traffic.py` (`OpTrafficMode` dispatch-mode tracer); `run_step` takes `tracer` | Timed steps unaffected (tracing is one extra step afterwards). Baseline: small train fp32 = 76.3 GB estimated HBM traffic per step, attention 64% of forward (§9.3) |
| 2026-10-14 | `--dtype {fp32,bf16,fp16}` (autocast, GradScaler for fp16; `dtype` column) and `--no-allocator-cache` (for `nsys --cuda-memory-usage`) added; peak memory skipped without the cache | fp32 runs unchanged. bf16: small train 172 → 114 ms, xl forward 878 → 342 ms; training memory down, forward-only memory up (§5, §6). `--memory-snapshot` verified on the 3090 (5.1 MB, 30,907 events) |
| 2026-10-14 | `--memory-snapshot PATH` added (records from before model build; dumps on success and on OOM; one file per size); `*.pickle` git-ignored | No effect on plain runs. Snapshot runs carry CPU-side recording overhead — timings not comparable |
| 2026-10-14 | README expanded (model structure, loop internals, return types, settings → profile effects) | Docs only |
| 2026-10-13 | `benchmark.py`, `model_configs.py` added (fp32, per-phase sync, naive attention) | Baseline; 3090 results in §6 |
