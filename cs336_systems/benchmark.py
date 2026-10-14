# CO AUTHORED BY CLAUDE
"""End-to-end benchmark of the assignment 1 TransformerLM: forward pass, backward pass, optimizer step.

Run from the repo root, for example:
    uv run python -m cs336_systems.benchmark --size small
    uv run python -m cs336_systems.benchmark --size all --mode forward
    uv run python -m cs336_systems.benchmark --size medium --context-length 1024 --warmup 0
    uv run python -m cs336_systems.benchmark --size small --num-layers 4     # override any dimension
    uv run python -m cs336_systems.benchmark --size small --warmup 1 --steps 2 \
        --memory-snapshot profiles/mem_small.pickle                          # memory snapshot (CUDA only)
    uv run python -m cs336_systems.benchmark --size all --dtype bf16         # mixed precision (autocast)
    nsys profile --cuda-memory-usage=true ... -m cs336_systems.benchmark --no-allocator-cache   # Nsight memory
    uv run python -m cs336_systems.benchmark --size small --op-traffic       # per-op HBM traffic estimate
    uv run python -m cs336_systems.benchmark --size small --saved-tensors    # memory saved for backward, per block
    uv run python -m cs336_systems.benchmark --size small --checkpoint block --saved-tensors   # activation checkpointing

Modes:
    forward           forward pass + loss only, under torch.no_grad() (inference)
    forward_backward  forward + loss + backward
    train             forward + loss + backward + AdamW step (one full training step)

Each measured step is timed per phase, with a device synchronize after every phase so the timer
waits for the GPU to actually finish instead of just for the kernel launches to return.

With --memory-snapshot, PyTorch's allocator records every allocation/free (with stack traces) from before
the model is built until the end of the run, and the history is dumped to a pickle for pytorch.org/memory_viz.
Recording slows the CPU side, so timings from snapshot runs are not comparable with plain runs.

With --dtype bf16/fp16, forward + loss run under torch.autocast: matmuls and other autocast-eligible ops run in
low precision (Tensor Cores), while weights, gradients and AdamW state stay fp32. fp16 also uses a GradScaler.

With --no-allocator-cache, PyTorch's caching allocator is disabled (PYTORCH_NO_CUDA_MEMORY_CACHING=1) so every
tensor is a real cudaMalloc/cudaFree; that lets `nsys --cuda-memory-usage=true` track per-tensor memory. Much slower.

With --op-traffic, one extra untimed step runs after the measured steps under a TorchDispatchMode that estimates
bytes read from / written to HBM per op (see op_traffic.py). Timings are unaffected.

With --checkpoint {block,attn,ffn}, torch.utils.checkpoint wraps those modules for every step (timings include the
recomputation). With --saved-tensors, one extra untimed forward pass reports how much is kept for backward per
TransformerBlock / attention / FFN, by saved-tensor hooks and by retained GPU memory (see activation_memory.py).
"""
import argparse
import contextlib
import dataclasses
import gc
import os
import statistics
import timeit
from pathlib import Path

import torch

from cs336_basics.model.functional import cross_entropy_loss
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.training.optim import AdamW
from cs336_systems.activation_memory import (
    CHECKPOINT_MODES,
    apply_checkpointing,
    format_saved_report,
    measure_saved_tensors,
)
from cs336_systems.op_traffic import OpTrafficMode, format_report, write_json
from cs336_systems.model_configs import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    MODEL_SIZES,
    ROPE_THETA,
    VOCAB_SIZE,
    ModelSize,
    count_parameters,
    training_state_gb,
)

MODES = ("forward", "forward_backward", "train")
PHASES = ("forward", "backward", "optimizer")
# cap on recorded allocation events so a long run can't make the snapshot (and host memory) grow without bound
MEMORY_HISTORY_MAX_ENTRIES = 1_000_000
# --dtype choices: the autocast dtype, or None for plain fp32 (no autocast)
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# wait until every kernel queued on the device has finished (CUDA calls return before the GPU is done)
def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_model(size: ModelSize, context_length: int, device: str) -> TransformerLM:
    return TransformerLM(
        VOCAB_SIZE, context_length, size.num_layers, size.d_model, size.num_heads, size.d_ff,
        theta=ROPE_THETA, device=device,
    )


# random weights and random token ids are fine: we only measure speed and memory, not quality
def random_batch(batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    return x, y


# mixed precision: autocast for bf16/fp16, a no-op context for fp32
def autocast_context(device: str, dtype: str):
    if DTYPES[dtype] is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=DTYPES[dtype])


# one step of the chosen mode; returns seconds spent in each phase that ran
def run_step(model, optimizer, x, y, mode: str, device: str, dtype: str = "fp32",
             scaler: torch.amp.GradScaler | None = None, tracer: OpTrafficMode | None = None) -> dict[str, float]:
    times = {}
    # tracer (op traffic): label every op with the phase it ran in
    set_phase = (lambda phase: setattr(tracer, "phase", phase)) if tracer is not None else (lambda phase: None)
    if mode != "forward":
        model.zero_grad(set_to_none=True)
    synchronize(device)

    start = timeit.default_timer()
    set_phase("forward")
    # only forward + loss go inside autocast; backward reuses the dtypes the forward ops chose
    with torch.set_grad_enabled(mode != "forward"), autocast_context(device, dtype):
        loss = cross_entropy_loss(model(x), y)
    synchronize(device)
    after_forward = timeit.default_timer()
    times["forward"] = after_forward - start
    if mode == "forward":
        return times

    set_phase("backward")
    if scaler is not None:
        scaler.scale(loss).backward()  # fp16: scale the loss up so small gradients don't underflow to zero
    else:
        loss.backward()
    synchronize(device)
    after_backward = timeit.default_timer()
    times["backward"] = after_backward - after_forward
    if mode == "forward_backward":
        return times

    set_phase("optimizer")
    if scaler is not None:
        scaler.step(optimizer)  # unscales gradients, skips the step if any are inf/nan
        scaler.update()
    else:
        optimizer.step()
    synchronize(device)
    times["optimizer"] = timeit.default_timer() - after_backward
    return times


def mean_std_ms(values: list[float]) -> tuple[float, float]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values) * 1000, std * 1000


# where one config's snapshot goes: the given path, or path_<size> when several sizes run in one invocation
def snapshot_path(base: str, size_name: str, multiple_sizes: bool) -> str:
    path = Path(base)
    if multiple_sizes:
        safe_name = size_name.replace("*", "_custom")
        path = path.with_name(f"{path.stem}_{safe_name}{path.suffix or '.pickle'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def dump_memory_snapshot(path: str) -> None:
    torch.cuda.memory._dump_snapshot(path)
    print(f"    memory snapshot written to {path}", flush=True)


def benchmark(size: ModelSize, context_length: int, batch_size: int, mode: str, warmup: int, steps: int, device: str,
              memory_snapshot: str | None = None, dtype: str = "fp32", op_traffic: bool = False,
              op_traffic_top: int = 15, checkpoint: str = "none", checkpoint_group: int = 1,
              saved_tensors: bool = False, saved_tensors_top: int = 10) -> dict:
    checkpoint_desc = f"block x{checkpoint_group}" if checkpoint == "block" and checkpoint_group > 1 else checkpoint
    result = {"size": size.name, "params": count_parameters(size), "context_length": context_length, "mode": mode,
              "dtype": dtype, "checkpoint": checkpoint_desc}
    model = optimizer = None
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        if memory_snapshot:
            # start before build_model so the weight allocations are part of the history
            torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_MAX_ENTRIES)
            result["memory_snapshot"] = memory_snapshot
        model = build_model(size, context_length, device)
        # activation checkpointing wraps module forwards in place; every step below (timed or not) uses it
        apply_checkpointing(model, checkpoint, checkpoint_group)
        optimizer = AdamW(model.parameters()) if mode == "train" else None
        x, y = random_batch(batch_size, context_length, device)
        # fp16 needs loss scaling whenever gradients are computed; bf16 has fp32's exponent range and doesn't
        scaler = torch.amp.GradScaler(device.split(":")[0]) if dtype == "fp16" and mode != "forward" else None

        for _ in range(warmup):
            run_step(model, optimizer, x, y, mode, device, dtype, scaler)
        measured = [run_step(model, optimizer, x, y, mode, device, dtype, scaler) for _ in range(steps)]

        for phase in PHASES:
            if phase in measured[0]:
                result[phase] = mean_std_ms([m[phase] for m in measured])
        result["step"] = mean_std_ms([sum(m.values()) for m in measured])
        # with the caching allocator disabled PyTorch's memory counters stay at 0, so don't report a fake peak
        if device.startswith("cuda") and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
            result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if memory_snapshot:
            dump_memory_snapshot(memory_snapshot)
        if op_traffic:
            # one extra, untimed step under the dispatch-mode tracer, after everything above was measured
            tracer = OpTrafficMode(model)
            try:
                with tracer:
                    run_step(model, optimizer, x, y, mode, device, dtype, scaler, tracer=tracer)
            finally:
                tracer.remove_hooks()
            result["op_traffic_rows"] = tracer.rows()
            print(format_report(result["op_traffic_rows"], top=op_traffic_top), flush=True)
        if saved_tensors:
            # one extra, untimed forward pass (grad enabled, same autocast) measuring what is kept for backward
            def run_forward():
                with autocast_context(device, dtype):
                    return cross_entropy_loss(model(x), y)

            measurable = device.startswith("cuda") and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1"
            result["saved_tensors"] = measure_saved_tensors(model, run_forward, cuda=measurable)
            print(format_saved_report(result["saved_tensors"], top=saved_tensors_top, checkpoint_desc=checkpoint_desc),
                  flush=True)
    except torch.OutOfMemoryError:
        result["oom"] = True
        if memory_snapshot:
            # dump before cleanup: the snapshot at the moment of OOM shows exactly what filled the GPU
            dump_memory_snapshot(memory_snapshot)
    finally:
        if memory_snapshot:
            torch.cuda.memory._record_memory_history(enabled=None)  # stop recording
        # free everything before the next config so one size's memory never leaks into the next
        del model, optimizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result


def format_table(results: list[dict]) -> str:
    def cell(r, key):
        if r.get("oom"):
            return "OOM"
        if key not in r:
            return "-"
        mean, std = r[key]
        return f"{mean:.2f} ± {std:.2f}"

    header = ("| size | params | mode | dtype | ckpt | ctx | forward (ms) | backward (ms) | optimizer (ms) | step (ms) "
              "| peak mem (GB) |")
    rows = [header, "|" + "---|" * 11]
    for r in results:
        peak = "OOM" if r.get("oom") else (f"{r['peak_mem_gb']:.2f}" if "peak_mem_gb" in r else "-")
        rows.append(
            f"| {r['size']} | {r['params'] / 1e6:.0f}M | {r['mode']} | {r['dtype']} | {r['checkpoint']} | "
            f"{r['context_length']} | "
            f"{cell(r, 'forward')} | {cell(r, 'backward')} | {cell(r, 'optimizer')} | {cell(r, 'step')} | {peak} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="small", choices=[*MODEL_SIZES, "all"])
    parser.add_argument("--mode", default="train", choices=MODES)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default=default_device())
    # optional overrides of a size's dimensions, e.g. --size small --num-layers 4
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    # memory profiling: record allocator history from before model build and dump it here (CUDA only)
    parser.add_argument("--memory-snapshot", metavar="PATH",
                        help="write a PyTorch memory snapshot (.pickle) for pytorch.org/memory_viz; "
                             "with --size all, one file per size: PATH_<size>.pickle")
    # mixed precision: run forward + loss under torch.autocast in this dtype (fp32 = no autocast)
    parser.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    # Nsight memory profiling: make every tensor a real cudaMalloc/cudaFree so nsys --cuda-memory-usage sees it
    parser.add_argument("--no-allocator-cache", action="store_true",
                        help="disable PyTorch's CUDA caching allocator (for nsys --cuda-memory-usage=true); much slower")
    # HBM traffic estimate: one extra untimed step traced op by op (bytes read/written per op, phase, module)
    parser.add_argument("--op-traffic", action="store_true",
                        help="print an estimate of bytes read/written per op for one extra (untimed) step")
    parser.add_argument("--op-traffic-json", metavar="PATH",
                        help="also save the per-op rows as JSON (implies --op-traffic); with --size all: PATH_<size>.json")
    parser.add_argument("--op-traffic-top", type=int, default=15, metavar="N", help="ops listed per phase (default 15)")
    # activation checkpointing (torch.utils.checkpoint) applied to every step
    parser.add_argument("--checkpoint", default="none", choices=CHECKPOINT_MODES,
                        help="checkpoint each TransformerBlock (block), only attention (attn), or only the FFN (ffn)")
    parser.add_argument("--checkpoint-group", type=int, default=1, metavar="N",
                        help="with --checkpoint block: checkpoint groups of N consecutive blocks (default 1)")
    # memory kept for backward: one extra untimed forward with saved-tensor hooks + retained-memory accounting
    parser.add_argument("--saved-tensors", action="store_true",
                        help="report memory saved for backward per TransformerBlock / attention / FFN")
    parser.add_argument("--saved-tensors-top", type=int, default=10, metavar="N",
                        help="largest saved tensors to list (default 10)")
    args = parser.parse_args()
    if args.memory_snapshot and not args.device.startswith("cuda"):
        parser.error("--memory-snapshot requires a CUDA device")
    if args.no_allocator_cache:
        # must happen before the first CUDA allocation, which is why it's the first thing after parsing
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    sizes = list(MODEL_SIZES.values()) if args.size == "all" else [MODEL_SIZES[args.size]]
    overrides = {k: getattr(args, k) for k in ("d_model", "d_ff", "num_layers", "num_heads") if getattr(args, k) is not None}
    if overrides:
        sizes = [dataclasses.replace(s, name=f"{s.name}*", **overrides) for s in sizes]

    print(f"device={args.device} batch={args.batch_size} ctx={args.context_length} mode={args.mode} "
          f"dtype={args.dtype} checkpoint={args.checkpoint} warmup={args.warmup} steps={args.steps}")
    if args.no_allocator_cache:
        print("  caching allocator disabled: every tensor is a real cudaMalloc/cudaFree; timings are not comparable")
    if args.memory_snapshot:
        print("  memory snapshot recording is on: timings below include recording overhead and are not "
              "comparable with plain runs")
    results = []
    for size in sizes:
        print(f"  {size.name}: {count_parameters(size) / 1e6:.0f}M params, "
              f"~{training_state_gb(size):.1f} GB fp32 training state (before activations) ...", flush=True)
        snapshot = snapshot_path(args.memory_snapshot, size.name, len(sizes) > 1) if args.memory_snapshot else None
        result = benchmark(size, args.context_length, args.batch_size, args.mode, args.warmup, args.steps,
                           args.device, memory_snapshot=snapshot, dtype=args.dtype,
                           op_traffic=args.op_traffic or bool(args.op_traffic_json), op_traffic_top=args.op_traffic_top,
                           checkpoint=args.checkpoint, checkpoint_group=args.checkpoint_group,
                           saved_tensors=args.saved_tensors, saved_tensors_top=args.saved_tensors_top)
        if args.op_traffic_json and "op_traffic_rows" in result:
            path = snapshot_path(args.op_traffic_json, size.name, len(sizes) > 1)
            config = {k: result[k] for k in ("size", "params", "context_length", "mode", "dtype")}
            config["batch_size"] = args.batch_size
            write_json(path, result["op_traffic_rows"], config)
            print(f"    op traffic written to {path}", flush=True)
        results.append(result)
    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
