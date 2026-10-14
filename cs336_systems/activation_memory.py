# CO AUTHORED BY CLAUDE
"""Activation ("residual") memory: what the forward pass keeps for the backward pass, and activation checkpointing.

Two measurements over one forward pass (grad enabled, same dtype/autocast as the benchmark):

1. Hook-counted saved tensors — the handout's §3.2 method. torch.autograd.graph.saved_tensors_hooks calls pack_hook
   for every tensor autograd saves for backward; we add up numel * element_size (skipping nn.Parameters) and
   attribute it to the innermost module running. Reported two ways:
     - per save   : every save counted (the handout's number; a tensor saved by two ops counts twice)
     - unique     : each underlying storage counted once (what it actually costs in memory)
   Caveat: inside a torch.utils.checkpoint region, checkpoint installs its own saved-tensor hooks, so our hook never
   sees those saves — checkpointed modules read ~0 here. That's why there is a second measurement.

2. Retained memory (CUDA only) — torch.cuda.memory_allocated() just before and just after each module's forward.
   The difference is what that module leaves allocated: tensors saved for backward plus its output (which the next
   module consumes and usually also saves). This is ground truth and reflects checkpointing correctly.

Checkpointing (apply_checkpointing) wraps forwards with torch.utils.checkpoint.checkpoint(use_reentrant=False):
  - "block": each TransformerBlock (or each group of `group` consecutive blocks) keeps only its input; everything
    inside is recomputed during backward.
  - "attn" : only each block's attention module is checkpointed.
  - "ffn"  : only each block's SwiGLU feed-forward is checkpointed.
Your model code is not edited: the instances' forward methods are wrapped at runtime.
"""
import re
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint

CHECKPOINT_MODES = ("none", "block", "attn", "ffn")


# ---------------------------------------------------------------- checkpointing


def apply_checkpointing(model: torch.nn.Module, mode: str, group: int = 1) -> None:
    if mode == "none":
        return
    layers = model.layers
    if mode in ("attn", "ffn"):
        for block in layers:
            _checkpoint_module(getattr(block, mode))
        return
    # mode == "block": checkpoint groups of `group` consecutive blocks
    group = max(1, group)
    if group == 1:
        for block in layers:
            _checkpoint_module(block)
        return
    originals = [block.forward for block in layers]
    for start in range(0, len(layers), group):
        members = originals[start:start + group]

        def run_group(x, token_positions=None, members=members):
            for forward in members:
                x = forward(x, token_positions)
            return x

        # the group's first block runs the whole group inside one checkpoint; the rest pass x through unchanged
        layers[start].forward = lambda x, token_positions=None, run_group=run_group: checkpoint(
            run_group, x, token_positions, use_reentrant=False)
        for j in range(start + 1, min(start + group, len(layers))):
            layers[j].forward = lambda x, token_positions=None: x


def _checkpoint_module(module: torch.nn.Module) -> None:
    original = module.forward
    module.forward = lambda *args, **kwargs: checkpoint(original, *args, use_reentrant=False, **kwargs)


# ---------------------------------------------------------------- measurement


def _collapse_layer(name: str) -> str:
    return re.sub(r"layers\.\d+", "layers.*", name)


def _block_of(name: str) -> str | None:
    match = re.match(r"(layers\.\d+)", name)
    return match.group(1) if match else None


def _part_of(name: str) -> str:
    # the block-level part a module belongs to: attn / ffn / ln1 / ln2 / (block itself)
    parts = name.split(".")
    return parts[2] if len(parts) >= 3 and parts[0] == "layers" else ("(block)" if parts[0] == "layers" else parts[0])


class SavedTensorsTracker:
    """Context manager: `with SavedTensorsTracker(model, cuda=True) as t: loss = run_forward()`."""

    def __init__(self, model: torch.nn.Module, cuda: bool):
        self.model = model
        self.cuda = cuda
        self._stack: list[str] = []
        self._handles = []
        self._seen_storages: set[int] = set()
        self.per_save = defaultdict(int)       # module -> bytes (every save)
        self.unique = defaultdict(int)         # module -> bytes (first time a storage is saved)
        self.retained = {}                     # module -> bytes left allocated by its forward (CUDA)
        self._alloc_at_entry: dict[str, list[int]] = defaultdict(list)
        self.tensors = defaultdict(lambda: [0, 0])  # (module*, shape, dtype, grad_fn) -> [count, bytes]
        self.total_retained = None

    # module bookkeeping
    def _pre(self, name):
        def hook(module, args):
            self._stack.append(name)
            if self.cuda:
                self._alloc_at_entry[name].append(torch.cuda.memory_allocated())
        return hook

    def _post(self, name):
        def hook(module, args, output):
            self._stack.pop()
            if self.cuda:
                self.retained[name] = torch.cuda.memory_allocated() - self._alloc_at_entry[name].pop()
        return hook

    def _pack(self, t: torch.Tensor):
        if isinstance(t, torch.nn.Parameter):
            return t  # parameters are kept anyway; the handout skips them too
        module = self._stack[-1] if self._stack else "(outside modules)"
        nbytes = t.numel() * t.element_size()
        self.per_save[module] += nbytes
        storage = t.untyped_storage()
        if storage.data_ptr() not in self._seen_storages:
            self._seen_storages.add(storage.data_ptr())
            self.unique[module] += storage.nbytes()
        grad_fn = type(t.grad_fn).__name__ if t.grad_fn is not None else "(input/leaf)"
        key = (_collapse_layer(module), tuple(t.shape), str(t.dtype).replace("torch.", ""), grad_fn)
        self.tensors[key][0] += 1
        self.tensors[key][1] += nbytes
        return t

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name:
                self._handles.append(module.register_forward_pre_hook(self._pre(name)))
                self._handles.append(module.register_forward_hook(self._post(name)))
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hooks.__enter__()
        if self.cuda:
            self._alloc_start = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.total_retained = torch.cuda.memory_allocated() - self._alloc_start
        self._hooks.__exit__(*exc)
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def measure_saved_tensors(model: torch.nn.Module, run_forward: Callable[[], torch.Tensor], cuda: bool) -> dict:
    """One forward pass with grad enabled; returns the summary dict described in summarize()."""
    model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
    with torch.enable_grad():
        with SavedTensorsTracker(model, cuda) as tracker:
            loss = run_forward()
        if cuda:
            torch.cuda.synchronize()
        summary = summarize(tracker, model)
        del loss  # frees the graph and everything it saved
    return summary


def summarize(tracker: SavedTensorsTracker, model: torch.nn.Module) -> dict:
    num_layers = len(model.layers)
    layer_rows = []
    for i in range(num_layers):
        prefix = f"layers.{i}"
        row = {"layer": i}
        for kind, source in (("per_save", tracker.per_save), ("unique", tracker.unique)):
            by_part = defaultdict(int)
            for module, nbytes in source.items():
                if module == prefix or module.startswith(prefix + "."):
                    by_part[_part_of(module)] += nbytes
            row[kind] = dict(by_part)
            row[f"{kind}_total"] = sum(by_part.values())
        if tracker.cuda:
            row["retained"] = {part: tracker.retained.get(f"{prefix}.{part}") for part in ("attn", "ffn", "ln1", "ln2")}
            row["retained_total"] = tracker.retained.get(prefix)
        layer_rows.append(row)

    other = {}
    for kind, source in (("per_save", tracker.per_save), ("unique", tracker.unique)):
        by_top = defaultdict(int)
        for module, nbytes in source.items():
            if _block_of(module) is None:
                by_top[module.split(".")[0]] += nbytes
        other[kind] = dict(by_top)
    if tracker.cuda:
        other["retained"] = {m: tracker.retained.get(m) for m in ("token_embeddings", "ln_final", "lm_head")}

    tensors = [
        {"module": m, "shape": list(shape), "dtype": dt, "grad_fn": gf, "count": c, "bytes": b}
        for (m, shape, dt, gf), (c, b) in tracker.tensors.items()
    ]
    tensors.sort(key=lambda r: -r["bytes"])
    return {
        "per_save_total": sum(tracker.per_save.values()),
        "unique_total": sum(tracker.unique.values()),
        "retained_total": tracker.total_retained,
        "layers": layer_rows,
        "other": other,
        "tensors": tensors,
    }


# ---------------------------------------------------------------- report


def _mib(n) -> str:
    return "-" if n is None else f"{n / 2**20:,.1f}"


def format_saved_report(summary: dict, top: int = 10, checkpoint_desc: str = "none") -> str:
    lines = [f"Saved for backward — one forward pass (checkpointing: {checkpoint_desc}); sizes in MiB"]
    lines.append("| measure | total (MiB) |")
    lines.append("|---|---|")
    lines.append(f"| hook-counted, per save (handout §3.2 method) | {_mib(summary['per_save_total'])} |")
    lines.append(f"| hook-counted, unique storages | {_mib(summary['unique_total'])} |")
    lines.append(f"| retained after forward (memory_allocated delta, includes loss/logits) | "
                 f"{_mib(summary['retained_total'])} |")

    lines.append("")
    lines.append("Per TransformerBlock (hook per-save / unique; retained = memory left allocated by that module's forward)")
    lines.append("| layer | block per-save | block unique | attn per-save | ffn per-save | norms per-save "
                 "| block retained | attn retained | ffn retained |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in summary["layers"]:
        ps = row["per_save"]
        norms = ps.get("ln1", 0) + ps.get("ln2", 0)
        retained = row.get("retained", {})
        lines.append(
            f"| {row['layer']} | {_mib(row['per_save_total'])} | {_mib(row['unique_total'])} | "
            f"{_mib(ps.get('attn', 0))} | {_mib(ps.get('ffn', 0))} | {_mib(norms)} | "
            f"{_mib(row.get('retained_total'))} | {_mib(retained.get('attn'))} | {_mib(retained.get('ffn'))} |"
        )

    other = summary["other"]
    lines.append("")
    lines.append("Outside the blocks")
    lines.append("| module | per-save | unique | retained |")
    lines.append("|---|---|---|---|")
    names = sorted(set(other["per_save"]) | set(other.get("retained", {})))
    for name in names:
        lines.append(f"| {name} | {_mib(other['per_save'].get(name, 0))} | {_mib(other['unique'].get(name, 0))} | "
                     f"{_mib(other.get('retained', {}).get(name))} |")

    if summary["tensors"]:
        lines.append("")
        lines.append(f"Largest saved tensors (summed over layers; top {min(top, len(summary['tensors']))})")
        lines.append("| module | shape | dtype | saved by (grad_fn) | count | total MiB |")
        lines.append("|---|---|---|---|---|---|")
        for r in summary["tensors"][:top]:
            lines.append(f"| {r['module']} | {tuple(r['shape'])} | {r['dtype']} | {r['grad_fn']} | {r['count']} | "
                         f"{_mib(r['bytes'])} |")
    return "\n".join(lines)
