# CO AUTHORED BY CLAUDE
"""Per-op HBM traffic estimate for one step, via a TorchDispatchMode.

Every aten op PyTorch dispatches during the traced step is intercepted with its input and output tensors.
In eager PyTorch each op is (roughly) its own kernel that reads its inputs from GPU memory (HBM) and writes its
outputs back, so for each op:

    bytes read    = unique input tensors' bytes   (a broadcast/expanded view counts its storage, not its numel)
    bytes written = output tensors' bytes

View/metadata ops (view, reshape, transpose, slice, expand, ...) and pure allocations (empty) launch no kernel and
count as 0. This is an estimate of HBM traffic, not a measurement: it can't see caching (L2), traffic inside a
kernel (SRAM / shared memory / registers), or what a fused kernel avoids writing. That last point is exactly what
it's for — a fused kernel (FlashAttention, torch.compile) shows up as one op with far fewer bytes than the chain of
eager ops it replaces.

Ops are attributed to a phase (forward / backward / optimizer, set by the benchmark) and, in the forward pass, to the
innermost module running (e.g. layers.3.attn.q_proj; layer indices collapse to layers.*). Backward and optimizer
ops have no module structure and are attributed to the phase only.
"""
import json
import re
from collections import defaultdict

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

# ops that only change metadata / alias storage (no kernel, no traffic) besides those PyTorch marks as views
NO_TRAFFIC_OPS = {
    "detach", "alias", "_unsafe_view", "lift_fresh", "empty", "empty_strided", "empty_like", "new_empty",
    "new_empty_strided", "_local_scalar_dense", "set_", "resize_", "_reshape_alias", "unsqueeze_", "squeeze_",
}


def _tensor_bytes(t: torch.Tensor) -> int:
    # an expanded/broadcast view (stride 0) reads its storage once, not numel times
    return min(t.numel() * t.element_size(), t.untyped_storage().nbytes())


def _unique_tensors(values) -> list[torch.Tensor]:
    seen, out = set(), []
    for v in tree_flatten(values)[0]:
        if isinstance(v, torch.Tensor):
            key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), v.dtype)
            if key not in seen:
                seen.add(key)
                out.append(v)
    return out


def _collapse(module_name: str) -> str:
    return re.sub(r"layers\.\d+", "layers.*", module_name) if module_name else "(outside modules)"


def _coarse(module_name: str) -> str:
    # layers.*.attn.q_proj -> layers.*.attn ; lm_head -> lm_head
    parts = module_name.split(".")
    return ".".join(parts[:3]) if parts[0] == "layers" else parts[0]


class OpTrafficMode(TorchDispatchMode):
    """Use as `with OpTrafficMode(model) as tracer: ...`; set `tracer.phase` before each phase."""

    def __init__(self, model: torch.nn.Module | None = None):
        super().__init__()
        self.phase = "forward"
        self._module_stack: list[str] = []
        self._handles = []
        # (phase, module, op) -> [calls, bytes_read, bytes_written]
        self.records: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        if model is not None:
            for name, module in model.named_modules():
                if name:  # skip the root; ops directly in TransformerLM.forward show as "(outside modules)"
                    self._handles.append(module.register_forward_pre_hook(self._push(name)))
                    self._handles.append(module.register_forward_hook(self._pop()))

    def _push(self, name):
        def hook(module, args):
            self._module_stack.append(name)
        return hook

    def _pop(self):
        def hook(module, args, output):
            self._module_stack.pop()
        return hook

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        is_view = getattr(func, "is_view", False)
        if is_view or name in NO_TRAFFIC_OPS:
            read = written = 0
        else:
            inputs = _unique_tensors((args, kwargs))
            if name in ("copy_", "copy"):
                inputs = inputs[1:]  # copy_(dst, src) only reads src
            read = sum(_tensor_bytes(t) for t in inputs)
            written = sum(_tensor_bytes(t) for t in _unique_tensors(out))
        module = _collapse(self._module_stack[-1]) if (self.phase == "forward" and self._module_stack) else (
            "(outside modules)" if self.phase == "forward" else f"({self.phase})")
        rec = self.records[(self.phase, module, name)]
        rec[0] += 1
        rec[1] += read
        rec[2] += written
        return out

    # ---------- reporting ----------

    def rows(self) -> list[dict]:
        return [
            {"phase": p, "module": m, "op": o, "calls": c, "bytes_read": r, "bytes_written": w}
            for (p, m, o), (c, r, w) in self.records.items()
        ]


def _gb(n: int) -> str:
    return f"{n / 1e9:.3f}"


def format_report(rows: list[dict], top: int = 15) -> str:
    lines = []
    by_phase = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        agg = by_phase[r["phase"]]
        agg[0] += r["calls"]; agg[1] += r["bytes_read"]; agg[2] += r["bytes_written"]
    total_r = sum(v[1] for v in by_phase.values())
    total_w = sum(v[2] for v in by_phase.values())

    lines.append("Estimated HBM traffic for one step (eager ops: each reads inputs from / writes outputs to HBM)")
    lines.append("| phase | ops | read (GB) | written (GB) | total (GB) |")
    lines.append("|---|---|---|---|---|")
    for phase in ("forward", "backward", "optimizer"):
        if phase in by_phase:
            c, r, w = by_phase[phase]
            lines.append(f"| {phase} | {c} | {_gb(r)} | {_gb(w)} | {_gb(r + w)} |")
    lines.append(f"| **all** | {sum(v[0] for v in by_phase.values())} | {_gb(total_r)} | {_gb(total_w)} | "
                 f"{_gb(total_r + total_w)} |")

    # forward pass broken down by module group (all layers summed)
    by_module = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["phase"] == "forward":
            agg = by_module[_coarse(r["module"])]
            agg[0] += r["calls"]; agg[1] += r["bytes_read"]; agg[2] += r["bytes_written"]
    if by_module:
        lines.append("")
        lines.append("Forward pass by module (layers.* = summed over all layers)")
        lines.append("| module | ops | read (GB) | written (GB) | total (GB) |")
        lines.append("|---|---|---|---|---|")
        for m, (c, r, w) in sorted(by_module.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
            lines.append(f"| {m} | {c} | {_gb(r)} | {_gb(w)} | {_gb(r + w)} |")

    # top ops per phase
    for phase in ("forward", "backward", "optimizer"):
        prow = [r for r in rows if r["phase"] == phase and r["bytes_read"] + r["bytes_written"] > 0]
        if not prow:
            continue
        prow.sort(key=lambda r: -(r["bytes_read"] + r["bytes_written"]))
        lines.append("")
        lines.append(f"Top {min(top, len(prow))} {phase} ops by traffic")
        lines.append("| module | op | calls | read (GB) | written (GB) |")
        lines.append("|---|---|---|---|---|")
        for r in prow[:top]:
            lines.append(f"| {r['module']} | {r['op']} | {r['calls']} | {_gb(r['bytes_read'])} | "
                         f"{_gb(r['bytes_written'])} |")
    return "\n".join(lines)


def write_json(path: str, rows: list[dict], config: dict) -> None:
    with open(path, "w") as f:
        json.dump({"config": config, "rows": rows}, f, indent=1)
