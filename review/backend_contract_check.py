"""Capture the fixed Backend fixtures on identified PyTorch kernels (K6).

Run with ./run.py backend-contracts. These finite primitive checks do not
establish a uniform kernel error bound, a whole-model ModelAccurate premise,
or a guarantee for uncaptured inputs/shapes. The output values are decoded
exactly and checked against independent rational witnesses from the Lean
theorems; no second floating-point implementation is an exact reference.
"""

import hashlib
import json
from pathlib import Path
import platform

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from backend_contract_arithmetic import NAMES, check


ROOT = Path(__file__).resolve().parent


class Probe(nn.Module):
    """Source primitive calls, fixed inputs, no training or optimizer changes."""

    def forward(self, x, gate, q, k, v, matrix):
        exponents = torch.arange(0, 4, 2, device=x.device, dtype=torch.float32) / 4
        phases = 10000.0**-exponents
        average = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return (F.rms_norm(x, (4,), eps=1 / 8388608), F.normalize(x, dim=-1, eps=1e-12),
                F.silu(gate), average[0, 0, 1], phases.cos(), phases.sin(), matrix @ x)


def inputs(device, dtype):
    """The SDPA last row has exact scaled scores [0, 1/2] and two unit values."""
    make = lambda value: torch.tensor(value, dtype=dtype, device=device)
    return (make([1, 2, 3, 4]), make([.5]),
            make([[[[1, 0, 0, 0], [1, 0, 0, 0]]]]),
            make([[[[0, 0, 0, 0], [1, 0, 0, 0]]]]),
            make([[[[1, 0, 0, 0], [0, 1, 0, 0]]]]),
            make([[1, .5, -.25, 0], [-1, 0, .125, 2]]))


def capture(device, dtype, compiled):
    model = Probe().to(device)
    args = inputs(device, dtype)
    if compiled:
        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        # Complete compilation/autotuning before recording kernel names.
        model(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            tensors = model(*args)
            if device == "cuda":
                torch.cuda.synchronize()
    outputs = {name: value.detach().cpu().flatten().tolist()
               for name, value in zip(NAMES, tensors, strict=True)}
    return {"device": device, "dtype": str(dtype),
            "compile": "max-autotune/fullgraph/static" if compiled else "eager",
            "sdpa_backend": "MATH", "outputs": outputs, "checks": check(outputs),
            "observed_events": sorted({event.key for event in profile.key_averages()}),
            "reduction_order": "backend-managed; no sequential-reduction premise is used"}


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    source = json.loads((ROOT / "sequence_model_source.json").read_text())
    for name, digest in source["files"].items():
        path = Path(source["source_root"]) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"pinned source changed: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("the declared CUDA case requires an available GPU")
    cases = [capture("cpu", torch.float32, False), capture("cpu", torch.float64, False),
             capture("cuda", torch.float32, True)]
    report = {
        "source_commit": source["repository_commit"], "source_hashes": source["files"],
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "scripts": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("backend_contract_check.py", "backend_contract_arithmetic.py")},
        "normalization": {"rms_epsilon": "1/8388608", "l2_epsilon": "1e-12"},
        "rope": {"head_dim": 4, "position": 1, "base": 10000,
                 "table_compute_dtype": "torch.float32", "real_phases": ["1", "1/100"]},
        "scope": "Exact rational premise checks for finite primitive fixtures; not universal backend or whole-model certification.",
        "cases": cases,
    }
    path = ROOT / "backend_contract_check.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "checked_sites": len(cases) * len(NAMES),
                      "report": str(path), "scope": report["scope"]}, indent=2))


if __name__ == "__main__":
    main()
