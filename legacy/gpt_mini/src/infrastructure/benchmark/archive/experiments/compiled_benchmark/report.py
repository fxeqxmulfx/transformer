"""Add compiler evidence to the unchanged parent stopping report format."""

import json
from pathlib import Path

from experiments.patience_benchmark.report import write_report as stopping_report


def write_report(directory):
    stopping_report(directory)
    path = Path(directory) / "REPORT.md"
    contents = path.read_text().replace("GPTMini comparison with validation patience",
                                       "Compiled GPTMini comparison with validation patience")
    contents = contents.replace("experiments/runs/patience_benchmark/", "experiments/runs/compiled_benchmark/")
    contents = contents.replace("-m experiments.patience_benchmark", "-m experiments.compiled_benchmark")
    rows = json.loads(Path("experiments/compiled_benchmark/results/preflight/throughput.json").read_text())
    lines = ["## Compiler and CUDA Graph evidence", "",
        "Inductor, `torch.compile(mode=\"reduce-overhead\", dynamic=False)` applied in place. "
        "Full model compilation for all methods except AdaFisher/AdaFisherW, whose current-activation "
        "and derivative hooks require partial compilation. Optimizer steps are eager. "
        "Sparsemax uses the tested capture adapter with unchanged finite-row mathematics.", "",
        "The runtime node counter checks actual CUDAGraph objects. On each of six representative "
        "attention/method cases, the 1220-step telemetry test found no new FX traces after "
        "training and the two validation shapes were compiled, and no new CUDA graph records "
        "in three subsequent train/validation cycles. AdaFisher has more graph segments from hooks. "
        "Each measured run also stores compiler snapshots after every evaluation.", "",
        "The warning `Not enough SMs to use max_autotune_gemm mode` occurred during preflight. "
        "No CUDA Graph execution error occurred in those tests or timing probes. "
        "This is preflight evidence, not a guarantee about every long optimizer trajectory.", "",
        "Training/total seconds in the ranking include cold compilation. "
        "`compile.cold_forward_seconds` separates first forward costs; the timing probe below "
        "uses 20 warmup steps and 200 timed updates on the actual model and batch.", "",
        "| Attention | Method | Eager ms/update | Compiled ms/update | Speedup |",
        "| --- | --- | ---: | ---: | ---: |"]
    for attention in ("softmax", "sparsemax"):
        for method in ("adamw", "magma_muon", "adafisher"):
            selected = [r for r in rows if r["attention"] == attention and r["method"] == method]
            eager = next(r for r in selected if not r["compiled"])
            compiled = next(r for r in selected if r["compiled"])
            e, c = eager["warmed_ms_per_step"], compiled["warmed_ms_per_step"]
            lines.append(f"| {attention} | {method} | {e:.3f} | {c:.3f} | {e / c:.2f}x |")
    lines += ["", "The compiled ranking uses its own protocol and retains all earlier "
              "fixed-budget and partial eager results separately.", ""]
    contents = contents.replace("## Scope and artifacts", "\n".join(lines) + "\n## Scope and artifacts")
    path.write_text(contents)
