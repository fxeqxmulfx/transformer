"""Separate initial tracing from steady recompilation and graph rerecording."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path, DATA_FILE as _data_file, WORK_ROOT as _work_root

import gc
import os
from pathlib import Path

root = _work_root
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / _bench_path('experiments/runs/inductor_cache')))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / _bench_path('experiments/runs/triton_cache')))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch
from torch.nn import functional as F
from torch._dynamo.utils import counters

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData, training_starts, windows
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import evaluate
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import write_json
from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import make_optimizer, selected_rates
from .backend import compile_model, graph_counts


def snapshot():
    return {"unique_fx_graphs": counters["stats"]["unique_graphs"],
            "traced_calls": counters["stats"]["calls_captured"],
            "dynamo_frames": dict(counters["frames"]),
            "graph_breaks": dict(counters["graph_break"]), **graph_counts()}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TextData.load(_data_file, "cuda")
    cfg = Config(vocab_size=65, n_layers=2, n_heads=4, d_model=128, d_ff=512, max_seq_len=64)
    starts = training_starts(len(data.train), 64, 32, 1220, 0).to("cuda")
    directory = root / _bench_path('experiments/compiled_benchmark/results/preflight')
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for attention in ("softmax", "sparsemax"):
        for method in ("adamw", "magma_muon", "adafisher"):
            torch.compiler.reset()
            counters.clear()
            model = compile_model(make_model(cfg, attention, 0, "cuda"), attention, method)
            opt = make_optimizer(method, model, selected_rates()[attention][method], 0)

            def train(begin, end):
                for i in range(begin, end):
                    opt.zero_grad(set_to_none=True)
                    inputs, targets = windows(data.train, starts[i], 64)
                    loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
                    loss.backward()
                    opt.step()
                torch.cuda.synchronize()

            train(0, 20)
            warm = snapshot()
            train(20, 220)
            steady = snapshot()
            evaluate(model, data.validation, 64, 32)
            first_eval = snapshot()
            cycles = []
            for cycle in range(4):
                train(220 + 250 * cycle, 470 + 250 * cycle)
                after_train = snapshot()
                evaluate(model, data.validation, 64, 32)
                cycles.append({"cycle": cycle + 2, "after_250_train_steps": after_train,
                               "after_validation": snapshot()})
            # The first call of an inference shape warms CUDA Graphs; its
            # second call records them. Partial Fisher graphs also record the
            # train path after the initial train/inference transitions. Demand
            # stability in three subsequent full validation/train cycles.
            warm_modes = cycles[0]["after_validation"]
            stable_traces = (warm == steady and all(
                first_eval["unique_fx_graphs"] == state["unique_fx_graphs"]
                and first_eval["traced_calls"] == state["traced_calls"]
                and first_eval["graph_breaks"] == state["graph_breaks"]
                for cycle in cycles for state in cycle.values() if isinstance(state, dict)))
            stable_records = all(
                warm_modes == state for cycle in cycles[1:]
                for state in cycle.values() if isinstance(state, dict))
            stable = stable_traces and stable_records
            final = cycles[-1]["after_validation"]
            row = {"attention": attention, "method": method, "warmup_20": warm,
                   "train_200_after_warmup": steady, "first_validation": first_eval,
                   "cycles": cycles, "final": final,
                   "steady_no_new_traces": stable_traces,
                   "steady_no_new_graph_records": stable_records,
                   "steady_no_new_traces_or_graph_records": stable}
            rows.append(row)
            write_json(directory / "recompilation.json", rows)
            print(f"{attention} {method}: FX {warm['unique_fx_graphs']} -> {steady['unique_fx_graphs']} "
                  f"(200 steps), eval -> {first_eval['unique_fx_graphs']} -> {final['unique_fx_graphs']}; "
                  f"CUDA nodes {warm['recorded_graph_nodes']} -> {steady['recorded_graph_nodes']} -> "
                  f"{first_eval['recorded_graph_nodes']} -> {final['recorded_graph_nodes']}; stable={stable}", flush=True)
            opt.close()
            del opt, model
            gc.collect()
            torch.cuda.empty_cache()
    if not all(row["steady_no_new_traces_or_graph_records"] for row in rows):
        raise RuntimeError("Repeated compile/record activity remains; inspect telemetry before the long run")


if __name__ == "__main__":
    main()
