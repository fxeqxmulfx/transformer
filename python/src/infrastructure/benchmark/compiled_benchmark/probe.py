"""Measure cold and warmed compile performance on the actual GPTMini shape."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path, DATA_FILE as _data_file, WORK_ROOT as _work_root

import gc
import os
from pathlib import Path
import time

root = _work_root
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / _bench_path('experiments/runs/inductor_cache')))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / _bench_path('experiments/runs/triton_cache')))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData, training_starts, windows
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import write_json
from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import make_optimizer, selected_rates
from .backend import compile_model, graph_counts


def measure(config, data, attention, method, compiled, steps=200, warmup=20):
    model = make_model(config, attention, 0, "cuda")
    if compiled:
        compile_model(model, attention, method)
    optimizer = make_optimizer(method, model, selected_rates()[attention][method], 0)
    starts = training_starts(len(data.train), config.max_seq_len, 32, warmup + steps, 0).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = windows(data.train, starts[index], config.max_seq_len)
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite compile probe loss")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return loss.item()

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    first_loss = step(0)
    cold_seconds = time.perf_counter() - cold_start
    for index in range(1, warmup):
        step(index)
    started = time.perf_counter()
    for index in range(warmup, warmup + steps):
        final_loss = step(index)
    seconds = time.perf_counter() - started
    result = {"attention": attention, "method": method, "compiled": compiled,
              "mode": "reduce-overhead" if compiled else "eager", "measured_steps": steps,
              "warmup_steps": warmup, "cold_first_step_seconds": cold_seconds,
              "warmed_ms_per_step": 1000 * seconds / steps, "first_loss": first_loss,
              "final_train_loss": final_loss, "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
              "cuda_graphs": graph_counts() if compiled else None,
              "compile_settings": model.compile_settings if compiled else None}
    optimizer.close()
    del optimizer, model, starts
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TextData.load(_data_file, "cuda")
    config = Config(vocab_size=len(data.characters), n_layers=2, n_heads=4,
                    d_model=128, d_ff=512, max_seq_len=64)
    directory = root / _bench_path('experiments/compiled_benchmark/results/preflight')
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for attention in ("softmax", "sparsemax"):
        for method in ("adamw", "magma_muon", "adafisher"):
            for compiled in (False, True):
                row = measure(config, data, attention, method, compiled)
                rows.append(row)
                write_json(directory / "throughput.json", rows)
                print(f"{attention} {method} {'compiled' if compiled else 'eager'} "
                      f"{row['warmed_ms_per_step']:.3f} ms/step "
                      f"cold={row['cold_first_step_seconds']:.2f}s peak={row['peak_memory_mib']:.0f}MiB "
                      f"graphs={row['cuda_graphs']}", flush=True)
    write_json(directory / "environment.json", {"torch": str(torch.__version__),
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "config": config.__dict__, "batch": 32, "dtype": "float32", "tf32": False})


if __name__ == "__main__":
    main()
