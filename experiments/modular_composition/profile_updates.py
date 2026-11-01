"""Profile declared source-model updates after warmup without altering a run.

Source: the unchanged lab CompiledStepper at33e412cb; Python3.15
profiling.sampling documentation; torch.profiler. Select a declared case
and one of plain/sampling/torch. The diagnostic copies its update2000
checkpoint, performs100 warmup updates, then benchmarks identical restored
states. It never writes back to a learning checkpoint or measures accuracy
for model selection. Raw artifacts live in the ignored profile directory.
"""

import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).parent
ARTIFACTS = HERE / "runs" / "profiles"
CASES = {"wide": "pairs-division-p97-w256d1", "deep": "pairs-division-p97-w128d4"}


def synchronize_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def steady_updates(stepper, sampler, count, rate):
    """The stable frame to select in a Tachyon flame graph."""
    started = synchronize_time()
    for _ in range(count):
        parts, _ = sampler.next()
        stepper.step(parts, rate, False)
    ended = synchronize_time()
    stepper.gradient_norms()
    return ended - started


def main():
    shape, mode = sys.argv[1:]
    if shape not in CASES or mode not in ("plain", "sampling", "torch"):
        raise ValueError("Select wide/deep and plain/sampling/torch")
    torch.set_num_threads(1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    label = CASES[shape]
    experiment = load(HERE).experiments[label]
    run = RunDirectory(HERE / "runs" / label)
    checkpoint = run.checkpoint("cuda")
    if checkpoint is None or checkpoint["step"] != 2000:
        raise ValueError("The declared source must have completed its early screen")
    task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cuda"))
    model = build_model(experiment.model, task.vocab, experiment.seeds.model).cuda()
    model.load_state_dict(checkpoint["model"])
    stepper = CompiledStepper(experiment, task, model, None)
    stepper.load_state_dict(checkpoint["optimizer"])
    sampler = task.sampler(experiment.budget.batch, experiment.seeds.batch_seed)
    sampler.restore(checkpoint)
    started = synchronize_time()
    stepper.prepare(sampler.sizes())
    preparation = synchronize_time() - started
    # Compilation/evaluation must not change the selected query or optimizer history.
    model.load_state_dict(checkpoint["model"])
    stepper.load_state_dict(checkpoint["optimizer"])
    warmup = steady_updates(stepper, sampler, 100, experiment.optimizer.lr)
    saved = (copy.deepcopy(model.state_dict()), copy.deepcopy(stepper.state_dict()), copy.deepcopy(sampler.state()))

    def restore():
        model.load_state_dict(saved[0])
        stepper.load_state_dict(copy.deepcopy(saved[1]))
        sampler.restore(copy.deepcopy(saved[2]))

    records = []
    events = []
    if mode in ("plain", "sampling"):
        for _ in range(3):
            restore()
            records.append(steady_updates(stepper, sampler, 2000, experiment.optimizer.lr))
    else:
        restore()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True) as prof:
            for _ in range(30):
                with record_function("complete_schedule_free_update"):
                    parts, _ = sampler.next()
                    stepper.step(parts, experiment.optimizer.lr, False)
                prof.step()
            torch.cuda.synchronize()
        stepper.gradient_norms()
        prof.export_chrome_trace(str(ARTIFACTS / f"{shape}-cuda.json"))
        for item in prof.key_averages():
            events.append({"name": item.key, "calls": item.count,
                           "self_cpu_us": item.self_cpu_time_total,
                           "self_device_us": item.self_device_time_total,
                           "cpu_total_us": item.cpu_time_total, "device_total_us": item.device_time_total})
        (ARTIFACTS / f"{shape}-cuda-table.txt").write_text(prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=40) + "\n")
    restore()
    started = synchronize_time()
    stepper.evaluate("heldout")
    evaluation = synchronize_time() - started
    result = {"case": label, "mode": mode, "recipe": describe(experiment),
              "source_checkpoint_sha256": hashlib.sha256((run.path / "checkpoint.pt").read_bytes()).hexdigest(),
              "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "prepare_seconds": preparation, "warmup100_seconds": warmup,
              "updates_per_repeat": 2000, "repeat_seconds": records,
              "median_updates_per_second": 2000 / statistics.median(records) if records else None,
              "heldout_evaluation_seconds": evaluation, "events": events}
    (ARTIFACTS / f"{shape}-{mode}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("events", "recipe")}), flush=True)


if __name__ == "__main__":
    main()
