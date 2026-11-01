"""Export warmed timings and non-overlapping kernel totals from raw profiles.

Python3.15 JSONL contains aggregate frame samples, not full call paths. Its
wall samples include GPU waits and startup; they are not CPU utilization.
GPU annotation ranges overlap kernels and are never added to kernel totals.
"""

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "runs" / "profiles"


def report(shape):
    plain, sampled, traced = [json.loads((RAW / f"{shape}-{mode}.json").read_text())
                              for mode in ("plain", "sampling", "torch")]
    rows = [json.loads(line) for line in (RAW / f"{shape}.jsonl").read_text().splitlines()]
    strings = {s["str_id"]: s["value"] for r in rows if r["type"] == "string_table" for s in r["strings"]}
    frames = {f["frame_id"]: f for r in rows if r["type"] == "frame_table" for f in r["frames"]}
    samples = collections.Counter()
    for row in rows:
        if row["type"] == "agg":
            assert row["kind"] == "frame" and row["scope"] == "final"
            for entry in row["entries"]:
                frame = frames[entry["frame_id"]]
                samples[strings[frame["func_str_id"]]] += entry["self"]
    total = next(r["samples_total"] for r in rows if r["type"] == "end")
    trace = json.loads((RAW / f"{shape}-cuda.json").read_text())["traceEvents"]
    kernels = collections.Counter()
    for event in trace:
        if event.get("cat") == "kernel":
            kernels[event["name"]] += event["dur"]
    kernel_us = sum(kernels.values())
    attention_us = sum(v for k, v in kernels.items() if k.startswith("fmha_cutlass"))
    optimizer = next(e for e in traced["events"] if e["name"] == "Optimizer.step#ScheduleFreeAdamW.step")
    return {
        "case": plain["case"], "source_checkpoint_sha256": plain["source_checkpoint_sha256"],
        "python": plain["python"], "torch": plain["torch"], "gpu": plain["gpu"],
        "plain": {k: v for k, v in plain.items() if k not in ("recipe", "events")},
        "sampling": {k: v for k, v in sampled.items() if k not in ("recipe", "events")},
        "sampler_wall_overhead_fraction": plain["median_updates_per_second"] / sampled["median_updates_per_second"] - 1,
        "sampling_valid_samples": total,
        "sampler_reported_error_rate": 12.69 if shape == "wide" else 8.28,
        "sampling_top_self_frames": [{"function": name, "samples": count, "fraction_of_valid": count / total}
                                      for name, count in samples.most_common(12)],
        "torch_profile_updates": 30, "kernel_time_us": kernel_us,
        "attention_kernel_time_us": attention_us, "attention_fraction_of_kernel_time": attention_us / kernel_us,
        "optimizer_annotation_device_us": optimizer["device_total_us"],
        "top_kernels": [{"name": name, "microseconds": value} for name, value in kernels.most_common(8)],
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(RAW.glob(f"{shape}*"))},
    }


def main():
    result = {"scope": "Uncontended warmed diagnostic copies; no checkpoint or learning recipe changed.",
              "limits": ["Wall-stack samples include startup and GPU waits; JSONL has aggregate frames only.",
                         "Sampler reported errors and an earlier binary collector UnicodeEncodeError are retained.",
                         "Profiler ranges overlap kernels; optimizer annotation is not added to kernel time.",
                         "CUDA traces perturb execution; plain repetitions define throughput."],
              "cases": [report(shape) for shape in ("wide", "deep")]}
    (HERE / "profile_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
