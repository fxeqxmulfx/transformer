"""Archive verified complete calibration histories, probes, diagnostics and curves.

Archives are portable and can be verified without PyTorch, model checkpoints,
the original run directory, or the local manuscripts.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from .stability_analysis import diagnostic_metrics, summarize
from .stability_integrity import validate_logs, validate_report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_rows(blob):
    return [json.loads(line) for line in blob.decode().splitlines()]


def file_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "artifact-hashes.json"}


def analysis_hashes():
    directory = Path(__file__).parent
    repo = directory.parent.parent
    names = ("stability_integrity.py", "stability_analysis.py", "stability_report.py",
             "stability_plots.py", "persistence.py", "paper_phases.py")
    return {str((directory / name).relative_to(repo)): hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names}


def run_scope(manifest, summary):
    if manifest["stage"] == "independent_confirmation_case":
        summary["scope"] = "one_completed_independent_confirmation_run; cohort_success_requires_all_frozen_repetitions"
    return summary


def csv_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def observation_rows(observations):
    return [{**{key: point[key] for key in ("step", "epochs_seen", "training_seconds", "wall_seconds", "last_batch_size")},
             **{f"{split}_{key}": value for split in ("train", "heldout") for key, value in point[split].items()}}
            for point in observations]


def verify_csv(path, rows):
    with path.open() as stream:
        actual = list(csv.DictReader(stream))
    expected = [{key: "" if row.get(key) is None else str(row[key]) for key in
                 sorted({key for row in rows for key in row})} for row in rows]
    if actual != expected:
        raise ValueError(f"Archive {path.name} differs from measured rows")


def gradient_scope(gradients):
    if not gradients:
        return {}
    maximum = max(gradients, key=lambda row: row["gradient_l2"])
    return {"gradient_trace": {"observations": len(gradients),
            "maximum_gradient_l2": maximum["gradient_l2"], "maximum_step": maximum["step"],
            "maximum_batch_size": maximum["batch_size"],
            "scope": "every_pre_update_gradient_norm; no_causal_claim"}}


def save_run(directory, name, destination, *, render=True):
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise FileExistsError("Archive destination must be fresh")
    plan_blob = (directory / "plan.json").read_bytes()
    manifest = json.loads(plan_blob)
    recipe = next((row for row in manifest["recipes"] if row["name"] == name), None)
    if recipe is None:
        raise ValueError("Run is absent from the frozen calibration plan")
    source = directory / name
    blobs = {"plan.json": plan_blob, "measurements.json": (source / "measurements.json").read_bytes()}
    for filename in ("diagnostics.jsonl", "probes.jsonl"):
        path = source / filename
        blobs[filename] = path.read_bytes() if path.exists() else b""
    if manifest["instrumentation"].get("trace_gradients", False):
        blobs["gradients.jsonl"] = (source / "gradients.jsonl").read_bytes()
    elif (source / "gradients.jsonl").exists():
        raise ValueError("Undeclared gradient trace")
    report = json.loads(blobs["measurements.json"])
    diagnostics, probes = (parse_rows(blobs[key]) for key in ("diagnostics.jsonl", "probes.jsonl"))
    gradients = parse_rows(blobs.get("gradients.jsonl", b""))
    assessment = validate_report(report, recipe, manifest)
    validate_logs(report, diagnostics, probes, manifest, gradients)
    summary = {**run_scope(manifest, summarize(report, assessment, probes, diagnostics, name)),
               **gradient_scope(gradients), "analysis_source_hashes": analysis_hashes(), "plots_included": render}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + "-", dir=destination.parent) as temporary:
        output = Path(temporary) / "archive"
        output.mkdir()
        for filename, blob in blobs.items():
            (output / filename).write_bytes(blob)
        write_json(output / "summary.json", summary)
        csv_rows(output / "metrics.csv", observation_rows(report["history"]))
        csv_rows(output / "neighbor-metrics.csv", observation_rows(probes))
        csv_rows(output / "diagnostic-metrics.csv", [diagnostic_metrics(row) for row in diagnostics])
        if gradients:
            csv_rows(output / "gradient-metrics.csv", gradients)
        if render:
            from .stability_plots import render_run
            render_run(report, probes, diagnostics, summary, output / "plots", gradients=gradients)
        checkpoint = source / "checkpoint.pt"
        hashes = {"files": file_hashes(output), "raw_checkpoint_sha256":
                  hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None,
                  "checkpoint_scope": "weights_and_optimizer_are_local_resources; not_part_of_this_portable_archive"}
        write_json(output / "artifact-hashes.json", hashes)
        verify_archive(output)
        output.rename(destination)
    return summary


def verify_archive(directory):
    directory = Path(directory)
    hashes = json.loads((directory / "artifact-hashes.json").read_text())
    if file_hashes(directory) != hashes["files"]:
        raise ValueError("Archive file hashes differ")
    manifest = json.loads((directory / "plan.json").read_text())
    report = json.loads((directory / "measurements.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    recipe = next(row for row in manifest["recipes"] if row["name"] == summary["name"])
    diagnostics = parse_rows((directory / "diagnostics.jsonl").read_bytes())
    probes = parse_rows((directory / "probes.jsonl").read_bytes())
    gradient_path = directory / "gradients.jsonl"
    if manifest["instrumentation"].get("trace_gradients", False):
        gradients = parse_rows(gradient_path.read_bytes())
    else:
        if gradient_path.exists():
            raise ValueError("Undeclared gradient trace")
        gradients = []
    assessment = validate_report(report, recipe, manifest)
    validate_logs(report, diagnostics, probes, manifest, gradients)
    derived = {**run_scope(manifest, summarize(report, assessment, probes, diagnostics, summary["name"])),
               **gradient_scope(gradients)}
    if any(summary.get(key) != value for key, value in derived.items()):
        raise ValueError("Archive summary differs from measured histories")
    verify_csv(directory / "metrics.csv", observation_rows(report["history"]))
    verify_csv(directory / "neighbor-metrics.csv", observation_rows(probes))
    verify_csv(directory / "diagnostic-metrics.csv", [diagnostic_metrics(row) for row in diagnostics])
    if gradients:
        verify_csv(directory / "gradient-metrics.csv", gradients)
    if summary["plots_included"] and any(not (directory / "plots" / f"{name}.{extension}").exists()
            for name in ("stability-overview", "collapse-neighbors") for extension in ("png", "pdf")):
        raise ValueError("Archive lacks its declared standalone curves")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_archive(args.directory)
    else:
        if args.run is None or args.archive is None:
            parser.error("Archiving requires --run and --archive")
        result = save_run(args.directory, args.run, args.archive)
    print(json.dumps({key: result[key] for key in ("name", "complete_run", "canonical_observations", "neighbor_observations")}))


if __name__ == "__main__":
    main()
