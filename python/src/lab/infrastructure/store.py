"""Run directories: one per labeled experiment, at `runs/<label>/` in the folder of its experiment.

    experiment.json    the description, and one segment per training session
    experiment.py      the experiment file as of the latest session
    history.jsonl      canonical observations; probes.jsonl, their neighbors
    diagnostics.jsonl  sampled per-tensor measurements
    gradients.jsonl    the gradient norm of every update
    attention.jsonl    fixed-validation attention observations, including update zero
    checkpoint.pt      model, optimizer and sampler state at the last checkpoint
    result.json        the summary, written when the budget is reached
"""

import hashlib
import json
from pathlib import Path
import time

import torch

STREAMS = ("history", "probes", "diagnostics", "gradients", "attention")


def replace(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def line(row):
    return json.dumps(row, allow_nan=False) + "\n"


class RunDirectory:
    def __init__(self, path):
        self.path = Path(path)

    def manifest(self):
        return read_json(self.path / "experiment.json")

    def description(self):
        manifest = self.manifest()
        return None if manifest is None else manifest["description"]

    def provenance(self):
        manifest = self.manifest()
        return None if manifest is None else manifest["segments"][-1]["provenance"]

    def result(self):
        return read_json(self.path / "result.json")

    def begin(self, description, provenance, source):
        manifest = self.manifest()
        if manifest is None:
            if self.path.exists() and any(self.path.iterdir()):
                raise FileExistsError(f"{self.path} holds files but no experiment.json")
            manifest = {"segments": []}
        self.path.mkdir(parents=True, exist_ok=True)
        manifest["description"] = description
        manifest["segments"].append({
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "updates": description["budget"]["updates"],
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "provenance": provenance})
        (self.path / "result.json").unlink(missing_ok=True)
        replace(self.path / "experiment.py", source)
        replace(self.path / "experiment.json", json.dumps(manifest, indent=2) + "\n")

    def checkpoint(self, device):
        path = self.path / "checkpoint.pt"
        return torch.load(path, map_location=device, weights_only=True) if path.exists() else None

    def save(self, checkpoint):
        temporary = self.path / "checkpoint.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(self.path / "checkpoint.pt")

    def records(self, stream):
        path = self.path / f"{stream}.jsonl"
        return [json.loads(text) for text in path.read_text().splitlines()] if path.exists() else []

    def record(self, stream, row):
        with (self.path / f"{stream}.jsonl").open("a") as file:
            file.write(line(row))

    def rewind(self, step):
        """Drop every record of an update after `step`."""
        for stream in STREAMS:
            path = self.path / f"{stream}.jsonl"
            if path.exists():
                replace(path, "".join(line(row) for row in self.records(stream) if row["step"] <= step))

    def finish(self, result):
        replace(self.path / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")


class RunDirectories:
    """The runs of one study, a directory per label under `root`."""

    def __init__(self, root):
        self.root = Path(root)

    def open(self, label):
        return RunDirectory(self.root / label)
