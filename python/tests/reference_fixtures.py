"""Replay independent, immutable historical trainers on the current numerical backend.

The checked-in JSON files retain the original measurements and configurations.
Floating-point outputs and gradients are compared to the original implementation
running with this test process's PyTorch, rather than another wheel or CPU.
No reference imports lab, and missing Git history is an error, never a skip.
"""

from functools import cache
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
REFERENCES = Path(__file__).parent / "references"
REVISIONS = {
    "modular": "95f73a907cff8eec1b637c144a3b900234152b9b",
    "recall": "3bae214fd24752fa7a36a668c8cbeb6209ba0000",
    "synthetic": "253d5c6d44a51ff0a828056319bf29d8a84b161b",
    "memorization": "f5def7e425f4f8b3e911c5c72c4cc193a7d99614",
    "zoo": "83a4c81702cde67d0de94c2e7c65bb983ee1a79c",
    "text": "317db81d49ec6895ccb0c7d2f1e58bf39eb05403",
}


def export_sources(revision, destination):
    """Export tracked Python sources only, without a worktree or executable hooks."""
    entries = subprocess.check_output(
        ["git", "ls-tree", "-r", revision, "--", "experiments", "legacy/gpt_mini"],
        cwd=ROOT, text=True).splitlines()
    paths = [entry.split("\t", 1)[1] for entry in entries
             if entry.startswith("120000 ") or entry.endswith((".py", "reference-source.json", "/migration.json"))]
    paths = [path for path in paths
             if not any(part in path.split("/") for part in ("baselines", "protocols", "runs", "tests"))
             and not path.endswith("tinyshakespeare.txt")]
    archive = subprocess.check_output(["git", "archive", revision, "--", *paths], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")


@cache
def replay_result(kind, selection):
    """Cache both successful replays and reference failures for this test process."""
    with tempfile.TemporaryDirectory(prefix=f"lab-reference-{kind}-") as directory:
        directory = Path(directory)
        sources, output = directory / "source", directory / "measurements.json"
        sources.mkdir()
        export_sources(REVISIONS[kind], sources)
        environment = {**os.environ, "PYTHONPATH": str(sources)}
        command = [sys.executable, str(REFERENCES / "replay.py"), kind, str(sources),
                   str(FIXTURES / f"legacy_{kind}.json"), str(output),
                   str(ROOT / "python/src/lab/infrastructure/benchmarks/data/tinyshakespeare.txt"), selection]
        result = subprocess.run(command, cwd=sources, env=environment, capture_output=True, text=True)
        if result.returncode:
            return None, f"Independent {kind}/{selection} reference failed:\n{result.stderr[-6000:]}"
        return json.loads(output.read_text()), None


def runtime_fixture(kind, selection="*"):
    """The pinned independent implementation's measurements on this runtime."""
    value, error = replay_result(kind, selection)
    if error is not None:
        raise RuntimeError(error)
    return value


def runtime_run(kind, golden):
    """Select a replay by its original scientific configuration, not its outputs."""
    fields = ("task", "model", "training") if kind in ("synthetic", "memorization") else ("config",)
    archived = json.loads((FIXTURES / f"legacy_{kind}.json").read_text())
    matching = [name for name, run in archived["runs"].items()
                if all(run[field] == golden[field] for field in fields)]
    if len(matching) != 1:
        raise ValueError(f"Expected one independent {kind} configuration, found {len(matching)}")
    name = matching[0]
    return runtime_fixture(kind, name)["runs"][name]
