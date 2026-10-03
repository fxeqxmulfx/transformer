"""Freeze new recipes together with their measured and Lean dependencies."""

import hashlib
import json
from pathlib import Path

from experiments.full_compile_benchmark.protocol import source_hashes as frozen_sources
from experiments.optimizer_benchmark.storage import read_results


ROOT = Path("experiments/amsgrad_extensions_benchmark")
BASELINE = Path("experiments/full_compile_benchmark/results/rtx3050_all")
METHODS = ("amsgradw", "amsgradmd", "amsgradmd_guarded")
RATES = {"amsgradw": (.0003,.001,.003), "amsgradmd": (.0003,.001,.003),
         "amsgradmd_guarded": (.1,.3,1.)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    result = frozen_sources()
    paths = list(ROOT.rglob("*.py"))
    for directory in ("AMSGrad", "AMSGradW", "MagnitudeDirection", "Optimization"):
        paths += list((Path("src/Transformer") / directory).rglob("*.lean"))
        paths.append(Path("src/Transformer") / f"{directory}.lean")
    for path in sorted(paths):
        result[str(path)] = digest(path)
    return result


def baseline():
    meta = json.loads((BASELINE / "metadata.json").read_text())
    audit = json.loads((BASELINE / "validation.json").read_text())
    if meta["protocol"]["source_hashes"] != frozen_sources():
        raise ValueError("Completed all24 measured sources changed")
    if digest(BASELINE / "runs.jsonl") != audit["raw_sha256"] or not audit["complete"]:
        raise ValueError("Completed all24 measurements changed")
    return read_results(BASELINE / "runs.jsonl"), meta["protocol"]


def prior_artifacts():
    return {str(BASELINE / name):digest(BASELINE / name)
            for name in ("metadata.json","runs.jsonl","summary.json","validation.json")}


def preflight_artifacts():
    directory = ROOT / "results/preflight"
    return {str(directory / name):digest(directory / name)
            for name in ("tests.log","tests.json","lean_build.log","lean_axioms.log",
                         "lean_individual_axioms.log")}


def check_initial(row, earlier):
    old = next(r for r in earlier if r["attention"] == row["attention"]
               and r["method"] == "amsgrad" and r["seed"] == row["seed"])
    if old["initial_sha256"] != row["initial_sha256"]:
        raise ValueError("New run does not share the existing initial model")
