"""Persist resumable experiments and render the domain ranking."""

import hashlib
import json
import os
from pathlib import Path

from ..domain.ranking import summarize
from .benchmark.paths import ARCHIVE_ROOT, WORK_ROOT, archive_path
from .benchmark.provenance import audit_archive, digest, source_hashes


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class FilesystemResults:
    def __init__(self, directory=None):
        self.directory = Path(directory or WORK_ROOT / "results/combined").resolve()
        if self.directory.is_relative_to(ARCHIVE_ROOT):
            raise ValueError("Historical evidence is immutable; choose a new output directory")
        self.protocol_hash = None

    def reference_rates(self):
        audit_archive()
        rates = {}
        for suite, cohort in (("full_compile_benchmark", "rtx3050_all"), ("amsgrad_extensions_benchmark", "rtx3050")):
            path = archive_path(f"experiments/{suite}/results/{cohort}/metadata.json")
            metadata = json.loads(path.read_text())
            if fingerprint(metadata["protocol"]) != metadata["protocol_sha256"]:
                raise ValueError("Historical protocol fingerprint changed")
            for attention, values in metadata["protocol"]["selected_rates"].items():
                rates.setdefault(attention, {}).update(values)
        return rates

    def source_context(self):
        audit_archive()
        return {"active": source_hashes(), "migration_sha256": digest(ARCHIVE_ROOT.parent / "migration.json")}

    def preflight_result(self, report):
        write_json(self.directory / "preflight.json", report)

    def prepare(self, protocol):
        path = self.directory / "metadata.json"
        self.protocol_hash = fingerprint(protocol)
        if path.exists():
            stored, rows = self.load()
            if stored != protocol:
                raise ValueError("Existing sources/settings differ; choose a new output directory")
            return rows
        if (self.directory / "runs.jsonl").exists():
            raise ValueError("Raw runs have no recorded protocol")
        write_json(path, {"protocol": protocol, "protocol_sha256": self.protocol_hash})
        return []

    def load(self):
        metadata = json.loads((self.directory / "metadata.json").read_text())
        self.protocol_hash = fingerprint(metadata["protocol"])
        if metadata["protocol_sha256"] != self.protocol_hash:
            raise ValueError("Saved protocol fingerprint changed")
        raw = self.directory / "runs.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()] if raw.exists() else []
        if any(row.get("protocol_sha256") != self.protocol_hash for row in rows):
            raise ValueError("Raw runs do not belong to their saved protocol")
        return metadata["protocol"], rows

    def artifact_directory(self):
        return str(self.directory / "checkpoints")

    def append(self, row):
        if self.protocol_hash is None:
            raise ValueError("Freeze the protocol before recording a run")
        row["protocol_sha256"] = self.protocol_hash
        with (self.directory / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def report(self, rows, seeds):
        summaries = summarize(rows, seeds)
        write_json(self.directory / "summary.json", {"methods": summaries})
        lines = ["# TinyShakespeare optimizer comparison", "",
                 "Validation selects the exact best checkpoint; test is evaluated after stopping.",
                 "All usable runs enter the ranking. Completion and failures are reported.", "",
                 "| Attention | Method | Test CE | Seeds | Complete |",
                 "| --- | --- | ---: | ---: | --- |"]
        for row in summaries:
            loss = f"{row['test_loss_mean']:.6f}" if row["test_loss_mean"] is not None else "n/a"
            lines.append(f"| {row['attention']} | {row['method']} | {loss} | {row['usable_seeds']}/{row['expected_seeds']} | {row['complete']} |")
        (self.directory / "REPORT.md").write_text("\n".join(lines) + "\n")

    def validation(self, result):
        raw = self.directory / "runs.jsonl"
        write_json(self.directory / "validation.json", result | {"raw_sha256": digest(raw) if raw.exists() else None})

    def progress(self, message):
        self.directory.mkdir(parents=True, exist_ok=True)
        print(message, flush=True)
        with (self.directory / "progress.log").open("a") as stream:
            stream.write(message + "\n")
