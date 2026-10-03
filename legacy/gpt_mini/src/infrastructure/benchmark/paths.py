"""Locate packaged inputs and separate writable benchmark outputs."""

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = PACKAGE_ROOT / "archive"
DATA_FILE = PACKAGE_ROOT / "data/tinyshakespeare.txt"
MIGRATION = json.loads((PACKAGE_ROOT / "migration.json").read_text())
ORIGINAL_ROOT = Path(MIGRATION["original_workspace"])
PROJECT_ROOT = next((p for p in PACKAGE_ROOT.parents if (p / "pyproject.toml").is_file()
                    and (p / "src/infrastructure/benchmark").is_dir()), None)
WORK_ROOT = Path(os.environ.get("GPT_MINI_BENCHMARK_HOME", PROJECT_ROOT or
                               Path.cwd() / "gpt-mini-benchmark")).expanduser().resolve()
TEST_ROOT = (PROJECT_ROOT / "tests/benchmark" if PROJECT_ROOT else PACKAGE_ROOT / "tests")


def logical_path(value):
    """Retain the original metadata names when locating archived evidence."""
    path = Path(value)
    return path.relative_to(ORIGINAL_ROOT) if path.is_absolute() and path.is_relative_to(ORIGINAL_ROOT) else path


def archive_path(value):
    path = logical_path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a preserved relative path: {value}")
    return ARCHIVE_ROOT / path


def benchmark_path(value):
    """Translate former output locations without writing into the archive."""
    path = Path(value)
    if path.parts[:2] == ("experiments", "runs"):
        return WORK_ROOT / "runs" / Path(*path.parts[2:])
    if path.parts[:1] == ("experiments",):
        return WORK_ROOT / "results" / Path(*path.parts[1:])
    return path


def artifact_path(value):
    """Resolve legacy checkpoint names and current caller-supplied paths."""
    logical = logical_path(value)
    if logical.parts[:2] == ("experiments", "runs"):
        return benchmark_path(logical)
    return Path(value)


def recorded_path(value):
    logical = logical_path(value)
    if logical.parts[:2] == ("experiments", "runs"):
        return benchmark_path(logical)
    if logical.parts[:1] in (("experiments",), ("scripts",), ("src",)):
        return archive_path(logical)
    return Path(value)
