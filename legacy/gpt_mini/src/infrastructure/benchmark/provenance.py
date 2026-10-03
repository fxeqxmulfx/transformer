"""Audit historical snapshots and fingerprint the active migrated package."""

import hashlib
import subprocess

from .paths import ARCHIVE_ROOT, MIGRATION, PACKAGE_ROOT, PROJECT_ROOT, TEST_ROOT, archive_path, recorded_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    result = {}
    root = PACKAGE_ROOT.parents[1]
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "archive" not in relative.parts and "__pycache__" not in relative.parts:
            result["gpt_mini/" + relative.as_posix()] = digest(path)
    for path in sorted(TEST_ROOT.rglob("*.py")):
        result["gpt_mini/infrastructure/benchmark/tests/" + path.relative_to(TEST_ROOT).as_posix()] = digest(path)
    test_base = TEST_ROOT.parent if PROJECT_ROOT else root / "tests"
    for path in sorted(test_base.glob("*.py")):
        result["gpt_mini/tests/" + path.name] = digest(path)
    for name in MIGRATION["measured_sources"]:
        if name.startswith("src/"):
            result["proof/" + name] = digest(archive_path(name))
    return result


def source_snapshot_matches(expected):
    if expected == source_hashes():
        return True
    if any(name.startswith(("gpt_mini/", "proof/")) for name in expected):
        return False
    try:
        return bool(expected) and all(digest(archive_path(name)) == value for name, value in expected.items())
    except (OSError, ValueError):
        return False


def artifact_snapshot_matches(expected):
    try:
        return bool(expected) and all(digest(recorded_path(name)) == value for name, value in expected.items())
    except OSError:
        return False


def audit_archive():
    expected = MIGRATION["archived_files"] | MIGRATION["measured_sources"]
    for name, value in expected.items():
        if digest(archive_path(name)) != value:
            raise ValueError(f"Preserved benchmark evidence changed: {name}")
    return len(expected)


def git_head():
    if PROJECT_ROOT is None:
        return None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
