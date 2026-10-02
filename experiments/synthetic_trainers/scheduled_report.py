"""Portable scheduled archives with unchanged records and native moment checks."""

from pathlib import Path
from unittest.mock import patch

from . import stability_report as original
from .attention_layout import digest
from .scheduled_integrity import validate_logs


_original_analysis_hashes = original.analysis_hashes


def analysis_hashes():
    result = _original_analysis_hashes()
    for name in ("scheduled_rates.py", "scheduled_integrity.py", "scheduled_report.py"):
        path = Path(__file__).parent / name
        result[str(path.relative_to(path.parent.parent.parent))] = digest(path)
    return result


def save_run(directory, name, destination, *, render=True):
    with patch.object(original, "validate_logs", validate_logs), patch.object(original, "analysis_hashes", analysis_hashes):
        return original.save_run(directory, name, destination, render=render)


def verify_archive(directory):
    with patch.object(original, "validate_logs", validate_logs):
        return original.verify_archive(directory)
