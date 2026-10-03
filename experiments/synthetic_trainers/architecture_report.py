"""Portable normalizer archives with the complete selected native rate path.

This layer preserves every scheduled integrity check. Complete paired metrics
do not open the all-six scientific architecture-selection gate themselves.
"""

from pathlib import Path
from unittest.mock import patch

from . import scheduled_report
from .architecture_metrics import paired_outcome
from .attention_layout import digest
from .scheduled_integrity import validate_logs as scheduled_logs


_analysis_hashes = scheduled_report.analysis_hashes


def analysis_hashes():
    result = _analysis_hashes()
    path = Path(__file__)
    result[str(path.relative_to(path.parent.parent.parent))] = digest(path)
    return result


def validate_logs(report, diagnostics, probes, manifest, gradients=()):
    config = report["plan"]["config"]
    if type(config["steps"]) is not int or config.get("attention_normalization") not in ("softmax", "sparsemax"):
        raise ValueError("Architecture archives require the explicit normalizer tag and integer budget")
    required = {"experiments/synthetic_trainers/" + name for name in
                ("architecture_training.py", "sparsemax_attention.py", "scheduled_training.py", "scheduled_rates.py")}
    if not required <= manifest["training_source_hashes"].keys():
        raise ValueError("Architecture archives lack the actual model and schedule training sources")
    scheduled_logs(report, diagnostics, probes, manifest, gradients)


def save_run(directory, name, destination, *, render=True):
    with patch.object(scheduled_report, "validate_logs", validate_logs), \
            patch.object(scheduled_report, "analysis_hashes", analysis_hashes):
        return scheduled_report.save_run(directory, name, destination, render=render)


def verify_archive(directory):
    with patch.object(scheduled_report, "validate_logs", validate_logs):
        return scheduled_report.verify_archive(directory)


def paired_normalizer_outcome(control, candidate):
    """Accept only the normalizer intervention; retain failures and null ratios."""
    if (control["config"].get("attention_normalization") != "softmax"
            or candidate["config"].get("attention_normalization") != "sparsemax"):
        raise ValueError("A normalizer pair requires unchanged softmax control and sparsemax candidate")
    result = paired_outcome(control, candidate, changed_fields=("attention_normalization",))
    result["benchmark_gate_opened"] = False
    result["claim_scope"] = "one_complete_normalizer_pair; independent_benchmark_and_frozen_campaign_gates_are_external"
    return result
