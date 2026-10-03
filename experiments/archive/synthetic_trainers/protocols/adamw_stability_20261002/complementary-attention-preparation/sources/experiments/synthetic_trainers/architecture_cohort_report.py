"""Portable six-pair outcomes: every complete failure and denominator is retained."""

import json
from pathlib import Path
import shutil
import statistics
import tempfile
from unittest.mock import patch

from . import architecture_report, stability_report
from .architecture_cohort_layout import case_manifest, validate_plan
from .attention_report import window_rows
from .persistence import PersistenceConfig
from .stability_comparison import hashes
from .stability_recovery import describe
from .stability_recovery_series import timeline


_scope = stability_report.run_scope
_analysis = architecture_report.analysis_hashes


def scope(manifest, summary):
    if manifest["stage"] == "paired_architecture_case":
        return {**summary, "scope": "one_complete_heldout_architecture_case; full_six_pair_campaign_required"}
    return _scope(manifest, summary)


def analysis_hashes():
    result = _analysis(); path = Path(__file__)
    result[str(path.relative_to(path.parent.parent.parent))] = architecture_report.digest(path)
    return result


def save_case(directory, name, destination, *, render=True):
    with patch.object(stability_report, "run_scope", scope), patch.object(architecture_report, "analysis_hashes", analysis_hashes):
        return architecture_report.save_run(directory, name, destination, render=render)


def verify_case(directory):
    with patch.object(stability_report, "run_scope", scope):
        return architecture_report.verify_archive(directory)


def read_runs(directory, plan, subdirectory):
    root = Path(directory) / subdirectory
    if {p.name for p in root.iterdir() if p.is_dir()} != {r["name"] for r in plan["recipes"]}:
        raise ValueError("Every planned architecture case, including failures, must remain archived")
    runs = {}
    for recipe in plan["recipes"]:
        path = root / recipe["name"]
        if json.loads((path / "plan.json").read_text()) != case_manifest(plan, recipe):
            raise ValueError("An architecture archive differs from its frozen case plan")
        summary = verify_case(path)
        if summary["config"] != recipe["config"]:
            raise ValueError("An architecture archive differs from its full-budget recipe")
        runs[recipe["name"]] = summary
    return runs


def derive(plan, benchmark, runs):
    validate_plan(plan, benchmark)
    if set(runs) != {r["name"] for r in plan["recipes"]}:
        raise ValueError("All twelve complete architecture runs are required")
    for recipe in plan["recipes"]:
        summary = runs[recipe["name"]]
        if not summary["complete_run"] or summary["config"] != recipe["config"]:
            raise ValueError("An architecture result differs from its complete frozen recipe")
    pairs = {}
    for pair in plan["pairs"]:
        control, candidate = runs[pair["control"]], runs[pair["candidate"]]
        if control["parameters"] != candidate["parameters"]:
            raise ValueError("The normalizer intervention must preserve the parameter count")
        pairs[pair["name"]] = architecture_report.paired_normalizer_outcome(control, candidate)
    rule = plan["improvement_rule"]
    support = sum(p["eligible_pair_support"] for p in pairs.values())
    quality = all(p["final_heldout_accuracy_difference"] >= rule["minimum_accuracy_gain"] for p in pairs.values())
    speed = support == 6 and all(p[f"control_to_candidate_{kind}_time_ratio"] >= rule["minimum_time_ratio"]
                                 for p in pairs.values() for kind in ("training", "wall"))
    statistics_by_kind = {}
    for kind in ("training", "wall"):
        values = [p[f"control_to_candidate_{kind}_time_ratio"] for p in pairs.values() if p["eligible_pair_support"]]
        statistics_by_kind[kind] = {"support": len(values), "planned_pairs": 6,
            "mean": statistics.mean(values) if values else None,
            "sample_SD": statistics.stdev(values) if len(values) > 1 else None}
    scientific = plan["scientific_run"] and not plan["CPU_fixture"]
    return {"complete_cohort_budget": True, "scientific_run": plan["scientific_run"], "CPU_fixture": plan["CPU_fixture"],
        "planned_pairs": 6, "completed_pairs": 6, "completed_runs": 12, "pairs": pairs,
        "total_updates": sum(s["config"]["steps"] for s in runs.values()),
        "total_canonical_observations": sum(s["canonical_observations"] for s in runs.values()),
        "eligible_pair_support": support, "eligible_target_time_ratio_statistics": statistics_by_kind,
        "all_pair_final_quality_rule_met": quality, "all_pair_timing_rule_met": speed,
        "repeatable_architecture_improvement": bool(scientific and support == 6 and (quality or speed)),
        "scientific_primary_benchmark_gate_verified": bool(scientific),
        "improvement_rule": rule, "all_failures_retained": True,
        "statistical_scope": plan["statistical_scope"], "parameter_initialization_pairing": plan["initialization_pairing"]}


def rows(result):
    return [{"pair": name, "role": role, **{k: v for k, v in p[role].items()
             if k not in ("memorization_plateau", "timing_scope")},
             "eligible_pair_support": p["eligible_pair_support"],
             "control_to_candidate_training_time_ratio": p["control_to_candidate_training_time_ratio"],
             "control_to_candidate_wall_time_ratio": p["control_to_candidate_wall_time_ratio"],
             "final_heldout_accuracy_difference": p["final_heldout_accuracy_difference"]}
            for name, p in result["pairs"].items() for role in ("control", "candidate")]


def recovery(directory, plan):
    result = {}
    for recipe in plan["recipes"]:
        report = json.loads((Path(directory) / "runs" / recipe["name"] / "measurements.json").read_text())
        criterion = PersistenceConfig(**plan["criterion"])
        result[recipe["name"]] = {"tail_and_episodes": describe(report, criterion, window_steps=10000),
            "whole_post_onset_series": timeline(report, criterion, window_steps=10000)}
    return {"recovery": result}


def prose(result):
    lines = ["# Complete held-out normalizer architecture cohort", "",
        "All six paired controls/candidates retain native AdamW, the selected rate path and full budgets.",
        "| Pair | Control held-out | Candidate held-out | Eligible timing support | Quality difference |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name, pair in result["pairs"].items():
        lines.append(f"| {name} | {100 * pair['control']['final_heldout_accuracy']:.4f}% | "
            f"{100 * pair['candidate']['final_heldout_accuracy']:.4f}% | {pair['eligible_pair_support']} | "
            f"{100 * pair['final_heldout_accuracy_difference']:.4f} percentage points |")
    lines += ["", f"Eligible timing support: {result['eligible_pair_support']}/6; every failure remains in the denominator.",
        f"Scientific run: {result['scientific_run']}. CPU fixture: {result['CPU_fixture']}.",
        f"Repeatable architecture improvement: {result['repeatable_architecture_improvement']}.",
        "Controls require stable grokking; candidates may retain the separate persistent-generalization phase label.",
        "Unavailable timings and post-confirmation episode rates remain null. CPU fixtures cannot open the scientific gate.",
        "Time-ratio means/sample SD are descriptive, not IID confidence intervals or universal speedups.",
        "The complete primary benchmark, source fingerprints, losses, exposure, costs and memory remain archived.", ""]
    return "\n".join(lines)


def verify_cohort(directory):
    directory = Path(directory)
    if hashes(directory) != json.loads((directory / "artifact-hashes.json").read_text())["files"]:
        raise ValueError("Architecture cohort artifact hashes differ")
    plan = json.loads((directory / "plan.json").read_text())
    result = derive(plan, directory / "benchmark", read_runs(directory, plan, "runs"))
    actual = json.loads((directory / "summary.json").read_text())
    if any(actual.get(k) != v for k, v in result.items()):
        raise ValueError("The architecture outcome differs from every complete paired trajectory")
    metrics = recovery(directory, plan)
    if json.loads((directory / "architecture-recovery.json").read_text()) != metrics:
        raise ValueError("The architecture recovery metrics differ from all complete histories")
    stability_report.verify_csv(directory / "architecture-pairs.csv", rows(result))
    stability_report.verify_csv(directory / "architecture-windows.csv", window_rows(metrics))
    if (directory / "REPORT.md").read_text() != prose(result):
        raise ValueError("Architecture prose differs from the complete outcomes")
    if actual["plots_included"] and any(not (directory / "plots" / f"{name}.{ext}").exists()
            for name in ("architecture-comparison", "architecture-recovery") for ext in ("png", "pdf")):
        raise ValueError("The cohort lacks its complete standalone comparison/recovery figures")
    return actual


def assemble(directory, destination, *, render=True):
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise FileExistsError("Architecture cohort archives require a fresh destination")
    plan = json.loads((directory / "plan.json").read_text())
    result = derive(plan, directory / "benchmark", read_runs(directory, plan, "archives"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + "-", dir=destination.parent) as temporary:
        output = Path(temporary) / "archive"; output.mkdir()
        shutil.copytree(directory / "benchmark", output / "benchmark")
        for recipe in plan["recipes"]:
            shutil.copytree(directory / "archives" / recipe["name"], output / "runs" / recipe["name"])
        stability_report.write_json(output / "plan.json", plan)
        stability_report.write_json(output / "summary.json", {**result, "plots_included": render})
        metrics = recovery(output, plan)
        stability_report.write_json(output / "architecture-recovery.json", metrics)
        stability_report.csv_rows(output / "architecture-pairs.csv", rows(result))
        stability_report.csv_rows(output / "architecture-windows.csv", window_rows(metrics))
        (output / "REPORT.md").write_text(prose(result))
        if render:
            from .architecture_cohort_plots import render_cohort
            render_cohort(output, result, metrics, plan)
        stability_report.write_json(output / "artifact-hashes.json", {"files": hashes(output)})
        verified = verify_cohort(output); output.rename(destination)
    return verified
