"""Portable tagged confirmation, preserving all-six gates and actual recovery histories."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from . import confirmation_report as original
from . import scheduled_report
from . import stability_report
from .attention_report import window_rows
from .persistence import PersistenceConfig
from .stability_comparison import hashes
from .stability_recovery import describe
from .stability_recovery_series import timeline
from .tagged_confirmation_layout import case_manifest,validate_plan


_derive=original.derive
_markdown=original.markdown_report


def archive_driver(directory):
    config=json.loads((Path(directory)/"plan.json").read_text())["recipes"][0]["config"]
    return scheduled_report if "learning_rate_schedule" in config else stability_report


def save_run(directory,name,destination,*,render=True):
    return archive_driver(directory).save_run(directory,name,destination,render=render)


def verify_archive(directory):
    return archive_driver(directory).verify_archive(directory)


def derive(plan,calibration,runs):
    with patch.object(original,"validate_plan",validate_plan):
        summary=_derive(plan,calibration,runs)
    summary.update(scientific_run=plan["scientific_run"],CPU_fixture=plan["CPU_fixture"],training_mode=plan["training_mode"])
    summary["repeatable_stable_benchmark"] &= plan["scientific_run"]
    summary["architecture_comparison_ready"] &= plan["scientific_run"]
    if plan["CPU_fixture"]:
        summary["next_action"]="retain_CPU_pipeline_fixture; no_scientific_learning_claim"
    return summary


def markdown(summary):
    return _markdown(summary)+f"\nScientific run: {summary['scientific_run']}. Explicit CPU fixture: {summary['CPU_fixture']}.\n"


@contextmanager
def context():
    with patch.object(original,"validate_plan",validate_plan),patch.object(original,"case_manifest",case_manifest),\
            patch.object(original,"verify_archive",verify_archive),patch.object(original,"derive",derive),\
            patch.object(original,"markdown_report",markdown):
        yield


def recovery(directory,plan):
    criterion=PersistenceConfig(**plan["criterion"])
    result={}
    for recipe in plan["recipes"]:
        report=json.loads((Path(directory)/"runs"/recipe["name"]/"measurements.json").read_text())
        result[recipe["name"]]={"tail_and_episodes":describe(report,criterion,window_steps=10000),
            "whole_post_onset_series":timeline(report,criterion,window_steps=10000)}
    return {"recovery":result}


def verify_confirmation(directory):
    directory=Path(directory)
    with context():
        result=original.verify_confirmation(directory)
    plan=json.loads((directory/"plan.json").read_text())
    expected=recovery(directory,plan)
    if json.loads((directory/"confirmation-recovery.json").read_text())!=expected:
        raise ValueError("Tagged confirmation recovery differs from all complete histories")
    stability_report.verify_csv(directory/"confirmation-windows.csv",window_rows(expected))
    if result["plots_included"] and any(not (directory/"plots"/f"confirmation-recovery.{ext}").exists() for ext in ("png","pdf")):
        raise ValueError("Tagged confirmation lacks complete recovery figures")
    return result


def assemble(directory,destination,*,render=True):
    directory,destination=Path(directory),Path(destination)
    if destination.exists():
        raise FileExistsError("Tagged confirmation archive requires a fresh destination")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+"-",dir=destination.parent) as temporary:
        output=Path(temporary)/"archive"
        with context():
            original.assemble(directory,output,render=render)
        plan=json.loads((output/"plan.json").read_text());metrics=recovery(output,plan)
        stability_report.write_json(output/"confirmation-recovery.json",metrics)
        stability_report.csv_rows(output/"confirmation-windows.csv",window_rows(metrics))
        if render:
            from .tagged_confirmation_plots import render_confirmation
            render_confirmation(output,metrics,plan)
        stability_report.write_json(output/"artifact-hashes.json",{"files":hashes(output)})
        result=verify_confirmation(output);output.rename(destination)
    return result


def verified_benchmark(directory):
    result=verify_confirmation(directory)
    if (not result["scientific_run"] or result["CPU_fixture"] or not result["repeatable_stable_benchmark"]
            or not result["architecture_comparison_ready"] or result["planned_runs"]!=6
            or result["stable_grokking_runs"]!=6 or result["optimizers"]!=["adamw"]):
        raise ValueError("Architecture selection requires all six scientific primary confirmations to pass")
    return result
