"""Portable complete normalizer comparison, including failure frequency/recovery."""

import json
from pathlib import Path
import shutil
import tempfile

from .architecture_metrics import paired_outcome
from .attention_layout import validate_pair_plan
from .persistence import PersistenceConfig
from .stability_comparison import assemble, hashes, verify_comparison
from .stability_recovery import describe
from .stability_recovery_series import timeline
from .stability_report import csv_rows, verify_archive, verify_csv, write_json


def derive(directory):
    directory = Path(directory)
    plan = validate_pair_plan(json.loads((directory/"plan.json").read_text()))
    summaries, recovery = {}, {}
    for recipe in plan["recipes"]:
        name = recipe["name"]
        path = directory/"runs"/name
        summaries[name] = verify_archive(path)
        report = json.loads((path/"measurements.json").read_text())
        criterion = PersistenceConfig(**plan["criterion"])
        recovery[name] = {"tail_and_episodes":describe(report,criterion,window_steps=10000),
                          "whole_post_onset_series":timeline(report,criterion,window_steps=10000)}
    control, candidate = summaries["adamw-softmax"],summaries["adamw-sparsemax"]
    if control["parameters"] != candidate["parameters"]:
        raise ValueError("Normalization must preserve the parameter count")
    outcome = paired_outcome(control,candidate,changed_fields=["attention_normalization"])
    rule = plan["improvement_rule"]
    quality = outcome["final_heldout_accuracy_difference"] >= rule["minimum_accuracy_gain"]
    speed = bool(outcome["eligible_pair_support"] and all(
        outcome[f"control_to_candidate_{kind}_time_ratio"] >= rule["minimum_time_ratio"]
        for kind in ("training","wall")))
    return {"complete_pair":True,"scientific_run":plan["scientific_run"],"scope":plan["scope"],
            "outcome":outcome,"improvement_rule":rule,"descriptive_single_pair_quality_rule_met":quality,
            "descriptive_single_pair_timing_rule_met":speed,"campaign_architecture_claim_allowed":False,
            "independent_confirmation_complete":False,"all_failures_retained":True,"recovery":recovery,
            "statistical_scope":"one_frozen_initialization_and_split; all_six_benchmark_confirmations_and_independent_architecture_repetitions_still_required"}


def rows(result):
    table = []
    for role in ("control","candidate"):
        outcome = result["outcome"][role]
        metric = result["recovery"][outcome["name"]]["tail_and_episodes"]["metrics"]["joint"]
        table.append({"name":outcome["name"],"role":role,
            **{k:v for k,v in outcome.items() if k not in ("name","memorization_plateau","timing_scope")},
            "post_long_onset_episode_count":metric["episode_count"],
            "post_long_onset_failure_fraction":metric["post_long_onset_failure_fraction"],
            "last_failed_observation":metric["last_failed_observation"],
            "final_target_streak_start":metric["final_target_streak"]["start"],
            "final_target_streak_span":metric["final_target_streak"]["sampled_span_steps"]})
    return table


def prose(result):
    lines=["# Complete softmax/sparsemax normalizer pair","",
        "Both fresh runs retain native AdamW, GPTMini parameters, corpus, seeds and full budgets.",
        "Only attention normalization changes; causal projection follows the existing GPTMini.Convex Lean specification.","",
        "| Normalizer | Final train / held-out | Persistent tail | Tail failures | Sampled episodes after long onset | Training / wall seconds |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows(result):
        episodes=row["post_long_onset_episode_count"]
        lines.append(f"| {row['name']} | {100*row['final_train_accuracy']:.4f}% / {100*row['final_heldout_accuracy']:.4f}% | "
                     f"{row['persistent_final_performance']} | {row['tail_failures']} | "
                     f"{episodes if episodes is not None else 'No long onset'} | {row['training_seconds']:.2f} / {row['wall_seconds']:.2f} |")
    lines += ["","All budgets, canonical histories, neighboring probes, tensor diagnostics and dense gradients remain archived.",
        "Recovery metrics retain episodes, durations, censored recoveries, fixed tail windows and the whole post-onset grid.",
        "Quality/time rules are prospective descriptive rules for this single pair; the all-six benchmark and independent architecture gates remain outstanding.",
        "Candidate persistent generalization without the memorization plateau retains its separate phase label.",
        "Eligible timing requires complete persistence in both runs and stable grokking in the softmax control.",
        f"Descriptive quality rule met: {result['descriptive_single_pair_quality_rule_met']}. Descriptive timing rule met: {result['descriptive_single_pair_timing_rule_met']}.",
        f"Scope: {result['scope']}.","No repeatable architecture improvement is certified by this pair.",""]
    return "\n".join(lines)


def window_rows(result):
    table=[]
    for name,data in result["recovery"].items():
        for score,metrics in data["tail_and_episodes"]["metrics"].items():
            table.extend({"name":name,"score":score,"scope":"fixed_tail",**w} for w in metrics["fixed_tail_windows"])
        series=data["whole_post_onset_series"]["metrics"]
        if series is not None:
            for score,metrics in series.items():
                leading=metrics["leading_partial_width_window"]
                if leading is not None:
                    table.append({"name":name,"score":score,"scope":"leading_partial_width",**leading})
                table.extend({"name":name,"score":score,"scope":"fixed_post_onset",**w} for w in metrics["fixed_grid_windows"])
    return table


def verify_pair(directory):
    directory=Path(directory)
    verify_comparison(directory)
    expected=derive(directory)
    if json.loads((directory/"normalizer-pair.json").read_text()) != expected:
        raise ValueError("Paired outcome/frequency/recovery differs from the complete histories")
    verify_csv(directory/"normalizer-metrics.csv",rows(expected))
    verify_csv(directory/"normalizer-windows.csv",window_rows(expected))
    if (directory/"PAIR.md").read_text() != prose(expected):
        raise ValueError("Normalizer comparison prose differs from complete outcomes")
    plan=json.loads((directory/"plan.json").read_text())
    if plan["scientific_run"]:
        if hashes(directory/"reference") != plan["reference_hashes"]:
            raise ValueError("Archived reference evidence differs from the frozen copy")
        if verify_comparison(directory/"reference") != plan["reference_complete_summary"]:
            raise ValueError("Complete reference summary differs")
    summary=json.loads((directory/"summary.json").read_text())
    if summary["plots_included"] and any(not (directory/"plots"/f"attention-recovery.{ext}").exists() for ext in ("png","pdf")):
        raise ValueError("Paired frequency/recovery figures are missing")
    return expected


def save_pair(paths,destination,*,reference=None,render=True):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError("Portable normalizer pair requires a fresh destination")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+"-",dir=destination.parent) as temporary:
        output=Path(temporary)/"archive"
        assemble(paths,output,render=render)
        plan=json.loads((output/"plan.json").read_text())
        if plan["scientific_run"]:
            if reference is None:
                raise ValueError("Scientific pair retains its complete reference archive")
            shutil.copytree(reference,output/"reference")
        result=derive(output)
        write_json(output/"normalizer-pair.json",result)
        csv_rows(output/"normalizer-metrics.csv",rows(result))
        csv_rows(output/"normalizer-windows.csv",window_rows(result))
        (output/"PAIR.md").write_text(prose(result))
        if render:
            from .attention_plots import render_pair
            render_pair(output,result)
        write_json(output/"artifact-hashes.json",{"files":hashes(output)})
        verify_pair(output)
        output.rename(destination)
    return result
