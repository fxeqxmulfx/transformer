"""Portable complete schedule calibration; descriptive costs are not architecture claims."""

import json
from pathlib import Path
import shutil
import tempfile

from .architecture_metrics import timing_outcome
from .attention_report import verify_pair as verify_reference
from .persistence import PersistenceConfig
from .scheduled_layout import validate_pair_plan
from .scheduled_report import assemble, verify_archive, verify_comparison
from .stability_comparison import hashes
from .stability_recovery import describe
from .stability_recovery_series import timeline
from .stability_report import csv_rows, verify_csv, write_json


def derive(directory):
    directory=Path(directory)
    plan=validate_pair_plan(json.loads((directory/"plan.json").read_text()))
    cases,recovery={},{}
    for recipe in plan["recipes"]:
        name=recipe["name"]
        path=directory/"runs"/name
        summary=verify_archive(path)
        report=json.loads((path/"measurements.json").read_text())
        cases[name]={"name":name,**timing_outcome(summary,control=True),
            "parameters":summary["parameters"],"config":summary["config"],
            "final_train_accuracy":summary["final"]["train"]["accuracy"],
            "final_heldout_accuracy":summary["final"]["heldout"]["accuracy"],
            "final_heldout_answer_loss":summary["final"]["heldout"]["answer_loss"],
            "final_heldout_EOS_loss":summary["final"]["heldout"]["EOS_loss"],
            "epochs_seen":summary["final"]["epochs_seen"],
            **{key:summary[key] for key in ("training_seconds","wall_seconds","diagnostic_seconds",
                "peak_cuda_allocated_bytes","peak_cuda_reserved_bytes")}}
        criterion=PersistenceConfig(**plan["criterion"])
        recovery[name]={"tail_and_episodes":describe(report,criterion,window_steps=10000),
                       "whole_post_onset_series":timeline(report,criterion,window_steps=10000)}
    control,candidate=cases["adamw-constant"],cases["adamw-cosine-tail"]
    if control["parameters"] != candidate["parameters"]:
        raise ValueError("The schedule must preserve the original softmax parameter count")
    eligible=bool(control["eligible_timing_support"] and candidate["eligible_timing_support"])
    costs={f"control_to_candidate_{kind}_time_ratio":(
        control[f"eligible_target_{kind}_seconds"]/candidate[f"eligible_target_{kind}_seconds"] if eligible else None)
        for kind in ("training","wall")}
    return {"complete_pair":True,"scientific_run":plan["scientific_run"],"scope":plan["scope"],
        "cases":cases,"eligible_pair_support":int(eligible),**costs,
        "eligible_scientific_calibration_recipes":[name for name,case in cases.items()
            if plan["scientific_run"] and case["stable_grokking"]],
        "final_heldout_accuracy_difference":candidate["final_heldout_accuracy"]-control["final_heldout_accuracy"],
        "campaign_architecture_claim_allowed":False,"independent_confirmation_complete":False,
        "all_failures_retained":True,"recovery":recovery,
        "interpretation":"fixed_optimizer_schedule_adaptation; changing_LR_also_changes_per_update_decoupled_decay; no_isolated_cause_or_architecture_claim",
        "statistical_scope":"one_frozen_initialization_and_split; new_six_case_confirmation_required_for_every_selected_recipe"}


def rows(result):
    table=[]
    for name,case in result["cases"].items():
        joint=result["recovery"][name]["tail_and_episodes"]["metrics"]["joint"]
        table.append({**{k:v for k,v in case.items() if k not in ("config","memorization_plateau","timing_scope")},
            "learning_rate_schedule":case["config"]["learning_rate_schedule"],
            "post_long_onset_episode_count":joint["episode_count"],
            "post_long_onset_failure_fraction":joint["post_long_onset_failure_fraction"],
            "last_failed_observation":joint["last_failed_observation"],
            "final_target_streak_start":joint["final_target_streak"]["start"],
            "final_target_streak_span":joint["final_target_streak"]["sampled_span_steps"]})
    return table


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
                table.extend({"name":name,"score":score,"scope":"fixed_post_onset",**w}
                             for w in metrics["fixed_grid_windows"])
    return table


def prose(result):
    lines=["# Complete native AdamW fixed-schedule calibration","",
        "Both fresh softmax cases retain the same model, optimizer, corpus, seeds and complete budget.",
        "Only the declared fixed schedule changes. Lower learning rates also reduce per-update decoupled decay; their contributions are not isolated.","",
        "| Schedule | Final train / held-out | Persistent tail | Tail failures | Post-onset episodes | Training / wall seconds |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows(result):
        episodes=row["post_long_onset_episode_count"]
        lines.append(f"| {row['name']} | {100*row['final_train_accuracy']:.4f}% / {100*row['final_heldout_accuracy']:.4f}% | "
            f"{row['persistent_final_performance']} | {row['tail_failures']} | "
            f"{episodes if episodes is not None else 'No long onset'} | {row['training_seconds']:.2f} / {row['wall_seconds']:.2f} |")
    lines += ["","Every failed target, actual learning rate, dense gradient, neighbor probe and native tensor diagnostic remains archived.",
        "Both joint and held-out recovery scores retain sampled durations, censoring, tail windows, the leading partial window and the whole post-onset grid.",
        "Timing ratios require complete stable grokking in both cases. All measured costs remain visible when ratios are ineligible.",
        "A scientific passing calibration still requires six fresh crossed confirmations. CPU fixtures provide no scientific eligibility.",
        f"Eligible scientific recipes: {', '.join(result['eligible_scientific_calibration_recipes']) or 'None'}.",
        f"Scope: {result['scope']}.","No architecture improvement or repeatable benchmark is certified by this calibration.",""]
    return "\n".join(lines)


def verify_pair(directory):
    directory=Path(directory)
    verify_comparison(directory)
    expected=derive(directory)
    if json.loads((directory/"schedule-pair.json").read_text()) != expected:
        raise ValueError("Schedule outcome/frequency/recovery differs from the complete histories")
    verify_csv(directory/"schedule-metrics.csv",rows(expected))
    verify_csv(directory/"schedule-windows.csv",window_rows(expected))
    if (directory/"PAIR.md").read_text() != prose(expected):
        raise ValueError("Schedule comparison prose differs from complete outcomes")
    plan=json.loads((directory/"plan.json").read_text())
    if plan["scientific_run"]:
        if hashes(directory/"reference") != plan["reference_hashes"]:
            raise ValueError("Complete normalizer reference differs from the frozen copy")
        if verify_reference(directory/"reference") != plan["reference_complete_outcome"]:
            raise ValueError("Complete normalizer outcome differs")
    summary=json.loads((directory/"summary.json").read_text())
    if summary["plots_included"] and any(not (directory/"plots"/f"schedule-recovery.{ext}").exists() for ext in ("png","pdf")):
        raise ValueError("Actual schedule/frequency/recovery figures are missing")
    return expected


def save_pair(paths,destination,*,reference=None,render=True):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError("Portable schedule pair requires a fresh destination")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+"-",dir=destination.parent) as temporary:
        output=Path(temporary)/"archive"
        assemble(paths,output,render=render)
        plan=json.loads((output/"plan.json").read_text())
        if plan["scientific_run"]:
            if reference is None:
                raise ValueError("Scientific scheduling retains the full normalizer reference")
            shutil.copytree(reference,output/"reference")
        result=derive(output)
        write_json(output/"schedule-pair.json",result)
        csv_rows(output/"schedule-metrics.csv",rows(result))
        csv_rows(output/"schedule-windows.csv",window_rows(result))
        (output/"PAIR.md").write_text(prose(result))
        if render:
            from .scheduled_plots import render_pair
            render_pair(output,result)
        write_json(output/"artifact-hashes.json",{"files":hashes(output)})
        verify_pair(output)
        output.rename(destination)
    return result
