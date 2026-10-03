"""Execute every frozen independent repeat, preserving successful and failed targets."""

import argparse
import fcntl
import json
from pathlib import Path

from .confirmation_layout import case_manifest
from .confirmation_protocol import freeze, resume, verify_live
from .stability import run_stage
from .stability_report import save_run, verify_archive, write_json


def run_confirmation(directory, plan, *, progress=None):
    directory = Path(directory)
    verify_live(directory, plan)
    completed = []
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this confirmation") from error
        for recipe in plan["recipes"]:
            verify_live(directory, plan)
            name = recipe["name"]
            state = {"status": "running", "planned_runs": plan["planned_runs"],
                     "completed_runs": len(completed), "current_run": name, "runs": completed}
            write_json(directory / "state.json", state)
            case = directory / "cases" / name
            archive = directory / "archives" / name
            try:
                run_stage(case, case_manifest(plan, recipe), progress=progress)
                if archive.exists():
                    summary = verify_archive(archive)
                    if json.loads((archive / "plan.json").read_text()) != case_manifest(plan, recipe):
                        raise ValueError("Existing confirmation archive has a different frozen plan")
                    for filename in ("measurements.json", "diagnostics.jsonl", "probes.jsonl"):
                        if (archive / filename).read_bytes() != (case / name / filename).read_bytes():
                            raise ValueError("Existing confirmation archive differs from the raw run")
                else:
                    summary = save_run(case, name, archive, render=plan["archive_plots"])
            except Exception as error:
                write_json(directory / "state.json", {**state, "status": "failed",
                           "error": f"{type(error).__name__}: {error}"})
                raise
            completed.append({"name": name, "assessment": summary["assessment"],
                              "archive": f"archives/{name}"})
            if progress:
                progress({"event": "completed_confirmation_case", **completed[-1]})
        state = {"status": "complete", "planned_runs": plan["planned_runs"],
                 "completed_runs": len(completed), "current_run": None, "runs": completed,
                 "repeatable_stable_benchmark": all(row["assessment"]["stable_grokking"] for row in completed)}
        write_json(directory / "state.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--recipe")
    parser.add_argument("--model-seeds", type=int, nargs="+")
    parser.add_argument("--data-seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.resume:
        plan = resume(args.output)
        if args.calibration is not None or args.recipe is not None:
            parser.error("Resume uses the existing frozen selection; omit --calibration and --recipe")
        for key in ("model_seeds", "data_seeds"):
            if getattr(args, key) is not None and getattr(args, key) != plan[key]:
                parser.error("Resume seed lists must match the frozen confirmation")
    else:
        if args.calibration is None or args.recipe is None:
            parser.error("A complete calibration comparison and passing recipe are required")
        plan = freeze(args.output, args.calibration, args.recipe,
                      model_seeds=args.model_seeds or (4, 5, 6), data_seeds=args.data_seeds or (2, 3))
    if not args.plan_only:
        run_confirmation(args.output, plan, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
