"""Task families and explicit experimental controls for the complete suite."""

from dataclasses import replace
import itertools

from .specs import CONTROL_TASKS, RASP_TASKS, RASPL_TASKS, TASKS


def task_names(trainer, mode="both"):
    prefix = ("dyck", "blocks") if mode == "both" else (mode,)
    groups = {
        "all": ("mqar", "lookup", *prefix, *RASP_TASKS, *RASPL_TASKS, "crasp"),
        "core": ("mqar", "lookup", *prefix), "prefix": prefix,
        "rasp": RASP_TASKS, "rasp-l": RASPL_TASKS,
    }
    names = groups.get(trainer, (trainer,))
    if any(name not in (*TASKS, *CONTROL_TASKS) for name in names):
        raise ValueError("Unknown trainer or prefix mode")
    return tuple(dict.fromkeys(names))


def expand_variants(spec):
    """Named controls override only the settings defining their comparison."""
    if spec.task == "histogram":
        return tuple(replace(spec, histogram_bos=bos) for bos in (True, False))
    if spec.task in ("copy", "reverse", "sort"):
        return tuple(replace(spec, unique=unique) for unique in (False, True))
    if spec.task == "mode":
        return tuple(replace(spec, scratchpad=mode) for mode in ("none", "counts", "itemized"))
    if spec.task == "parity":
        return tuple(replace(spec, scratchpad=mode, index_hints=hints) for mode, hints in
                     (("none", False), ("running", False), ("running", True), ("ones", True)))
    if spec.task == "addition":
        sampling = ("standard", "balanced") if spec.carry_length is None else (spec.carry_sampling,)
        return tuple(replace(spec, addition_order=order, index_hints=hints, carry_sampling=carry)
                     for order, hints, carry in itertools.product(("forward", "reverse"), (False, True), sampling))
    if spec.task == "boolean_and":
        return tuple(replace(spec, and_shift=shift) for shift in (True, False))
    if spec.task == "dyck2":
        return tuple(replace(spec, bracket_types=types) for types in (2, 3))
    if spec.task == "crasp":
        return tuple(replace(spec, formula_depth=depth) for depth in (1, 2, 3))
    return (spec,)
