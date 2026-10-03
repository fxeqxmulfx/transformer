"""Validate active controls before generating data or allocating a model."""

import math


def validate_task(spec, minimum):
    if spec.task in ("mqar", "lookup"):
        if not 1 <= spec.queries <= spec.pairs <= spec.symbols:
            raise ValueError("Require 1 <= queries <= pairs <= symbols")
        if spec.query_gap < 0:
            raise ValueError("query_gap must be nonnegative")
        if not math.isfinite(spec.alpha) or spec.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if spec.task == "lookup" and not 1 <= spec.hops < spec.pairs:
            raise ValueError("lookup requires 1 <= hops < pairs")
        required = (1 + 2 * spec.pairs + spec.queries if spec.task == "mqar"
                    else 2 + 4 * spec.pairs + 2 * spec.queries)
        if minimum < required + spec.query_gap:
            raise ValueError(f"Input needs at least {required + spec.query_gap} tokens")
    elif spec.task in ("dyck", "dyck2", "blocks"):
        if not math.isfinite(spec.neutral_fraction) or not 0 <= spec.neutral_fraction < 1:
            raise ValueError("neutral_fraction must be in [0, 1)")
        if spec.task == "blocks" and spec.blocks < 1:
            raise ValueError("blocks must be positive")
        if spec.task in ("dyck", "dyck2") and spec.max_balance < 1:
            raise ValueError("max_balance must be positive")
        if spec.task == "dyck2" and not 2 <= spec.bracket_types <= 8:
            raise ValueError("dyck2 requires between 2 and 8 bracket types")
        if spec.max_neutral_gap is not None and spec.max_neutral_gap < 0:
            raise ValueError("max_neutral_gap must be nonnegative")
        for length in range(minimum, spec.length + 1):
            active = spec.active_size(length)
            needed = (6 if spec.task == "dyck" else 4 if spec.task == "dyck2"
                      else spec.blocks + (2 if spec.blocks > 1 else 1))
            if active < needed:
                raise ValueError(f"Too few active symbols: need at least {needed}")
            gaps = length - 1 - active
            if spec.max_neutral_gap is not None and gaps > spec.max_neutral_gap * (active + 1):
                raise ValueError("Neutral tokens do not fit max_neutral_gap")
    elif spec.task == "crasp":
        if minimum < 2 or not 1 <= spec.formula_depth <= 5 or spec.formula_seed < 0:
            raise ValueError("C-RASP needs length >= 2, depth in [1, 5], and nonnegative formula_seed")
    else:
        validate_generation(spec, minimum)


def validate_generation(spec, minimum):
    if spec.symbols < 2 or spec.number_limit < 1:
        raise ValueError("Generation needs symbols >= 2 and a positive number_limit")
    if spec.task in ("copy", "reverse", "sort") and spec.unique and spec.symbols < spec.length:
        raise ValueError("Unique-token sequences require symbols >= length, including OOD lengths")
    if spec.task == "mode":
        if spec.scratchpad not in ("none", "counts", "itemized"):
            raise ValueError("Mode scratchpad must be none, counts, or itemized")
        if minimum < 3:
            raise ValueError("Mode needs at least three symbols to construct a unique winner")
    if spec.task == "parity" and spec.scratchpad not in ("none", "running", "ones"):
        raise ValueError("Parity scratchpad must be none, running, or ones")
    if spec.task == "boolean_and" and spec.and_shift and minimum < 4:
        raise ValueError("Position-shift AND needs at least four input positions")
    if spec.task == "boolean_and" and spec.and_region not in ("early", "late"):
        raise ValueError("and_region must be early or late")
    if spec.uses_numbers and spec.length > spec.number_limit:
        raise ValueError("number_limit must cover all problem lengths, including OOD")
    if spec.task == "addition":
        if spec.addition_order not in ("forward", "reverse"):
            raise ValueError("addition_order must be forward or reverse")
        if spec.carry_sampling not in ("standard", "balanced"):
            raise ValueError("carry_sampling must be standard or balanced")
        if spec.carry_length is not None and not 0 <= spec.carry_length <= minimum:
            raise ValueError("carry_length must fit every sampled operand length")
        if spec.index_hints and spec.length + 1 > spec.number_limit:
            raise ValueError("Indexed addition needs number_limit >= length + 1")
