"""All four losses for each pair of actual attention-head removals.

Source: Nanda et al., arXiv:2301.05217v1, appendix Further speculations on
grokking, Hypothesis: Phase Transitions are inherent to composition; Lean
Transformer.Grokking.Composition.StationarySaddle at b141ba0. Deviations:
measure ordinary GPTMini heads at all prompt positions, after XSA and
before their output projection, instead of identifying a scalar bilinear
score with an attention circuit. Answer-only CE is measured on both data
splits, separately retaining and excluding the trivial zero quotient.
Removal contrasts measure sensitivity; their signs are not unique circuit
identification, a numerical certificate, or a forecast of generalization.
"""

from contextlib import contextmanager, ExitStack
from itertools import combinations

import torch
from torch.nn import functional as F

from probes.ablation import remove_head
from probes.capture import evaluating, forward


@contextmanager
def remove_heads(model, selected, heads):
    """Validate all addresses before installing temporary removal hooks."""
    selected = tuple(selected)
    if heads < 1 or len(set(selected)) != len(selected):
        raise ValueError("Head count must be positive and head addresses distinct")
    for block, head in selected:
        if not 0 <= block < len(model.blocks) or not 0 <= head < heads:
            raise ValueError("Head address is outside the model")
        if model.blocks[block].attention.output.weight.shape[1] % heads:
            raise ValueError("Merged attention width must be divisible by head count")
    with ExitStack() as stack:
        for block, head in selected:
            stack.enter_context(remove_head(model.blocks[block].attention, head, heads))
        yield


def corners_contrast(both_present, a_removed, b_removed, both_removed):
    """Raw four-corner arithmetic, with a declared heuristic sign tolerance."""
    a, b = a_removed - both_present, b_removed - both_present
    interaction = both_present - a_removed - b_removed + both_removed
    tolerance = 1e-9 + 1e-7 * max(abs(value) for value in
                                (both_present, a_removed, b_removed, both_removed))
    return {"interaction": interaction, "single_a_loss_increase": a,
            "single_b_loss_increase": b,
            "joint_loss_increase": both_removed - both_present,
            "both_single_removals_hurt": a > tolerance and b > tolerance,
            "negative_interaction": interaction < -tolerance,
            "sign_tolerance": tolerance}


def metrics(logits, targets, scopes, baseline):
    """Keep empty scopes explicit; do not fabricate losses for them."""
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Head-intervention logits must be finite")
    result = {}
    for name, selected in scopes.items():
        if len(selected) == 0:
            result[name] = {"examples": 0, "loss": None, "accuracy": None,
                            "prediction_change_fraction": None}
            continue
        x, y = logits[selected].double(), targets[selected]
        prediction = x.argmax(-1)
        result[name] = {"examples": len(selected), "loss": float(F.cross_entropy(x, y)),
                        "accuracy": float((prediction == y).double().mean()),
                        "prediction_change_fraction": float(
                            (prediction != baseline[selected].argmax(-1)).double().mean())}
    return result


@torch.no_grad()
def measure(model, inputs, targets, train, heldout, prime, heads, batch=512):
    """Enumerate every unordered pair without fitting or selecting on labels."""
    if len(inputs) != prime * (prime - 1) or len(targets) != len(inputs):
        raise ValueError("Complete quotient-ordered division rows are required")
    scopes = {f"{name}_{part}": selected if part == "all" else selected[selected >= prime - 1]
              for name, selected in (("train", train), ("heldout", heldout))
              for part in ("all", "nonzero")}
    addresses = [(block, head) for block in range(len(model.blocks)) for head in range(heads)]

    def label(address):
        return f"blocks.{address[0]}.head{address[1]}"

    with evaluating(model):
        baseline = forward(model, inputs, batch)
        base_metrics = metrics(baseline, targets, scopes, baseline)
        singles, pairs = {}, []
        for address in addresses:
            with remove_heads(model, [address], heads):
                changed = forward(model, inputs, batch)
            singles[address] = metrics(changed, targets, scopes, baseline)
        for a, b in combinations(addresses, 2):
            with remove_heads(model, [a, b], heads):
                changed = forward(model, inputs, batch)
            corners = {"both_present": base_metrics, "a_removed": singles[a],
                       "b_removed": singles[b],
                       "both_removed": metrics(changed, targets, scopes, baseline)}
            contrast = {}
            for name in scopes:
                values = [value[name]["loss"] for value in corners.values()]
                contrast[name] = corners_contrast(*values) if values[0] is not None else None
            pairs.append({"a": label(a), "b": label(b), "corners": corners,
                          "contrasts": contrast})
    return {"objective": "answer_only_ce_at_equals; EOS_excluded",
            "removal": "all_prompt_positions; merged_head_after_XSA_before_projection",
            "selection": "all_heads_and_all_unordered_pairs",
            "zero_quotient": "first_prime_minus_one_rows; explicit_nonzero_scopes",
            "sign_tolerance": "heuristic_1e-9_plus_1e-7_max_abs_corner_loss; not_certified",
            "baseline": base_metrics, "head_count": len(addresses),
            "singles": {label(address): value for address, value in singles.items()},
            "pairs": pairs}
