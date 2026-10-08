"""Interactions before loss, at actual GPTMini computation stages.

Source: Nanda et al., arXiv:2301.05217v1, appendix Further speculations
on grokking, Hypothesis: Phase Transitions are inherent to composition;
Transformer.Grokking.Composition.OutputInteractions.Basic and
NonIdentification. Ported head-removal/capture behavior: pair_interactions
and probes.capture at 6ca0b3a. Deviations: capture attention projections,
FFN projections, residuals, final normalization and answer logits at equals;
center only the class logits, before interpreting output interaction.
All pairs are enumerated. Float64 diagnostics on observed model outputs
are not exact certificates, unique circuits or training-time forecasts.
"""

from contextlib import ExitStack
from itertools import combinations
import math

import torch

from pair_interactions import metrics, remove_heads
from probes.capture import evaluating, forward


@torch.no_grad()
def observe(model, inputs, batch=512):
    """Read each named stage at equals; retain all state and remove all hooks."""
    modules = dict(model.named_modules())
    names = [name for index in range(len(model.blocks))
             for name in (f"blocks.{index}.attention.output", f"blocks.{index}.ffn.output",
                          f"blocks.{index}")]
    if model.final_norm is not None:
        names.append("final_norm")
    chunks = {name: [] for name in names}

    def keep(name):
        return lambda module, args, value: chunks[name].append(value[:, 4].detach().cpu().clone())

    with evaluating(model), ExitStack() as stack:
        for name in names:
            handle = modules[name].register_forward_hook(keep(name))
            stack.callback(handle.remove)
        logits = forward(model, inputs, batch)
    return {"logits": logits, **{name: torch.cat(values) for name, values in chunks.items()}}


@torch.no_grad()
def node_statistics(corners, scopes, *, logits=False, energy_floor=1e-12):
    """Contrast actual outputs; keep offsets, zero denominators and gaps explicit."""
    if len(corners) != 4 or len({tuple(value.shape) for value in corners}) != 1:
        raise ValueError("Four identically shaped output arrays are required")
    if any(value.ndim != 2 or not torch.isfinite(value).all() for value in corners):
        raise FloatingPointError("Observed node arrays must be finite row-by-coordinate matrices")
    z, a, b, both = (value.detach().double() for value in corners)
    raw = z - a - b + both
    delta = raw - raw.mean(-1, keepdim=True) if logits else raw
    da, db = z - a, z - b
    base = z
    if logits:
        da, db, base = (value - value.mean(-1, keepdim=True) for value in (da, db, base))
    rows = {"interaction_mean_square": delta.square().mean(-1),
            "baseline_mean_square": base.square().mean(-1),
            "single_a_change_mean_square": da.square().mean(-1),
            "single_b_change_mean_square": db.square().mean(-1)}
    maximum = delta.abs().amax(-1)
    offsets = raw.mean(-1).square() if logits else None
    result = {}
    for name, selected in scopes.items():
        record = {"examples": len(selected)}
        record.update({key: float(value[selected].mean()) if len(selected) else None
                       for key, value in rows.items()})
        if len(selected):
            if not all(math.isfinite(record[key]) for key in rows):
                raise FloatingPointError("Interaction statistics overflowed")
            denominator = record["single_a_change_mean_square"] + record["single_b_change_mean_square"]
            record.update(interaction_rms=math.sqrt(record["interaction_mean_square"]),
                          interaction_max_abs=float(maximum[selected].max()),
                          interaction_over_singles_energy=(record["interaction_mean_square"] / denominator
                                                          if denominator > energy_floor else None),
                          common_row_offset_rms=(math.sqrt(float(offsets[selected].mean()))
                                                 if offsets is not None else None))
        else:
            record.update(interaction_rms=None, interaction_max_abs=None,
                          interaction_over_singles_energy=None, common_row_offset_rms=None)
        result[name] = record
    return {"class_centered": logits, "coordinates": z.shape[-1], "scopes": result}


@torch.no_grad()
def reconstruction_metrics(corners, targets, scopes):
    """Fixed-baseline competitor margins and a hypothetical additive output."""
    z, a, b, both = (value.detach().double() for value in corners)
    reconstruction = a + b - both
    result = metrics(reconstruction, targets, scopes, z)
    competitors = z.clone()
    competitors.scatter_(1, targets[:, None], -torch.inf)
    other = competitors.argmax(-1)
    rows = torch.arange(len(z))
    delta = z - reconstruction
    margin = delta[rows, targets] - delta[rows, other]
    for name, selected in scopes.items():
        result[name]["target_margin_interaction_mean"] = float(margin[selected].mean()) if len(selected) else None
        result[name]["target_margin_interaction_rms"] = float(margin[selected].square().mean().sqrt()) if len(selected) else None
    return result


@torch.no_grad()
def measure(model, inputs, targets, train, heldout, prime, heads, batch=512, energy_floor=1e-12):
    """All pairs and stages; label use is confined to scoring current outputs."""
    if prime < 3 or heads < 1 or len(inputs) != prime * (prime - 1) or len(targets) != len(inputs):
        raise ValueError("A positive head count and complete quotient-ordered division rows are required")
    scopes = {f"{name}_{part}": selected if part == "all" else selected[selected >= prime - 1]
              for name, selected in (("train", train), ("heldout", heldout))
              for part in ("all", "nonzero")}
    addresses = [(block, head) for block in range(len(model.blocks)) for head in range(heads)]

    def label(address):
        return f"blocks.{address[0]}.head{address[1]}"

    with evaluating(model):
        baseline = observe(model, inputs, batch)
        singles, pairs = {}, []
        for address in addresses:
            with remove_heads(model, [address], heads):
                singles[address] = observe(model, inputs, batch)
        for a, b in combinations(addresses, 2):
            with remove_heads(model, [a, b], heads):
                both = observe(model, inputs, batch)
            corners = {name: (baseline[name], singles[a][name], singles[b][name], both[name])
                       for name in baseline}
            nodes = {name: node_statistics(values, scopes, logits=name == "logits", energy_floor=energy_floor)
                     for name, values in corners.items()}
            pairs.append({"a": label(a), "b": label(b), "same_block": a[0] == b[0],
                          "nodes": nodes,
                          "additive_reconstruction": reconstruction_metrics(corners["logits"], targets, scopes)})
    return {"selection": "all_heads_and_all_unordered_pairs; all_named_stages_at_equals",
            "removal": "all_prompt_positions; merged_head_after_XSA_before_projection",
            "objective": "answer_only_ce; EOS_excluded",
            "precision": f"float64_statistics_of_observed_{baseline['logits'].dtype}_outputs; not_certified",
            "centering": "class_logits_only; intermediate_features_unshifted",
            "normalization": "interaction_energy_over_sum_of_single_effect_energies; floor_retains_None",
            "energy_floor": energy_floor, "head_count": len(addresses),
            "baseline": metrics(baseline["logits"], targets, scopes, baseline["logits"]),
            "pairs": pairs}
