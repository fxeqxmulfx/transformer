"""Sharper local certificates for the unchanged source capture, manuscript M3.

Reuse the capture exporter for literal layout and proof-site coverage. Only
operation allowances and certificate adapters change. Lean rechecks every
result, including matrix norms, normalization gains and final propagation.
"""
from fractions import Fraction as Q
import json
from pathlib import Path
import re

from model_trace_arithmetic import check, dot, rows, SHAPES, PARAMETERS, EPS
from model_trace_lean import render as coarse_render

ROOT = Path(__file__).resolve().parents[1]
DEST = Path("src/Transformer/SequenceModel/Attention/Backend/Refined")
PREFIX = "Transformer.SequenceModel.Attention.Backend.Refined"
# Coordinate residual, resulting Euclidean allowance. Analytic adapters add
# their separately proved scale or Taylor error before the second bound.
BOUNDS = {
    "initial": ("12/100000000", "1/4000000"),
    "pre": ("1/10000000000000", "3/1000000000000"),
    "qnorm": ("2/100000000", "5/100000000"),
    "knorm": ("2/100000000", "5/100000000"),
    "direction": ("7/100000000", "15/100000000"),
    "attn_output": ("1/10000000", "21/100000000"),
    "ffn_pre": ("5/100000000", "11/100000000"),
    "ffn_output": ("1/10000000", "21/100000000"),
    "normalized": ("5/100000000", "11/100000000"),
    "fused": ("2/100000000", "8/100000000"),
    "projected": ("3/1000000000", "6/1000000000"),
    "gate": ("2/100000000", "8/100000000"),
    "value": ("2/100000000", "8/100000000"),
    "product": ("1/1000000000", "4/1000000000"),
    "ffn_projected": ("2/10000000000", "4/10000000000"),
    "mid": ("0", "0"), "output": ("6/100000000", "12/100000000"),
    "query": ("1/100000000", "2/100000000"),
    "key": ("1/100000000", "2/100000000"),
    "heads": ("4/100000000", "8/100000000"),
    "scores": ("2/1000000000", "2/1000000000"), "logits": ("0", "0"),
    "activated": ("2/100000000", "12/100000000"),
    "average": ("5/100000000", "3/10000000"),
    "cos": ("4/100000000", "1/10000000"),
    "sin": ("4/100000000", "1/10000000"),
}


def check_refined(capture):
    """Reject residuals exceeding any local box, including Q/K/V separately."""
    certificate = check(capture)
    for site, residual in certificate["maximum_rational_residuals"].items():
        if Q(residual) > Q(BOUNDS[site.split('.')[0]][0]):
            raise ValueError(f"{site}: refined residual failed")
    t = {k: rows(v, SHAPES[k]) for k, v in capture["trace"].items()}
    p = {k: rows(v, PARAMETERS[k]) for k, v in capture["parameters"].items()}
    inputs = dict(initial=[p["token_embedding_table.weight"][i] for i in capture["tokens"]],
                  pre=t["initial"], qnorm=[x[:4] for x in t["fused"]],
                  knorm=[x[4:8] for x in t["fused"]], direction=[x[8:] for x in t["fused"]],
                  attn_output=t["projected"], ffn_pre=t["mid"],
                  ffn_output=t["ffn_projected"], normalized=t["output"])
    for name, vectors in inputs.items():
        for x, a in zip(vectors, certificate["witnesses"][name], strict=True):
            square = dot(x, x) if name == "direction" else dot(x, x)/4 + EPS
            if abs(Q(a)**2 * square - 1) > Q(1, 10**12):
                raise ValueError(f"{name}: refined scale failed")
    for i in range(2):
        for c, row in enumerate(p["blocks.0.sa.c_attn.weight"]):
            allowance = Q(2, 10**12) if c < 8 else Q(2, 10**8)
            if abs(t["fused"][i][c] - dot(row, t["pre"][i])) > allowance:
                raise ValueError("component: refined residual failed")
    return certificate


def render(capture, certificate):
    """Replay the existing proof sites with separately chosen tolerances."""
    output = {}
    groups = {"NormInputs", "NormOutputs", "Linear", "Arithmetic", "Scalars", "Activations", "Softmax"}
    for path, text in coarse_render(capture, certificate).items():
        if path.stem not in groups:
            continue
        start = text.index('/-- L3:')
        body = text[start:text.rindex('\nend ')]
        blocks = re.split(r'(?=/-- L3:)', body)
        changed = []
        for block in blocks:
            if not block.strip():
                continue
            name = re.search(r'theorem checked_([a-z_0-9]+)', block).group(1)
            site = re.sub(r'_[01]$', '', name)
            if site == "tables":
                block = block.replace('rope_fixture', 'rope_local').replace('2 / 1000000', '1 / 10000000')
            else:
                eta, delta = (f"({entry.replace('/', ' / ')} : ℝ)" for entry in BOUNDS[site])
                block = block.replace('≤ 1 / 10000 :=', f'≤ {delta} :=')
                for adapter in ('rms', 'l2'):
                    block = block.replace(f'apply {adapter}_checked',
                                          f'apply {adapter}_local {eta} {delta} (by norm_num)')
                if 'apply vector_checked' in block:
                    budget = ('by have h : Real.sqrt (11 : ℝ) ≤ 4 := Real.sqrt_le_iff.mpr ⟨by norm_num, by norm_num⟩; nlinarith'
                              if site in ('gate', 'value', 'product') else
                              'by have h : Real.sqrt (12 : ℝ) ≤ 4 := Real.sqrt_le_iff.mpr ⟨by norm_num, by norm_num⟩; nlinarith'
                              if site == 'fused' else 'by norm_num')
                    block = block.replace('apply vector_checked (by decide)',
                                          f'apply vector_local {eta} {delta} (by norm_num) ({budget})')
                block = block.replace('silu_vector_checked', 'silu_vector_local').replace('average_checked', 'average_local')
            changed.append(block.replace('checked_', 'local_').replace('L3:', 'M3:'))
        header = f'''module

public import {PREFIX}.Analytic

/-!
# Refined captured {path.stem} certificates

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M3.
Generated by `./run.py model-trace-refined`. The frozen tensors and weights
are imported unchanged from Complete; these are ordinary Lean proofs.
-/

@[expose] public section
noncomputable section
namespace {PREFIX}
open GPTMini Stability Execution Complete
open scoped BigOperators
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

'''
        output[DEST / path.name] = header + ''.join(changed) + f'\nend {PREFIX}\n'
    return output


def main():
    capture = json.loads((ROOT / 'review/model_trace_capture.json').read_text())
    certificate = check_refined(capture)
    for path, text in render(capture, certificate).items():
        (ROOT / path).write_text(text)
    print('Checked 52 refined sites, scale witnesses and Q/K/V slices; exported 7 Lean modules')


if __name__ == '__main__':
    main()
