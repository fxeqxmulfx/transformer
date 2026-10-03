# Grokking x / y mod 97

Does a small transformer generalize long after it fits x / y mod 97, and
does GPTMini do the same under AdamW and under raw AMSGradW? The setting is
that of *Convexifying Transformers* (arXiv:2211.11052v1, §4), whose standard
transformer fits the training equations in about 1,000 updates and
generalizes after more than 100,000. These are the 2026-10-02 runs of
`paper_reproduction.grokking` of the synthetic trainers, in the lab's
language.

## Runs

An equation is `a / b = c` in the field of 97 elements, and both `c` and
the end token are predicted. A run trains on a fraction of the 9,312
equations, in batches of 512, for 150,000 updates, and is evaluated on all
the others every 250 updates. The rate 1e-3 is warmed up over 10 updates.

The reference is the openai/grok transformer: width 128, two post-norm
layers of four heads, LayerNorm, sinusoidal positions, ReLU, an untied
readout and PyTorch's initialization, under AdamW with betas (0.9, 0.98).
GPTMini is the [reference GPTMini](../archive/gpt_mini/gpt_mini.py) at the
same width and depth: pre-norm with RMSNorm, fused QKV, QK normalization,
XSA, RoPE, ReLU² and a tied readout, initialized at Normal(0.02).

| Label | Model | Optimizer | Train | Decay | Seeds (data / model) |
| --- | --- | --- | --- | --- | --- |
| `fraction20-wd1` | reference | AdamW | 20% | 1 | 0 / 0 |
| `fraction50-wd1` | reference | AdamW | 50% | 1 | 0 / 0 |
| `fraction50-wd01` | reference | AdamW | 50% | 0.1 | 0 / 0 |
| `reference-adamw-seed{1,2,3}` | reference | AdamW | 50% | 0.1 | 1 / 1, 2, 3 |
| `gptmini-adamw-seed{1,2,3}` | GPTMini | AdamW | 50% | 0.1 | 1 / 1, 2, 3 |
| `gptmini-amsgradw-seed{1,2,3}` | GPTMini | raw AMSGradW, betas (0.9, 0.999) | 50% | 0.1 | 1 / 1, 2, 3 |

The first three calibrate the reference; the other nine confirm the one
calibration that passed its launch condition, on a new split. Eager
execution issues the updates as the historical trainer did.

## Running

```sh
./make.py check experiments/mod97_grokking              # the runs, and what differs between them
./make.py run experiments/mod97_grokking [label ...]    # train every run, or the labeled ones
./make.py report experiments/mod97_grokking [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The 2026-10-02 runs are in the synthetic trainers' records:
[`paper_reproduction/MODULAR.md`](../archive/synthetic_trainers/paper_reproduction/MODULAR.md)
describes the protocol, and `baselines/mod97_*` in
[their baselines](../archive/synthetic_trainers/baselines) hold the
measurements. A run fits, or generalizes, at the first of two consecutive
evaluations at 99% train, or held-out, accuracy; a plateau is a stretch of
fitted training equations with held-out accuracy at most 10% before it
generalizes.

| Runs | Fit at | Generalized at | At the end |
| --- | --- | --- | --- |
| `fraction20-wd1` | 500 | never | held-out accuracy 0.018 |
| `fraction50-wd1` | 3,000 | 4,000 | held-out accuracy 0.61 |
| `fraction50-wd01` | 750 | 31,500, after a plateau | 1.0 |
| `reference-adamw-seed1/2/3` | 750 / 750 / 1,000 | 34,750 / 38,000 / 62,500, each after a plateau | at 99% |
| `gptmini-adamw-seed1/2/3` | 10,250 / 1,000 / 500 | 49,750 after a plateau / 1,250 / 1,000 | at 99% |
| `gptmini-amsgradw-seed1/2/3` | 3,000 / 5,000 / 7,500 | 21,000 / 40,500 / 41,750 | seed 2 below 99% |

The reference grokked in every confirmation run. GPTMini under AdamW
generalized within 500 updates of fitting in two of three runs, and under
raw AMSGradW without a plateau. None of these runs is the stable grokking
the later [stability protocols](../mod97_stability) ask for: in its last
50,000 updates every reference confirmation falls below 99% at three
evaluations, GPTMini under AdamW at 6, 3 and 4, and under raw AMSGradW at
16, 148 and 13 ([`STABILITY.md`](../archive/synthetic_trainers/STABILITY.md)).
