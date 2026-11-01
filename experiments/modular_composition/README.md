# Mixed modular arithmetic in the normalized RoPE transformer

Can the actual normalized source model learn `a^3+ab mod P`, and can it
learn a sequence of `z <- z^3+b*z+c mod P` updates from the final answer?
Board #138 implements and measures this first screen; #139 tracks an
efficient construction that uses depth in the Lean source model.
The existing exponential-width finite-table construction is only a
sufficient universality bound. It is not a necessary width for arithmetic
or other structured tasks. This screen keeps each architecture's width
fixed across all chain lengths. The next construction should reuse local
operations with polynomial resources, ideally fixed width at fixed prime
and growing depth, while accounting for routing and numerical precision.

Protocol was frozen before training; the screen is in progress. The
exponent-one division control has a rank-one multiplicative-character
route. For the mixed task, the additive-character matrix is exactly flat;
minimum-nuclear-norm completion fails on every proper observation pattern.
Neither assertion predicts the convergence of the optimizer below.

| Cases | Count | Difference |
| --- | ---: | --- |
| Pairwise screen | 12 | Mixed or division; P17/97/193; W256 D1 or W128 D4 |
| Initialization confirmation | 4 | Mixed P97/P193, both shapes; model seed17 |
| Full-table fitting | 3 | Mixed P17/97/193, W256 D1 |
| Composed chains | 24 | P17/97/193; L1/2/4/8; both shapes |
| Optimizer sensitivity | 4 | Mixed P97 W256 D1; decay0 or rate.001; seeds11/17 |
| Compiler/timing pilot | 1 | 100 updates; excluded from learning comparisons |

Every main case runs 2000 updates, batch256, with Schedule-Free AdamW
lr.003, betas(.9,.98), eps1e-8, decay.1, and ten external warmup updates.
Only the named sensitivity cases change rate or decay. Evaluation uses
averaged weights; no early stopping or held-out checkpoint selection.
Seeds are model11/data11/batches11 except the named model17 confirmations.
Python3.15, Torch2.15 and `torch.compile(mode="max-autotune", fullgraph=True,
dynamic=False)` are required. CUDA optimizer steps remain eager.

The model ports `diff_gpt/domain/model` at b98c665b, pinned by
`review/sequence_model_source.json`: exact geometric RoPE, parameter-free
embedding/final and Peri-RMSNorm, per-head Q/K RMSNorm, causal softmax, XSA
including the explicit first-position zero, SwiGLU, and tied embedding.
All four heads use RoPE in every block. The source initialization is
retained, and vocabularies are unpadded. W256 D1 and W128 D4 approximately
match trainable parameter counts; their head dimensions, compute and
training seconds differ and must be reported. This is not a compute-matched
depth theorem or a comparison with the quarter-turn pooling model.

Pairs use half of the exhaustive table, shuffled with SplitMix64. Division
excludes zero denominators; mixed arithmetic includes them. Full-table
controls have no held-out split. Chains use 2048 unique training inputs
and 1024 disjoint same-length held-out inputs, sampled uniformly without
replacement. The input is `[z0,b0,c0,...]`; only the final answer is
supervised. Independent uniform `c` makes the full-domain output uniform.
An additional 1024 unique inputs at length2L are evaluated once on the
final model, within the preconfigured 33-token RoPE context. These results
do not measure generalization to new primes or unbounded lengths.
Report the majority-class and copy-last-input baselines on every split.
For chains, copying the final c succeeds on a fraction (2P-1)/P^2 of the
uniform full domain: z=0 permits every b; each nonzero z permits b=-z^2.
Fresh uniform c keeps every intermediate state uniform. Thus the copy
baseline is about11.42% at P17, 2.05% at P97 and1.03% at P193, rather than
1/P. The measured fixed-sample baselines are retained separately.

The committed `hypotheses.json` separates exact spectral checks from
exploratory learning comparisons. Success means final accuracy at least99%;
all best/final accuracies, failures, parameter counts, sample hashes and
runtime are retained. Finite-budget failure is not an impossibility result.
Any follow-up motivated by these outcomes requires a new declared protocol.

Validation before training: three CPU cases match the pinned source's
initial weights, logits and all gradients bit for bit. Independent float64
equations check every parameter gradient. A CUDA max-autotune comparison
has maximum logit error7.45e-8 and gradient error3.50e-7. Tests also check
exhaustive pair labels, independent Horner chain labels, split uniqueness,
answer isolation and exact CPU Schedule-Free checkpoint continuation.
`spectral_checks.json` records all three primes: the mixed Gram errors are
below9e-14 and the division second singular values below4.5e-13. These
numerical checks accompany the existing exact Lean results; they are not
new proofs of learning or whole-chain spectral behavior.

Setup checks: `./run.py test` ran443 lab tests successfully (three skips),
then104 optimizer tests with the two already recorded ScheduleFreeMuon
CUDA-allocation failures (float32/float64,75 allocations of8519680 bytes).
Schedule-Free AdamW passed. After adding the copying baseline, the affected
eight-test arithmetic suite passed. The mandatory command's overall exit
status remains nonzero; these existing failures are not relabeled as passes.

## Falsification and profiling follow-up

`falsification.json` records explicit claims to challenge. It was written
after the early screen began and before any long continuation. The exact
spectral results are retained even if a learning interpretation fails.
`recovery_patterns.py` checks the actual division training splits for the
connected graph and perfect matching needed by ideal character recovery,
and records integer witnesses and faithful-character phase checks.

`profile_updates.py` copies the frozen P97 division checkpoints for W256D1
and W128D4. After compilation and100 warmup updates, plain and sampling
modes each measure three2000-update intervals from the same restored state.
The torch mode records30 complete updates with CPU/CUDA events. Evaluation
is timed separately. Nothing is written back to the learning checkpoints.
These copies are diagnostics, not additional learning outcomes.

Use Python3.15's [sampling profiler](https://docs.python.org/3.15/library/profiling.sampling.html)
to launch the diagnostic worker; inspect the `steady_updates` frame rather
than attributing compilation to ordinary updates. The sampled-versus-plain
wall-time difference estimates profiler overhead. Python stack samples do
not measure GPU kernels; the PyTorch CUDA trace supplies that measurement.
Run from `python/`, substituting the fixed `wide` or `deep` case:

```sh
uv run --locked python ../experiments/modular_composition/profile_updates.py wide plain
uv run --locked python -m profiling.sampling run --jsonl -o ../experiments/modular_composition/runs/profiles/wide.jsonl ../experiments/modular_composition/profile_updates.py wide sampling > ../experiments/modular_composition/runs/profiles/wide-sampling.log 2>&1
uv run --locked python ../experiments/modular_composition/profile_updates.py wide torch
```

Raw profiles stay in the root-linked ignored `runs/profiles/`. Their hashes
and portable measurements are in `profile_summary.json`, regenerated with
`python3 experiments/modular_composition/report_profiles.py`.

| Warmed measurement | W256 D1 | W128 D4 |
| --- | ---: | ---: |
| Plain median updates/second, three2000-update repetitions | 292.31 | 133.68 |
| Python sampling overhead relative to plain wall time | 1.73% | 0.50% |
| Cached compilation/preparation, seconds | 1.91 | 4.19 |
| Full P97 heldout evaluation, seconds | .048 | .083 |
| Attention forward/backward share of GPU kernel time | 51.16% | 70.97% |

The30-update CUDA traces show attention kernels dominate, especially at
four layers. Optimizer device annotations cover12.27/15.58ms respectively;
they overlap kernel events and are not added to kernel totals. Wall-stack
samples spend49.16%/63.81% of valid samples in batch input transfer, which
includes waiting for queued GPU work: this is not evidence that Python data
arithmetic consumes that much CPU. Device-resident/pinned index transfer and
short-context attention kernels are candidates for a separately validated
performance change. Do not replace the attention backend or update ordering
inside the frozen long cohort. Evaluation costs about5.6%/4.5% of a250-update
interval for both train and heldout at this prime, before other observations.
Long trajectories amortize compilation better than many short fresh runs.

The Python3.15 binary collector failed with a UnicodeEncodeError on a
surrogate-containing frame; the failure artifact is retained. JSONL worked.
The JSONL traces retain24454/49517 valid samples. The original report copied
error rates12.69% and8.28% into the exporter, but the console logs needed to
verify them were not retained. Self-review found and removed those constants:
the portable report now marks these historical rates as missing evidence.
Future runs retain the console log and parse its actual percentage; an absent
log produces `null`, never zero or a previous run's rate. Coverage cannot be
quantified from the retained JSONL alone. JSONL provides aggregate frames, so these figures
cover the whole worker, including startup, not an isolated CPU-only steady
region. Plain timed repetitions define throughput; the CUDA trace supplies
kernel attribution. These profiles were completed before long training.

## Frozen early outcomes (2000 updates)

All48 declared runs completed, including one100-update timing pilot excluded
from learning conclusions. The original CUDA capture failure for the P193
deep mixed case and its unchanged successful retry are both retained.
`results.json` includes every recorded learning curve, final result, parameter
count, runtime, split/checkpoint hash and replay comparison.

| Task | Prime | W256 D1 heldout | W128 D4 heldout |
| --- | ---: | ---: | ---: |
| mixed | 17 | 4.828% | 6.897% |
| mixed | 97 | 1.467% | 1.700% |
| mixed | 193 | 0.881% | 0.838% |
| division | 17 | 5.882% | 8.088% |
| division | 97 | 4.725% | 2.255% |
| division | 193 | 100.000% | 2.062% |

Training accuracy is100% in these cases except the two P193 deep models,
which each miss one training row. The full-table mixed controls fit100% at
P17/P97 but only58.62% at P193. The four model-seed17 confirmations and
four P97 rate/decay controls remain below2% heldout; see the complete JSON.
These are fixed finite-budget observations, not impossibility results.

| Chain prime | Updates in chain | W256 D1 heldout / doubled length | W128 D4 heldout / doubled length |
| --- | ---: | ---: | ---: |
| 17 | 1 | 6.93% / 7.81% | 8.59% / 7.52% |
| 17 | 2 | 8.59% / 6.64% | 8.01% / 7.32% |
| 17 | 4 | 6.54% / 6.93% | 6.54% / 6.15% |
| 17 | 8 | 8.11% / 7.03% | 6.74% / 6.54% |
| 97 | 1 | 2.34% / 1.76% | 1.07% / 1.37% |
| 97 | 2 | 1.37% / 0.68% | 1.86% / 1.66% |
| 97 | 4 | 0.88% / 0.68% | 1.27% / 1.37% |
| 97 | 8 | 0.98% / 1.76% | 0.88% / 1.66% |
| 193 | 1 | 1.17% / 1.27% | 1.07% / 1.07% |
| 193 | 2 | 1.27% / 0.29% | 0.78% / 0.78% |
| 193 | 4 | 0.98% / 0.39% | 0.68% / 0.29% |
| 193 | 8 | 1.46% / 0.88% | 1.07% / 0.39% |

No chain reaches the99% criterion. Compare with the measured copy-last and
majority baselines in each split; the uniform copy baseline is not1/P.
All pair labels, source correspondence and faithful-character recovery graph
premises pass their independent checks. The connected/perfect-matching
division splits do not imply that sampled CE training discovers that kernel.

All48 final saved states replay eagerly on CUDA. Eight cases with discrepancies
also replay through the original compiled path; tolerances were not widened.
Two compiled comparisons retain a one-answer difference: model17 P193 deep
mixed training, and P17 one-step deep chain heldout. The latter loss differs
by.007345, exceeding the declared3e-4 relative/absolute tolerance. Saving
training queries after evaluation adds an FP32 mode roundtrip on replay;
that is a possible contributor, not an isolated causal diagnosis. Every
99% success classification agrees. The reports preserve both measured values;
they do not claim bitwise-equivalent saved-state metrics.

The long continuation is a separate adaptive cohort and does not replace
these endpoints. Its protocol was frozen before starting, after profiling.
## Residual and numerical follow-up

The fixed-batch residual audit is in `residual_diagnostics.json`; it includes
all parameter gradients and separates repeated-token ambient gradients from
their sum over shared positions. `precision_diagnostics.json` records the
original versus centered forward comparison. Centering alone subsequently
failed the full parameter-gradient audit. `parameter_precision.json` retains
all six original/centered/guarded FP32/FP64 variants on all eight P97/P193
checkpoints, using the actual diff_gpt source. The guarded common-prefix path
fixes the measured failure. These diagnostic copies never alter frozen runs.

The exact algebra and full-depth prefix equalities are proved in Lean's
`SequenceModel.Attention.CenteredXSA`, `ConstantTokens`, and `ConstantPrefixes`.
The proof's clipping guard matters; it is not a floating-point error bound or
a learning guarantee. The source fix is diff_gpt commit `5b84da8`. To reproduce
its CPU diagnostic from this repository's `python/` directory:

```sh
PYTHONPATH=/home/misha/python_projects/gpt/src CUDA_VISIBLE_DEVICES='' \
  uv run --locked python ../experiments/modular_composition/parameter_precision.py
```

The [long continuation](../modular_composition_long/README.md) retains the
original arithmetic and compares the same early and late residual batches.
