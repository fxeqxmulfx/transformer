# Raw AMSGradW stability and architecture follow-up

This loop starts on 2026-10-02 from the completed nine-run modular campaign.
The local manuscript is *Convexifying Transformers*, arXiv:2211.11052v1,
Section 4. The experiment retains its mod-97 division task and explicitly
different GPTMini architecture and raw AMSGradW optimizer. The double-descent
control remains the completed *Deep Double Descent* Appendix C random-feature
archive; rerunning it is not needed to inspect the modular instability.

## Prospective criteria

The existing two-observation onset, legacy memorization diagnostic, epoch-error
shape, and final accuracy remain separate measurements. Fresh follow-up plans
also freeze [PersistenceConfig](persistence.py):

- Memorization requires train accuracy at least 99% and exhaustive held-out
  accuracy at most 10% for at least five consecutive canonical observations
  spanning at least 1,000 updates, before the first held-out 99% crossing.
- Long confirmation requires both train and held-out accuracy at least 99%
  for 20 consecutive canonical observations (4,750 updates at cadence 250).
- Persistent final performance requires both scores at least 99% at every
  canonical observation over the last 50,000 updates, including the final one.
  A 150,000-update run has 201 such observations at cadence 250.
- Stable grokking requires the complete budget/history, the memorization
  plateau, later long confirmation, and persistent final performance.

This criterion concerns scheduled finite observations. It makes no guarantee
between evaluations or beyond the budget. Rapid, persistent generalization
without the low-held-out plateau is reported as such. Tail failures and final
rebounds remain failures of persistence.

Applied retrospectively, no old confirmation run passes this stronger tail
criterion. Reference/AdamW seeds 1/2/3 each have three failing tail observations;
GPTMini/AdamW has 6/3/4; GPTMini/raw AMSGradW has 16/148/13. These new criteria
do not alter the historical frozen targets or their recorded successes.

## Frozen calibration

Four exploratory runs share data seed 0, initialization seed 0, the exhaustive
4,656/4,656 split, width 128, two layers, four heads, 423,816 parameters,
float32, batch 512, decay 0.1, ten-update warmup, and 150,000 updates.
All use raw AMSGradW betas (0.9, 0.999), epsilon 1e-8, no bias correction,
all-parameter decoupled decay, and no clipping or schedule change.

| Recipe | Learning rate | Batch policy | Changed mechanism |
| --- | ---: | --- | --- |
| `short-lr001` | 0.001 | Original short epoch tail | Unchanged-model control |
| `wrap-lr001` | 0.001 | Full batch across shuffled epochs | Sampling policy only |
| `short-lr0003` | 0.0003 | Original short epoch tail | Learning rate only |
| `short-lr0001` | 0.0001 | Original short epoch tail | Learning rate only |

All four record the same instrumentation: per-tensor gradients, weights, actual
updates, moment norms, learned inverse temperatures, batch/cursor state, and
numeric-answer/EOS losses every 250 updates and at its immediate neighbors.
Canonical exhaustive evaluations remain every 250 updates, plus step zero and
the final step. Neighbor evaluations are separate probes and never certify
targets. The sampling control consumes more examples at fixed updates; actual
exposure is retained. Measurements do not establish why any collapse occurs.

The driver writes an immutable `plan.json` before training and a separate
`state.json` for progress. It freezes recipes, criteria, instrumentation,
portable source hashes, local manuscript hashes, split fingerprints, and the
actual environment. Resume rejects changed sources, settings, or environment.
Completed runs are verified against complete histories and exhaustive scoring;
unsuccessful targets are retained. Only one process can execute a given stage.

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability \
  --output experiments/runs/amsgradw_stability_20261002/calibration
```

Use the same command with `--resume` after a checkpointed interruption.
`--plan-only` freezes without starting training; executing that plan requires
`--resume`. Every scheduled budget completes regardless of target crossings.

After each complete run, archive it separately before the next campaign result
is available. The archive contains its unchanged measurements, all diagnostic
and neighbor logs, canonical/neighbor/diagnostic CSV, an assessment and collapse
neighborhood report, standalone PNG/PDF curves, and artifact checksums. Example:

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability_report \\
  experiments/runs/amsgradw_stability_20261002/calibration \\
  --run short-lr001 \\
  --archive experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr001_seed0_data0_20261002
python3 -m experiments.synthetic_trainers.stability_report \\
  experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr001_seed0_data0_20261002 --verify
```

The destination must be fresh. Verification checks the budget, exact scheduled
supports, exhaustive split scoring, declared batch exposure and warmup, metric
and norm consistency, checksums, derived assessment, and CSV rows. It runs
without PyTorch or the original data/checkpoints. A default complete run has
601 canonical observations, 1,200 neighbor probes, and 1,800 diagnostic records.
Checkpoint hashes are recorded, with weights/optimizer kept as local resources.
Single completed calibrations do not complete the four-run stage or certify
independent confirmation. Neighbor categories can overlap and describe observed
associations; they do not establish the cause of a collapse.

## Environment and next decisions

The restored environment initially had Python 3.12.13 and PyTorch 2.7.1+cu118.
Before calibration it was upgraded to PyTorch 2.14.0+cu126, NumPy 2.5.3, and
Matplotlib 3.11.2. CUDA smoke training passed on a GTX 1050 with 2 GiB VRAM.
The official [PyTorch packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
identifies 2.14/CUDA 12.6 as the last published build for Pascal. Install with:

```bash
uv pip install --python .venv/bin/python 'torch==2.14.0' \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python 'numpy==2.5.3' 'matplotlib==3.11.2'
```

The old manuscripts and self-contained archives are available. Historical
model/optimizer checkpoints and Fashion-MNIST IDX files were not transferred.
Fresh modular training rebuilds its complete oracle corpus and uses the tracked,
licensed reference model; these missing historical weights/data do not block
the current calibration. Timing comparisons use paired controls on this GPU.

Finish and archive all four calibration trajectories, diagnose neighbor/tail
effects and norm changes, and select a passing recipe before confirmation.
Any new fraction, decay, schedule, or model control gets a separately frozen
plan. If all four fail, preserve that result and freeze a justified controlled
follow-up without relaxing the success criterion.

Independent confirmation will exclude calibration model/data seed 0: use model
seeds 4/5/6 crossed with data seeds 2/3 in a new frozen plan. Require stable
grokking in all six to declare the benchmark repeatable; report every failure.
The architecture comparison follows after reproducibility. Freeze a targeted
change justified by diagnostics, retain an unchanged-model paired control,
and compare complete-budget quality and measured time to the predefined
target, including support, persistence, parameters, and peak memory.
Extend that comparison to MQAR, copy/parity, lookup/composition, and prefix
counting with independently frozen seeds, pools, targets, and budgets. Report
novel-ID and length transfer separately. Preserve all curves, negative results,
source/data hashes, and full reports. Commit verified logical changes during
the loop and update the handoff with measured outcomes and the next action.
