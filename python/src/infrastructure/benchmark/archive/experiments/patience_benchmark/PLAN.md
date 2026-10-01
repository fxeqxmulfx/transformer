# Validation-stopped GPTMini optimizer comparison

## Fixed stopping protocol

This experiment changes the earlier 1000-update budget into a shared
validation-based stopping rule. It uses the unchanged Tiny Shakespeare,
GPTMini, causal softmax/Sparsemax adapters, parameter routing, optimizer
recipes, seeds, and validation-selected learning rates of the previous
experiments. This is a separate experiment: do not combine its best-checkpoint
test scores with the previous fixed-budget last-iterate ranking.

- Check the entire fixed validation split every 250 updates.
- A meaningful improvement is a decrease of more than 0.0001 CE from the
  previous meaningful-improvement anchor. Small decreases accumulate against
  this anchor. Patience is 8 consecutive checks without meaningful improvement.
- Track the exact minimum independently of patience; save every strictly
  lower validation score, even when the improvement is below min_delta.
- Stop on 3 consecutive checks at least 0.1 CE above the exact best score.
- Ordinary patience/divergence stopping starts at 1000 updates. Nonfinite
  training or validation loss stops immediately, preserving the previous best.
- A 20,000-update emergency cap bounds a still-improving run. Label such a
  run `max_steps`, not plateau or convergence; report censoring explicitly.
- Restore the best validation checkpoint and evaluate test exactly once at
  the end. Also record last validation score, best step, actual steps, stop
  reason, curves, timing, peak memory, masks and safeguard acceptance.
- Same thresholds and update checks for every method, both attentions and
  seeds 0, 1, 2. Different realized lengths are part of this stopping protocol.
- Reuse earlier validation-selected learning rates without new test-based
  tuning. No scheduler, warmup, clipping, AMP, or optimizer recipe changes.

The user selected **all 24 earlier methods**, both attentions, seeds 0,1,2:
144 independently stopped runs. The CLI also supports a ten-method paired
subset, but the active experiment uses `--all`. Scope is recorded in metadata.
Execution remains eager, without torch.compile or CUDA Graphs, matching the
earlier execution mode. These settings are explicit in the new protocol.

## Work checklist

- [x] Write controller, recovery and checkpoint-selection tests first.
- [x] Implement the stopping loop outside both frozen measured packages.
- [x] Pass CPU/CUDA tests before any benchmark training.
- [x] Check old raw/source fingerprints and common initialization/batches.
- [ ] Run the chosen methods on the RTX 3050 Laptop GPU.
- [ ] Replay stopping decisions from logs and reload every best checkpoint.
- [ ] Save all runs, source/data hashes, timing, curves and the new ranking.

Output: `experiments/patience_benchmark/results/rtx3050_paired/` (default)
or `rtx3050_all/` (`--all`). All best model checkpoints and live curves are
saved under ignored `experiments/runs/patience_benchmark/`. Resume skips
completed run identifiers after verifying source/data/protocol fingerprints;
an interrupted unfinished run is restarted. Best model snapshots are for
selection/evaluation, not arbitrary within-run optimizer-state continuation.

```bash
.venv/bin/python -u -m experiments.patience_benchmark
.venv/bin/python -u -m experiments.patience_benchmark --all
```

## Execution log

- Tests were written before implementation. The initial test run failed with
  the missing controller module; all nine controller tests then passed.
- 2026-10-01 08:38 UTC: all 74 combined CPU/CUDA tests passed, no skips.
  Long streams retain the previous 1000-update minibatch prefixes. Both frozen
  measured source sets and raw logs were checked before starting.

This empirical stopping rule is not a proof of training convergence. A
best validation checkpoint can overfit the validation split through repeated
selection; the held-out test is never involved in stopping or selection.
