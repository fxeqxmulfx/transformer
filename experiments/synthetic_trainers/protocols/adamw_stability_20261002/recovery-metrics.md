# Sampled failure frequency and recovery

The user requested recovery and decreasing failure frequency as additional
metrics on 2026-10-02, then specified a maximum of 300,000 training updates
per run. These are descriptive follow-ups to the modular setting in
*Convexifying Transformers*, Section 4. They preserve the existing phase,
last-50,000-update persistence and independent-confirmation gates.

[stability_recovery.py](../../stability_recovery.py) works without PyTorch.
It requires the exact scheduled canonical prefix, finite accuracies and a
matching final observation. It reports both joint train/held-out accuracy and
held-out accuracy separately, using the frozen 99% target.

- Every failed canonical observation remains counted. Consecutive failed
  observations after the first long-confirmation onset form one episode.
- Each episode retains its first/last failed observation, minimum accuracy,
  threshold deficit, first observed recovery and sampled recovery interval.
  An ongoing episode has no invented recovery time.
- Fixed 10,000-update windows partition the frozen final 50,000 updates.
  Each keeps its observed/expected support, failed fraction, number of new
  episode onsets and onset rate per 10,000 updates. Windows are half-open;
  the final window includes the budget endpoint. At cadence 250, complete
  supports are 40/40/40/40/41, totaling the original 201 observations.
- Empty and partial windows remain visible. An empty fraction is unavailable;
  incomplete windows cannot contribute an onset rate or an adjacent-window
  change. An episode crossing a boundary has one onset; its failed
  observations contribute to every window where they occur.
- The final consecutive-target span and updates since the last failed
  observation are distinct quantities. Neither certifies unsampled updates.

```bash
python3 -m experiments.synthetic_trainers.stability_recovery \
  experiments/synthetic_trainers/baselines/adamw_stability_mod193_fraction25_lr0003_seed0_data0_20261002 \
  --window-steps 10000
```

The CLI verifies the complete archive before deriving metrics. Missing,
duplicated or nonfinite observations cannot improve the frequency. A history
without long confirmation retains its failed-window support but has no
post-generalization episode count.

The [validation](recovery-metrics-validation.json) records 13 passing affected
and persistence tests, four verified complete historical archives, unchanged
success gates and agreement with all existing tail supports/failure counts.
It also pins the exact observed lower-rate prefix through 137,750: three
episodes with 2/1/5 failed observations, recovering at 105,750, 109,250 and
112,000. Its first three complete tail windows have episode onset counts
2/1/0, but failed fractions 7.5%/12.5%/0%. The fourth window was partial and
the fifth empty in this snapshot; they were not treated as completed evidence.
All 31 frozen core/manuscript hashes and both active launchers still matched.

These windows were selected after observing the current calibration. Future
plans must freeze the 10,000-update width and this metric's source hash before
new training. Observed counts do not establish statistical monotonicity,
permanent recovery or a causal explanation. The separately frozen continuation
to a total of 300,000 updates must preserve the original complete 150,000-
update negative archive and evaluate its own final window at 250,000–300,000.
It is a budget extension of an exploratory calibration, followed by fresh
full-budget independent confirmations only if that calibration passes.
