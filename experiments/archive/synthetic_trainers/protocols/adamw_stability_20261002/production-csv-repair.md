# Ordinary CSV verification after the frozen campaigns

Both optimizer-pair and mod-193 scientific budgets and archive workers are
terminal, and their complete results have been verified and committed. The
previously verified runtime optimization is now applied directly to
`stability_report.verify_csv`: compute the complete sorted column union once,
then compare every CSV cell against every measured row with identical string
and missing-value rules. Dense verification now traverses the row collection
twice rather than once per measured row.

The [validation](production-csv-repair-validation.json) records 235 passing
CPU tests with no skips. The real 150,000-row traversal regression now checks
both the ordinary verifier and historical runtime adapter. Heterogeneous,
empty, missing, boolean and quoted/newline cells and corruption rejection
retain the same outcomes. Three complete dense individual archives and both
complete comparisons verify offline without PyTorch and return identical
results through the ordinary verifier and adapter (25.67 seconds for all
paired checks). Every original archive remains byte-for-byte unchanged.

Only one of the 31 frozen source/manuscript files differs from the completed
plans: this analysis function. All training sources, data, criteria and
manuscripts are unchanged. The runtime adapter and exact historical worker
snapshots remain available for provenance; future plans must record the new
ordinary-verifier SHA256. Historical strict launchers still pin their original
source version and should be reproduced from that version, while completed
archives verify with the current ordinary tools.

Ordinary offline commands now suffice:

```bash
python3 -m experiments.synthetic_trainers.stability_report \
  experiments/synthetic_trainers/baselines/adamw_stability_mod193_lr001_seed0_data0_20261002 --verify
python3 -m experiments.synthetic_trainers.stability_comparison \
  experiments/synthetic_trainers/baselines/adamw_stability_mod193_calibration_20261002 --verify
```

This change concerns verification cost only. It does not create a passing
calibration, alter the prospective gate, or establish a scientific speedup.
