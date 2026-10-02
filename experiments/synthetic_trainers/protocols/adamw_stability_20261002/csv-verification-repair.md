# Linear execution of the frozen CSV verification

The first complete dense-gradient archive exposed a quadratic loop in the
frozen `stability_report.verify_csv`: its union of all column names was
recomputed for every row. At 150,000 rows this performs 22.5 billion row
visits, before accounting for individual keys. The archive worker
remained live inside this verification; it had not published a finished archive.

The [runtime adapter](csv_verification.py) computes the same sorted column
union once, then performs the unchanged complete-table string comparison.
All original history, diagnostic, criterion, checksum, summary and CSV checks
still execute. No frozen source file, model, optimizer, batch policy, scoring
rule, training budget or manifest was edited. This is an explicit post-freeze
implementation repair of analysis execution. Its additional SHA256 is recorded
in worker metadata and the [validation](csv-verification-repair-validation.json).

Four affected checks passed, including original/accelerated agreement on
heterogeneous tables and a real CPU archive, corrupted-cell rejection, a real
150,000-row table with two traversals, and restoration after an exception.
The full CPU suite passed all 235 tests in 34.481 seconds. Both the old staged
full AdamW archive and the newly published archive verified without PyTorch
in 3.858 seconds combined; their summaries and all ten measurement/CSV files
are identical. These are archive-processing measurements, not training speedups.
Figures were regenerated and individually checksummed.

The original non-training workers were deliberately stopped for this known
performance defect: archive PID 112672 and queued PID 115191 exited with 143.
The GPU trainer PID 112572/session 92762 continued without a restart. Replacement
archive PID 117200/session 10373 and queue PID 117203/session 46645 are live;
their exact [source snapshots](csv-repair-workers/archive-optimizer-pair-worker-v2.py)
and [queue snapshot](csv-repair-workers/queued-mod193-worker-v2.py) are retained.
Recheck actual process state before any restart.

The queue uses the adapter only for CSV verification before invoking the
unchanged frozen mod-193 launcher. Its `--check-only` passed; scientific mod-193
training remains queued behind the complete verified parent optimizer pair.
The adapter does not select a recipe or relax either scientific gate.

For dense archives while the frozen campaigns remain live, use:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification \
  archive experiments/synthetic_trainers/baselines/adamw_stability_adamw_short_lr001_seed0_data0_20261002
```

Use `comparison` for a complete comparison archive. After these frozen
campaigns finish, move the identical column-union optimization into the
ordinary verifier before freezing subsequent scientific plans.
