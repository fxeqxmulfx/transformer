# Portable completed schedule-case verification

The read-only verification program (`verify_scheduled_case.py`) checks a
completed constant or cosine-tail case against its full archive using system
Python. It imports no Torch and loads no checkpoint tensors. Scientific cases
must match the exact committed prospective plan, retain all 300,000 updates
and 436,104 parameters, and preserve all 55 Python/seventeen native/nine Lean/
two manuscript/audit fingerprints. Raw measurements, canonical history,
neighbor/tensor/gradient bytes, recovery derivations, recorded segment peaks
and the actual raw checkpoint fingerprint must match. Full scientific counts
remain 1,201 / 2,400 / 3,600 / 300,000.

Both previously completed full-width **40-update CPU cases** have passed this
check. Each has 412,296 parameters and 9 / 16 / 24 / 40 observations; neither
passes stable grokking. Two fresh executions of the exact saved program
reproduce all six program/recovery files byte for byte. Eight actual rejection
checks preserve both existing receipts, reject the incomplete live scientific
case and an altered prospective plan, and reject altered raw canonical,
gradient, recovery and checkpoint bytes. Every rejected fresh output remains
absent. No optimizer updates or GPU context are created. The
[receipt and snapshots](scheduled-case-verification-preparation/validation.json)
retain the executed exercise, both CPU outputs and replay stdout. These are
archive-inspection fixtures, not scientific learning results or an additional
full test-suite run.

After the full constant case and its archive exist, run from the repository
root:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_scheduled_case adamw-constant
```

For the second case substitute `adamw-cosine-tail`. Default destinations are
`scheduled-constant-complete-metrics` and `scheduled-cosine-tail-complete-metrics`
under this protocol; an existing destination is rejected. Explicit `--stage`,
`--archive` and `--output` arguments support CPU inspection and fresh replay.
When executing a copied snapshot directly, supply the repository import path
with `PYTHONPATH="$PWD" python3 path/to/verification-snapshot.py ...`.

The generated receipt leaves native-state and PNG/PDF review flags unset.
Run the separate [native CPU checkpoint inspection](scheduled-checkpoint-audit-preparation.md),
review the actual figures, retain those outputs and update their hashes before
committing a scientific result. This helper opens no independent confirmation
gate; complete and review both full budgets and consume both terminal handles
before selecting the next scientific stage. Frozen training sources are unchanged.
