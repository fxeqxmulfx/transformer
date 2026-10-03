# Completed native schedule checkpoint inspection

The read-only [audit program](audit_scheduled_checkpoint.py) is prepared for
both full scientific schedule cases. It checks the complete frozen case,
loads the actual checkpoint on CPU, restores the native AdamW class, and
verifies all model/state tensors, state counts/shapes/steps, one-time coverage
of every parameter, betas (0.9, 0.98), epsilon 1e-8, all-parameter decay and
the exact **last applied rate** from the prospective schedule. Plain AdamW
has no maximum-second-moment buffers. The optional archive fingerprint must
match the actual checkpoint. No forward training or optimizer step is performed.

The program has inspected both previously completed 40-update, full-width
CPU preparation cases. Each has 412,296 parameters and eleven finite native
states at step 40. Their final rates are 0.0003 and
2.9999999999999997e-05. The smaller vocabulary explains their parameter count;
the scientific mod-193 model still has 436,104 parameters. These checkpoints
provide implementation evidence, not scientific learning or stability results.
Raw fingerprints and the exact executed program are in the
[inspection receipt](scheduled-checkpoint-audit-preparation/validation.json).

Four targeted tests pass in **6.934 seconds**. They train two actual tiny
CPU cases of twelve updates each, inspect both distinct rates, reject forged
rates/steps/nonfinite model or moment tensors, reject incomplete execution,
and preserve an existing CLI receipt. Separately, the actual live scientific
constant case is rejected because its full report is not yet present, leaving
no output receipt. The final test log, script/test snapshots and actual CPU
inspection receipts are hashed and retained. This adds no scientific updates
and leaves all 55 Python/seventeen native/nine Lean/paper/audit pins unchanged.
The latest previously completed full CPU suite remains the archived 288-test run;
these four new targeted checks are reported separately.

After a full constant case and its archive exist, run:

```bash
env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m \
  experiments.synthetic_trainers.protocols.adamw_stability_20261002.audit_scheduled_checkpoint \
  --stage experiments/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k \
  --case adamw-constant \
  --archive experiments/synthetic_trainers/baselines/adamw_schedule_adamw-constant_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002 \
  --output experiments/runs/adamw_stability_20261002/bootstrap/scheduled-constant-final-CPU-audit.json
```

For the second full case substitute `adamw-cosine-tail` and a fresh output.
Do not overwrite a previous receipt. Transfer the ignored raw checkpoint
separately for inspection; portable measurements verify without Torch or weights.
Finite native state alone opens no scientific gate. Complete both budgets,
review actual PNG/PDF figures and histories, commit each case and the pair,
and consume both terminal handles before selecting the next scientific stage.
