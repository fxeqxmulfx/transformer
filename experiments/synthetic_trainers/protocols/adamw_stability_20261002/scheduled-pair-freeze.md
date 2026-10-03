# Conditional fixed-schedule scientific freeze

The reviewed negative normalizer pair is committed at `8c6f1ad`. Both cases
finish 300,000 updates, both actual final native states and PNG/PDF figures
are reviewed, and both terminal sessions are consumed with exit code 0.
The primary softmax control fails its unchanged persistence criterion, so
the prepared conditional schedule gate applies; the all-six gate remains closed.

At **2026-10-03 05:08:56.498006 UTC**, the actual scientific freezer produces
the [prospective manifest](scheduled-pair-plan.json) and
[portable freeze receipt](scheduled-pair-freeze-validation.json). No scientific
optimizer update has started at this freeze. The raw plan and committed copy
are identical, and all 64 artifacts of the complete normalizer/reference
archive are preserved byte-for-byte in the new raw stage.

The two fresh cases run in fixed order:

| Case | Rate after ten-update warmup | Full update budget |
| --- | --- | ---: |
| `adamw-constant` | Constant 0.0003 | 300,000 |
| `adamw-cosine-tail` | 0.0003 through 150,000; cosine reduction to 0.00003 at 250,000; then constant | 300,000 |

Only `learning_rate_schedule` differs in their configuration. All other
fields retain native AdamW, all-parameter decay 0.1, betas (0.9, 0.98), epsilon
1e-8, no clipping, the same 436,104-parameter GPTMini, mod-193 division,
25% train fraction, model/data seeds 0/0 and batches 512 with 48-example tails.
The schedule is fixed before training, without early target stopping. Reducing
the rate also reduces AdamW's decoupled decay per update; these effects are
not isolated by this intervention.

The original memorization plateau, first twenty-observation long joint 99%
confirmation and **all 201 final-window observations, 250,000–300,000**, remain
required. Full canonical/neighbor/tensor/gradient logs, applied rates, actual
exposure, measured training/wall cost, peak memory and descriptive 10k recovery
bins remain mandatory. Recovery never overrides the strict final criterion.
Keep the 55 Python, seventeen native training, nine Lean, two manuscript and
audit fingerprints immutable until both trainer and archive worker finish.

Raw stage:
`experiments/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k`.
Commit the manifest before the actual launcher's `--check-only` validation
and GPU execution. Use `.venv/bin/python` for both the trainer and plotting
archive worker; system Python verifies portable reports without Torch.

The failed historical and normalizer evidence remains intact. This freeze
certifies neither stable grokking nor an architecture improvement. A complete
passing scheduled recipe must enter all six fresh crossed confirmations before
scientific architecture or complementary studies. No schedule outcome is known.

The manifest is subsequently committed at `d8ccb18`. The actual launcher's
preflight session 26391 terminates with code 0 before training. The
[launch receipt](scheduled-pair-launch-validation.json) and
[captured evidence](scheduled-pair-launch-evidence/native-case-plan.json)
record trainer **PID 168237/session 11503** and venv plotting archive worker
**PID 168332/session 60201**. Both are actually polled live. The captured
constant prefix through 3,500 has fifteen non-time observations equal to the
frozen same-seed normalizer control. Its first eleven applied rates match
native warmup exactly, starting at zero and reaching 0.0003 at update 11.
All source/proof/paper/audit pins remain unchanged. This prefix does not
establish persistence; no cosine case or independent confirmation is complete.

Logs are `experiments/runs/adamw_stability_20261002/scheduled-pair-driver.log`
and `experiments/runs/adamw_stability_20261002/scheduled-pair-archive-worker.log`.
The worker archives each full case for review/commit, then the whole pair;
review actual PNG/PDF files and native state before choosing the next gate.
Consume each real terminal session exactly once when finished. Keep the
scientific goal active throughout live GPU work; do not restart a healthy
process because a wait yields or intermediate evidence is incomplete.

The [completed-checkpoint CPU inspection](scheduled-checkpoint-audit-preparation.md)
is separately prepared and verified on both existing full-width CPU cases,
with four targeted tests and preserved snapshots. It uses each case's exact
last scheduled rate and performs no updates. The incomplete live scientific
case is explicitly rejected. Use it only after a full case/archive is ready;
it changes none of the 55 frozen training/protocol sources or scientific gates.

The [portable completed-case verifier](scheduled-case-verification-preparation.md)
is also exercised on both existing CPU archives, with exact saved-program
replays and actual guards against incomplete/altered evidence or overwrites.
It imports no Torch and leaves native-state/visual checks external. Use it
only after the corresponding full archive is ready; all frozen sources and
scientific gates remain intact.
