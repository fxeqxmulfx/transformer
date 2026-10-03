# Tagged native AdamW confirmation preparation

Prepared on 2026-10-03 while the immutable sparsemax/softmax pair continues.
The primary optimizer remains the user-selected native AdamW. The task is the
modular division adaptation of *Convexifying Transformers*, Section 4.
No scientific confirmation is selected, frozen or started. Both current
normalizer cases must finish, be verified/reviewed and have their results
committed before selecting the next scientific stage.

The existing confirmation driver expects ordinary `RunConfig` fields and its
earlier source set. It cannot directly execute the seventeen-source normalizer
or schedule recipes. The separate [adapter](../../tagged_confirmation_protocol.py)
retains those explicit tags and calls their existing native trainers. None of
the current pair's 43 Python or nine Lean files changes. The already prepared
schedule pipeline's 55 fingerprints also remain unchanged.

The first new scientific cohort remains exactly model seeds 4/5/6 crossed with
data seeds 2/3: six cases, each fresh, all frozen before the first update.
Historical/calibration model seeds 0/1/2/3 and data seeds 0/1 are excluded.
Model configuration, optimizer/rate fields, instrumentation, task and scoring
remain identical to the selected recipe; only model/data seeds change. Data
fingerprints are regenerated and checked, with one common split per data seed.
Every scientific case retains 300,000 total updates and the unchanged phase,
long-confirmation and final 50,000-update persistence rules.

Only the passing primary softmax case of a complete scientific normalizer pair,
or a passing softmax recipe of a complete conditional schedule pair, can be
selected. Sparsemax remains an architecture candidate. Selection requires
the full paired archive, a committed result receipt, actual PNG/PDF review and
consumed terminal handles. The scientific launcher requires its new manifest
to be committed. All 62 implementation fingerprints, the exact seventeen
training sources, reference artifact bytes, nine Lean fingerprints and their
audit remain checked. These adapters do not change any success criterion.

The serial driver preserves completed cases on interruption and never retrains
them. Every native gradient, neighboring probe, diagnostic and measurement
must remain byte-identical in an existing case archive. Memory snapshots retain
the maximum observed across execution segments; unused CUDA cache is cleared
before each case. Setup, verification and archival costs stay separate from
the trainer's measured update time. Each complete case archive can be reviewed
and committed while the remaining cases run.

The [portable report](../../tagged_confirmation_report.py) keeps all six full
histories, complete calibration evidence and per-case outcomes. It retains
both joint and held-out recovery scores, sampled durations/censoring, final
tail windows, leading partial windows and the whole post-onset grid. Missing
long onset remains unavailable rather than zero episodes. The figures retain
all six canonical trajectories and every actual learning rate. Crossed means
and sample SD are descriptive, not IID evidence or confidence intervals.
Use `tagged_confirmation_report.verified_benchmark` for later architecture
selection: all six scientific primary cases must pass. CPU fixtures cannot
open this gate, including hypothetical all-success summary flags used only
in a unit test. A failed scientific cohort consumes its seeds; any later
cohort needs a new prospectively frozen set, not a silent reuse of this first set.

Four [affected tests](tagged-confirmation-preparation-validation.json) pass in
12.484 seconds. They exercise both actual native paths, exact matrix/tag/source
guards, source drift before training, interrupted/resumed complete cases,
unmodified cached cases and offline verification after deleting raw runs.
A dropped case, forged recovery after rehashing and a self-consistent forged
gradient archive that differs from its raw trace are rejected.

The [actual CPU preparation](tagged-confirmation-preparation/validation.json)
executes six full-width softmax cases at 20 updates and six full-width annealed
cases at 40 updates: twelve real cases and 360 total updates. Every negative
outcome is retained. Both complete archives verify in system Python without
Torch, and both report no repeatable scientific benchmark. The mod-7 CPU model
has 412,296 parameters; these fixtures do not replace the scientific mod-193
436,104-parameter model. Case-level figures are omitted in these fixtures;
the scientific freeze defaults to including them. All four newly rendered
combined PNGs and their four actual PDFs are inspected. The copied calibration
figures remain byte-identical to their already reviewed references.

The actual scientific freezer rejects the live incomplete normalizer pair
before creating a manifest or initializing CUDA. If the complete fresh
softmax passes, the future commands are:

```bash
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.freeze_tagged_confirmation --calibration experiments/synthetic_trainers/baselines/adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002 --recipe adamw-softmax --output experiments/runs/adamw_stability_20261002/first_tagged_six_case_scientific_confirmation
# Review and commit tagged-confirmation-plan.json before launching.
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_tagged_confirmation --output experiments/runs/adamw_stability_20261002/first_tagged_six_case_scientific_confirmation --check-only
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_tagged_confirmation --output experiments/runs/adamw_stability_20261002/first_tagged_six_case_scientific_confirmation
```

If the primary control fails, retain that result and consider the separately
prepared schedule calibration. Six scientific confirmations, architecture
repetitions and scientific complementary mechanics remain outstanding.
