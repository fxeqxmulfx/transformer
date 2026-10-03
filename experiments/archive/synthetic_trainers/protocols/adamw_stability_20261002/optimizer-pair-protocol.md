# AdamW primary benchmark and paired optimizer control

The user explicitly selected AdamW as the primary optimizer on 2026-10-02.
The five completed raw AMSGradW calibrations remain complete negative evidence.
This prospective pair tests the optimizer change under the unchanged phase and
persistence criterion; historical AdamW also failed that stronger criterion, so
its stability is not assumed.

The task is the mod-97 division adaptation of *Convexifying Transformers*,
Section 4. Use softmax GPTMini, width 128, two layers, four heads, 423,816
parameters, float32, initialization/data seed 0, the exhaustive 4,656/4,656
split, short-final batches of 512/48, learning rate 0.001, decay 0.1,
ten-update warmup, and 150,000 updates per recipe. Both numeric answer and EOS
must be correct. AdamW runs first; the paired raw AMSGradW control runs second.
Only `optimizer` differs between their complete training configurations.

Native PyTorch AdamW uses betas (0.9, 0.98), epsilon 1e-8, bias correction,
no maximum second-moment buffer, and all-parameter decoupled decay. Raw
AMSGradW uses betas (0.9, 0.999), epsilon 1e-8, a maximum second moment, no
bias correction, and the same decay scope. This is a comparison of the whole
optimizer selection, not an isolation of beta2, bias correction, or the maximum.
The author's optimizer correspondence is documented in
[MODULAR.md](../../paper_reproduction/MODULAR.md); GPTMini remains an explicit
architecture adaptation. The local manuscript omits the exact fraction and
regularization settings, so those calibrated values are retained explicitly.

Both recipes have canonical exhaustive evaluations every 250 updates, step
zero, and the final update; immediate neighbor probes and full tensor/moment/
temperature diagnostics remain separate observations. Each complete run has
601 canonical, 1,200 neighbor, and 1,800 full diagnostic records. New identical
instrumentation also stores the already-computed gradient norm at **every**
update, along with batch size, epoch-tail status, and actual learning rate.
No gradient clipping is introduced. AdamW diagnostics report its actual
`exp_avg` and `exp_avg_sq` buffers without fabricating an AMSGrad maximum.

Dense tracing changes host synchronization and logging costs. Fresh paired
controls use identical instrumentation and separately retain diagnostic cost;
historical seconds are descriptive. CPU and real GTX 1050 checks preserve model,
optimizer buffers, canonical scores, shuffle state, and checkpoint continuation
exactly with tracing enabled. These short checks establish instrumentation
equivalence, not optimizer stability or a timing speedup.

The criterion in [STABILITY.md](../../STABILITY.md) is unchanged: train ≥99% /
held-out ≤10% memorization for five consecutive canonical observations spanning
1,000 updates before first held-out 99%; later joint ≥99% for 20 consecutive
canonical observations; joint ≥99% at all 201 observations in the last 50,000
updates. Full histories and all failures are required. Persistent rapid
generalization without the plateau is labeled separately and cannot qualify
as the stable-grokking baseline. Finish both full budgets regardless of targets.

The [frozen plan](optimizer-pair-plan.json) precedes scientific training and
records complete recipes, source and analysis/confirmation hashes, local paper
and split fingerprints, GPU/software environment, criterion, and launcher SHA256.
The [preparation validation](launch-preparation-validation.json) includes 224
passing CPU tests and exact 20-update CUDA instrumentation checks for both
optimizers. Launch or resume this custom stage from the repository root:

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_optimizer_pair
```

Add `--check-only` to verify without training. The default raw four-recipe CLI
does not reconstruct this plan. The launcher rejects changed recipes, sources,
instrumentation, criterion, environment, manuscripts, or the committed manifest.

Archive each complete result before waiting for the other recipe. Preserve
unchanged measurements, all three JSONL logs, all CSV tables, checkpoint SHA256,
complete assessments, artifact hashes, and standalone PNG/PDF figures. Verify
portable archives without PyTorch, inspect figures, and commit measured results.
Only a passing **primary AdamW** recipe can justify a separately frozen six-case
confirmation (new model seeds 4/5/6 crossed with data seeds 2/3). Every case
must pass before the architecture comparison. Those confirmations, the paired
architecture improvement, and complementary mechanics with separate novel-ID /
length transfer remain required later stages of the active goal.
