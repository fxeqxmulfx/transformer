# Actual constant-control phase and first sampled recovery

The live scientific schedule pair's fresh constant softmax control has an
actual preserved prefix through **105,750 / 300,000 updates**. The frozen
criterion finds a 4,500–27,250 memorization plateau (92 observations), followed
by twenty joint train/held-out 99% observations at 95,750–100,500. Observed
confirmation costs are 2,617.54 training / 3,223.61 wall seconds. These are
times to an observed event, not eligible persistent-target times.

Its first sampled post-confirmation episode fails held-out at 105,250 and
105,500, reaching a canonical minimum of 97.481290%. The next passing canonical
observation is 105,750, a sampled duration of 500 updates. Train and EOS stay
100% at these recorded boundaries. Adjacent probes still fail at 105,501 and
pass at 105,749; the exact first intervening recovery time is unobserved.
All twelve retained boundary gradient records use full batch 512 and rate
0.0003. The records do not identify the cause of the episode.

The [captured witness](scheduled-constant-first-phase-and-recovery/verification-receipt.json)
retains all 424 canonical observations, four boundary observations, eight
neighbor probes, twelve actual tensor/gradient records, the complete frozen
root/native plans, an independently recomputed phase/recovery assessment and
exact program snapshots. All 424 non-time canonical records equal the previous
full softmax reference with the same initialization/data seeds. This A/A
observation is neither independent confirmation nor parameter-tensor identity.

An actual fresh execution of the saved capture program reproduces eight core
files byte for byte. The [standard-library verifier](verify_scheduled_constant_phase.py)
independently checks artifact hashes, phases, failures and recovery against the
unchanged 99% / twenty-observation / final-50,000-update criterion, without
Torch or updates. The actual [PNG](scheduled-constant-first-phase-and-recovery/constant-phase-and-recovery.png)
and [PDF](scheduled-constant-first-phase-and-recovery/constant-phase-and-recovery.pdf)
have been visually reviewed, including a raster of the PDF; their hashes and
the executed plotting source are retained.

No final-window observations are present in this prefix. Stable grokking,
full-budget completion and the independent confirmation gate remain false.
Both real scientific processes continue under unchanged 55 Python/seventeen
native/nine Lean/two paper/audit pins. Keep the full constant and cosine-tail
budgets, review and commit both complete archives, and consume their terminal
handles before selecting the next scientific stage. Recheck the saved witness:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_scheduled_constant_phase \
  experiments/synthetic_trainers/protocols/adamw_stability_20261002/scheduled-constant-first-phase-and-recovery
```
