# Frozen native AdamW continuation to 300,000 total updates

The user requested a maximum total of 300,000 training updates per run after
asking to measure later recovery and decreasing failure frequency. Preserve
the [complete negative 150,000-update calibration](lower-rate-result.md).
This continuation fixes its additional 150,000 updates before they execute.
It remains an exploratory posthoc extension of the original budget.

The [new manifest](budget300k-plan.json) is frozen at
2026-10-02T21:15:19.158465+00:00 from implementation commit `5c744d8`.
Only total steps change. Keep learning rate 0.0003, decay 0.1, native AdamW
betas (0.9, 0.98), ten-update warmup, mod 193/fraction 25%, seeds 0/0, GPTMini
width 128/two layers/four heads and all-parameter decay without clipping.
Restore the byte-identical 150,000-update checkpoint and all original logs in
a fresh stage. Weights, moments, parameter steps and sampling generator/
permutation/cursor are retained; warmup starts at its original absolute step.

The [preparation](budget-extension-preparation.md) verifies exact continuation
against uninterrupted CPU training. The actual launcher `--check-only` passes,
and the [guard audit](budget300k-guard-validation.json) rejects eight isolated
changed manifests/resources. The complete parent, full native optimizer step,
example exposure/cursor/permutation, environment, corpus, instrumentation and
criterion match. All 39 core/manuscript/extension/recovery sources are pinned.
The [freeze receipt](budget300k-freeze-validation.json) precedes scientific updates.

The final 50,000-update persistence window is now 250,000–300,000, containing
201 scheduled exhaustive evaluations at cadence 250. The memorization and
long-confirmation rules remain unchanged. Additional 10,000-update recovery
windows cover both the final tail and the whole post-long-onset history.
Partial/empty windows retain support and cannot supply completed frequency
rates. Episode frequency, depth, recovery intervals and final-target streaks
supplement the strict success gate. Every original failure remains visible.

The total budget consumes 146,273,904 examples and must retain 1,201 canonical
evaluations, 2,400 immediate neighbors, 3,600 full diagnostics and 300,000
gradient norms. CUDA peaks combine the original parent and observed extension
segments. Host peak snapshots contribute to measured wall time; cumulative
checkpoint training/diagnostic/execution-wall offsets exclude the intervening
pause. This cannot establish a causal speedup or unsampled future stability.

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_budget_extension
```

Add `--check-only` for verification. Output is
`experiments/runs/adamw_stability_20261002/calibration_mod193_fraction25_lr0003_budget300k/`.
Record the actual trainer PID before starting the NoTorch archive worker with
`--trainer-pid`. The worker waits for complete stage state and finalized peak
metadata, then verifies full individual/whole-stage archives and additional
recovery metrics. Review all standalone PNG/PDF and commit the full result.
Keep all 39 pinned sources immutable until trainer and archive worker finish.
Do not restart on a monitoring timeout or extend beyond the user's cap.

The manifest is committed as `3cf14e2`. Actual trainer PID 135520/session
64768 and NoTorch archive worker PID 135597/session 37682 are running. The
[launch receipt](budget300k-launch-validation.json) verifies their commands,
all 39 hashes, original log prefixes and actual first additional step 150,001
at rate 0.0003. First added canonical evaluation is 150,250. The full budget
and independent confirmations remain pending. Monitor
`experiments/runs/adamw_stability_20261002/budget300k-driver.log` and
`experiments/runs/adamw_stability_20261002/bootstrap/budget300k-archive-worker.json`.

Only a full passing calibration permits six fresh 300,000-update cases:
model seeds 4/5/6 crossed with data seeds 2/3. All must pass before architecture
selection, paired architecture controls and the outstanding complementary tasks.
This is a *Convexifying Transformers*, Section 4 adaptation measuring novel
fixed-length operand pairs, with length transfer assessed separately later.
