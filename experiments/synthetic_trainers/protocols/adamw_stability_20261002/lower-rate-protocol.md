# Frozen lower-rate AdamW calibration

The [complete quarter-split parent](fraction25-result.md) observes delayed
generalization but fails three final-window targets, including held-out 48.9206%
at 139,000. This control tests a lower nominal learning rate while preserving
the task and all success requirements. It does not identify the cause or
assume that a lower rate will restore persistence.

The [prospective plan](lower-rate-plan.json) is frozen at
2026-10-02T19:11:49.393424+00:00 from training/analysis source commit
`bee6e44`. Recipe `adamw-mod193-fraction25-lr0003` changes only learning rate
0.001 to 0.0003. All training/analysis sources, environment, manuscripts,
corpus, instrumentation and criteria match the complete parent. Both launcher
and archive-worker SHA256 are pinned. The
[real launcher and negative checks](lower-rate-freeze-validation.json) verify
the parent and reject changed rates, optimizers, criteria and corpus support
in isolated copies. The [CPU preparation](lower-rate-preparation.md) verifies
identical complete corpus/initial states and real lower-rate warmup updates.

Keep mod 193, fraction 25%, seeds 0/0, GPTMini width 128/two layers/four
heads/436,104 parameters, 9,264 training and 27,792 held-out pairs. Native
AdamW retains betas (0.9, 0.98), epsilon 1e-8, bias correction, all-parameter
decay 0.1 and no clipping. Use warmup 10, short-final batch 512, 150,000
updates, exhaustive evaluation/diagnostics every 250 with immediate neighbors,
and every pre-update gradient norm. The 48-example tail, 19 updates per epoch
and 73,137,184 full-budget examples are unchanged. Rate reduction also changes
cumulative decoupled shrinkage; it does not isolate the decay contribution.

The criterion remains train ≥99% / held-out ≤10% for five consecutive
observations spanning 1,000 updates before first held-out 99%, followed by
20 joint ≥99% observations and joint ≥99% at all 201 final-window points
100,000–150,000. Finish the whole budget and retain every failure. Only a
full passing primary calibration permits the separately frozen six cases
model seeds 4/5/6 crossed with data seeds 2/3; all must pass before architecture
selection. Scientific complementary tasks remain outstanding as well.

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_lower_rate
```

Add `--check-only` to validate without training. Output is
`experiments/runs/adamw_stability_20261002/calibration_mod193_fraction25_lr0003/`.
Use a fresh module process and record its actual PID before starting the
NoTorch archive worker with `--trainer-pid`. Review full standalone PNG/PDF
figures and verify/commit both complete archives after termination. Never
restart on an observation timeout or mutate frozen files during this stage.

The verified manifest is committed as `ed83c85`. Scientific trainer PID
126400/session 93469 and NoTorch archive worker PID 126478/session 19327 are
live, as recorded in the [launch receipt](lower-rate-launch-validation.json).
This is one incomplete calibration; the whole-budget and six-case requirements
remain pending. Driver output and worker metadata are under
`experiments/runs/adamw_stability_20261002/lower-rate-driver.log` and
`experiments/runs/adamw_stability_20261002/bootstrap/lower-rate-archive-worker.json`.

If ordered phases are observed before completion, preserve the first long
confirmation with `record_phase_prefix` and review `plot_phase_prefix` output
in a separate fresh directory. The
[export validation](phase-prefix-export-validation.json) reproduces all five
existing quarter-split prefix files byte for byte, rejects overwrites and
rejects the actual unconfirmed lower-rate stage without creating a destination.
This NoTorch exporter is outside the frozen training/analysis source list;
it reuses the existing unchanged semantic prefix verifier. A saved phase
prefix never opens the full-budget or independent-confirmation gate.

This is a *Convexifying Transformers*, Section 4 adaptation with explicit
fractions, rates and stronger persistence choices. It measures fixed-length
novel operand-pair generalization; it does not establish length transfer,
an internal algorithm, universal optimizer stability or a causal speedup.
