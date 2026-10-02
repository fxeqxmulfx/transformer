# Lower-rate native AdamW preparation

The quarter-split mod-193 calibration has the required delayed-generalization
phase but a verified final-window failure at 115,000. A lower learning rate
is a single-field follow-up hypothesis for persistent performance. It is
not an identified cause or a demonstrated stability repair. Finish and review
the full parent archive before freezing the scientific follow-up.

[Preparation](lower-rate-preparation/preparation.json) changes only learning
rate 0.001 to 0.0003. The complete 9,264 / 27,792 corpus, CPU initial model
and buffers, 436,104 parameters, 150,000-update exposure and all 31 current
frozen fingerprints match. Prime 193, 25% fraction, seeds 0/0, width 128,
two layers, four heads, short-final batch 512, warmup 10 and decay 0.1 remain
fixed. Native AdamW keeps betas (0.9, 0.98), epsilon 1e-8, bias correction,
all-parameter decay and no gradient clipping. The rate also changes cumulative
decoupled shrinkage within the fixed budget, despite the unchanged decay
coefficient; the intervention does not isolate an individual causal component.

The [real CPU smoke](lower-rate-preparation/cpu-smoke.json) completes 20
full-width updates, three exhaustive observations at 0/19/20 and every gradient
record. All warmup rates are independently checked; update 19 uses the
48-example tail and update 20 the next full 512-example batch. Its complete
portable archive verifies with the ordinary tools without PyTorch. CUDA was
hidden from this CPU process, and no CUDA context was created. This smoke
overlapped the end of the parent's GPU training; retain that activity when
interpreting the parent's descriptive costs. No causal timing ratio is inferred.

[Validation](lower-rate-preparation-validation.json) checks all artifact hashes,
the exact executed source snapshot, scientific configuration, original history
bytes, portable measurements and unchanged frozen sources. The
[helper](prepare_lower_rate.py) records these checks with explicit preparation
scope. This is an adaptation of *Convexifying Transformers*, Section 4; its
learning rate and stronger phase/persistence criterion are study choices.

No new scientific plan or GPU training is created by this preparation. The
unchanged rule still requires the pre-target memorization plateau, 20 later
joint-target observations and all 201 observations in the final 50,000 updates.
Six fresh independent cases are permitted only after a full passing primary
calibration. Architecture and scientific complementary comparisons follow that
gate; they remain outstanding.
