# Preparation for a smaller training fraction on mod-193

The complete canonical mod-193 prefix reaches held-out 99% at 40,000 without
the required memorization plateau. A prepared 25% training fraction, versus
the current 50%, will test whether fewer observed equations separate fitting
the training set from generalizing. This is a hypothesis, not a learning
result. The current GPU budget remains unchanged; this preparation has not
frozen or launched a new scientific campaign.

This follow-up adapts *Convexifying Transformers*, Section 4. The local
manuscript describes delayed generalization on modular division but does not
specify these train fractions. The thresholds and persistence window are
prospective follow-up choices. Keep the primary AdamW optimizer, mod-193,
model/data seeds 0/0, dimensions, vocabulary, learning rate 0.001, decay 0.1,
warmup 10, 150,000 updates, exhaustive evaluation cadence and instrumentation.
Only `train_fraction` changes in the prepared scientific configuration.

| Derived quantity | Current 50% split | Prepared 25% split |
| --- | ---: | ---: |
| Train examples | 18,528 | 9,264 |
| Exhaustive held-out examples | 18,528 | 27,792 |
| Vocabulary | 335 | 335 |
| Parameters | 436,104 | 436,104 |
| Full 512-example batches per epoch | 36 | 18 |
| Short epoch tail | 96 | 48 |
| Updates per epoch | 37 | 19 |
| Full-budget example exposure | 75,113,536 | 73,137,184 |

The [preparation records](fraction25-preparation/preparation.json) check every
legal equation by independent inverse and multiplication oracles, unique
bounded operand pairs, disjoint exhaustive splits, and all numerator,
nonzero-denominator and answer classes in each split. The new training set
is a prefix of the original training set; the original held-out set is an
unchanged subset of the new held-out set. CPU initial model states match
tensor for tensor, including buffers, with a recorded state SHA256. The
fraction intervention therefore preserves parameter count and initialization,
unlike the earlier change of prime.

Changing the training fraction also changes epoch length, tail size, exposure,
held-out composition/support and evaluation cost. These are explicit derived
differences. Target timing across the fractions is descriptive on different
training/evaluation supports; this preparation does not establish a causal or
repeatable speedup. Neither case is a length-transfer experiment.

The [real CPU smoke](fraction25-preparation/cpu-smoke.json) uses full width,
20 updates and exhaustive evaluations at 0/19/20. It checks the 48-example
tail at 19, next 512-example batch at 20, all 20 dense gradient records,
native AdamW moment diagnostics and complete portable archive verification
without PyTorch. GPU visibility is disabled for this CPU subprocess and no
CUDA context is created. These checks establish implementation only. The
original visible-GPU CPU smoke completed its budget but failed the additional
no-CUDA-context guard; its ignored archive is retained. In the installed
PyTorch, the optimizer's accelerator graph-capture health check obtains an
accelerator stream even for CPU parameters. The archived executed helper's
docstring incorrectly attributed this to checkpoint CUDA RNG capture; that
explanation is corrected here and in the current helper. Their executable
ASTs agree, as checked in the [validation](fraction25-preparation-validation.json).

To reproduce at the recorded source version in fresh destinations:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -u \
  -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.prepare_fraction25
```

All 31 frozen production/manuscript fingerprints remain unchanged. Finish and
verify the current full mod-193 result, then apply the already verified CSV
column-union optimization to the ordinary verifier before freezing the next
scientific plan. Only a passing full primary calibration can open the six new
independent confirmations; architecture and scientific complementary mechanics
remain required after their unchanged gates.
