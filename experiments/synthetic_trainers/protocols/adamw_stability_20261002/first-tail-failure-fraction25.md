# Quarter-split delayed generalization fails final-window persistence

At update 115,000, the frozen `adamw-mod193-fraction25-lr001` calibration
fails the joint 99% target for the first time in its 100,000–150,000 final
window. Exhaustive train / held-out complete RHS accuracy is
92.7461%/91.6271%. The preceding 60 canonical observations in that window
pass. One failed observation refutes the unchanged requirement that all
201 final-window observations pass, irrespective of later recovery.
The full 150,000-update budget continues and every later failure is retained.

The required [memorization phase and later long confirmation](first-long-confirmation-fraction25.md)
were observed: the longest plateau spans 4,500–11,500, followed by first
held-out 99% at 20,000 and 20 consecutive joint targets through 24,750.
Thus this calibration has measured delayed generalization but fails the
stronger stable-grokking benchmark. It cannot open six-case independent
confirmation. Long-confirmation timing remains an observed event and is
ineligible as a successful persistent-target benchmark cost.

The [portable bundle](first-tail-failure-fraction25/summary.json) preserves
the unchanged frozen plan and all 461 canonical observations from zero
through 115,000, five exhaustive observations with full tensor diagnostics,
and all 252 gradient records from 114,750 through 115,001. The existing
[offline verifier](verify_first_tail_failure.py) recomputes assessment,
coverage, exhaustive support, batch/exposure arithmetic, native AdamW buffers,
sampled/dense agreement and artifact hashes without PyTorch or original runs.
[Validation](first-tail-failure-fraction25-validation.json) checks original
prefix/interval bytes, original exhaustive/tensor samples and all 31 unchanged
source/manuscript fingerprints. Four corrupted copies with recomputed hashes
are rejected. The negative-second-moment check also recomputes derived metrics
to reach the semantic nonnegative-buffer guard.

Both scores are 100% at 114,750 and 114,751. The failure neighborhood is:

| Update | Last batch | Train accuracy | Held-out accuracy | EOS accuracy, both splits |
| --- | ---: | ---: | ---: | ---: |
| 114,999 | 512 | 94.0846% | 93.1851% | 100% |
| 115,000 | 512 | 92.7461% | 91.6271% | 100% |
| 115,001 | 512 | 96.1896% | 95.5347% | 100% |

The numeric answers fail at both immediate neighbors as well as the canonical
evaluation. Every preserved sample follows a full batch. The interval's largest
gradient norm is 80.2917 at update 114,996, also on a full batch; accuracy was
not measured at that exact update, so it does not identify the causal trigger.
The sampled joint stored second-moment norm rises from 1.9544e-6 at 114,751
to 2.3416 at 114,999. These are uncorrected native buffer norms, not effective
optimizer denominators or a controlled causal explanation.

This remains a GPTMini/AdamW mod-193 adaptation of *Convexifying Transformers*,
Section 4, with explicitly chosen phase/persistence rules. Keep the observed
phase and failed persistence separate. Finish and verify the complete result
and curves before freezing a justified next intervention; no independent
confirmation or architecture comparison has been launched by this inspection.

```bash
python3 -c 'from experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_first_tail_failure import verify; print(verify("experiments/synthetic_trainers/protocols/adamw_stability_20261002/first-tail-failure-fraction25"))'
```
