# Mod-193 first final-window failure

At update 102,750, `adamw-mod193-lr001` fails the joint 99% target for the
first time in its frozen 100,000–150,000 final window. Exhaustive train /
held-out complete RHS accuracy is 95.1749%/94.6082%. The preceding eleven
canonical observations in that window pass. This failed observation rules
out the requirement that every final-window observation passes, irrespective
of later recovery. The unchanged full 150,000-update budget continues.

The [portable bundle](first-tail-failure-mod193/summary.json) preserves the
unchanged frozen plan and every canonical observation from zero through
102,750 (412 total), five exhaustive observations and full tensor diagnostics,
and all 252 gradient records from 102,500 through 102,751. The existing
offline verifier (`verify_first_tail_failure.py`) checks the hashes, complete
prefix coverage, exhaustive support, modular batch/exposure arithmetic,
native AdamW buffers, sampled/dense agreement and partial assessment without
PyTorch or original run paths. The
[validation](first-tail-failure-mod193-validation.json) checks original source
bytes/samples, unchanged 31 frozen fingerprints and four semantic corruption
checks with repaired artifact hashes. The negative-buffer check also recomputes
the derived summary to reach the native nonnegative-buffer guard.

At 102,500 and 102,501 both scores are 100%. Around the failed evaluation:

| Update | Last batch | Train accuracy | Held-out accuracy | EOS accuracy, both splits |
| --- | ---: | ---: | ---: | ---: |
| 102,749 | 96 | 95.0345% | 94.7107% | 100% |
| 102,750 | 512 | 95.1749% | 94.6082% | 100% |
| 102,751 | 512 | 95.7038% | 95.3368% | 100% |

All three failures concern numeric answers. Unlike the mod-97 canonical
evaluations, this failed canonical observation follows an ordinary full
batch. Its preceding neighbor follows a short tail and also fails; this
neighborhood does not exclude a preceding-batch effect or identify a cause.

The preserved interval's largest gradient norm is 66.7514 at 102,727,
following a full 512-example batch. Accuracy was not evaluated at that update,
so its position before the failed canonical evaluation does not establish
causal ordering with the underlying regression. The sampled joint stored
second-moment norm rises from 1.1160e-6 at 102,501 to 2.1452 at 102,749;
these are uncorrected buffer norms, not effective optimizer denominators.

The [required memorization phase](first-target-mod193.md) was already absent.
Thus larger-modulus training fails both frozen phase and persistence gates;
it cannot open independent confirmation. First long confirmation at 57,000
remains an observed event, not a persistent-target timing success.

This is a GPTMini/AdamW mod-193 adaptation of *Convexifying Transformers*,
Section 4, with explicitly chosen phase/persistence rules. The result does
not show that AdamW is universally unstable or that a fraction intervention
will solve it. Finish the full result, retain every subsequent failure and
curve, then freeze a justified next intervention. The prepared 25% fraction
tests the absent memorization phase; it has no scientific success yet.

Recompute this partial result offline:

```bash
python3 -c 'from experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_first_tail_failure import verify; print(verify("experiments/synthetic_trainers/protocols/adamw_stability_20261002/first-tail-failure-mod193"))'
```
