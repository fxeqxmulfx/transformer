# Mod-193 first target: longer learning without the required memorization phase

The frozen harder-task AdamW recipe first reaches exhaustive held-out 99%
at update 40,000. Complete RHS train/held-out accuracy is
99.5952%/99.3253%, with 1,015.69 measured training / 1,257.28 wall seconds.
The first canonical train 99% observation is at 39,500, when held-out
accuracy is already 98.5320%. Before the first held-out target, **zero**
observations meet train ≥99% and held-out ≤10% simultaneously.

The absent required pre-target plateau cannot be supplied by later observations.
Consequently, this recipe is phase-ineligible for the unchanged stable-grokking
benchmark, irrespective of its eventual final-tail persistence. Increasing the
prime delayed the observed target on this calibration but did not separate
memorization from generalization. A positive gap between threshold crossings
alone does not meet the prospectively specified plateau criterion.

The [portable prefix](first-target-mod193) preserves every canonical observation
from zero through 40,000 (161) and every pre-update gradient record (40,000),
the unchanged frozen plan, complete partial assessment and artifact hashes.
The [validation](first-target-mod193-validation.json) checks source prefix
lines, every observation's metrics/exposure, gradient batch/rate/finiteness,
and all 31 unique frozen source/manuscript fingerprints without PyTorch.
The existing pure verifier recomputes the assessment offline:

```bash
python3 -c 'from experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_first_confirmation import verify; print(verify("experiments/synthetic_trainers/protocols/adamw_stability_20261002/first-target-mod193"))'
```

This remains an incomplete scientific prefix. The archived cutoff has no long
confirmation yet and no final-window observations. Continue the entire frozen
150,000-update budget, retain all failures and measured costs, and verify the
complete archive. Do not declare stable training from the current rebound or
open six-case confirmation from this absent phase.

The task adapts *Convexifying Transformers*, Section 4. Only the prime changes
relative to the primary mod-97 configuration; the vocabulary, total parameter
count, common-seed initial weights, examples per epoch, exposure and exhaustive
evaluation cost also change as documented in
[the frozen protocol](larger-modulus-protocol.md). This one split/initialization
does not identify an algorithm, isolate arithmetic difficulty, or establish a
repeatable speed ratio. Criteria and every frozen production file remain intact.
