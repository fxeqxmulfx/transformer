# Observed-equation coverage in quotient discovery

Does including the remaining observations remove a discovery refusal caused
by an underdetermined equation prefix? The frozen task232 Fibonacci case has
4704 observations but constructs only the first4096 equations. The prefix
omits observed raw class33. Its modulus97 system has573 variables and five
free coordinates after the two shift gauges; the allowed nullity is one.
The original55-stage catalogue admits no code. Missing class coverage explains
at least one unconstrained coordinate, not necessarily all five directions.

| Property | Frozen original | Coverage pilot |
| --- | --- | --- |
| Equation allowance | 4096 | 8190; includes all4704 observations |
| Prime catalogue | All55 primes through257 | Same |
| Elimination limits | 500M scalar updates;2M row operations; nullity1 | Same |
| Initialization, ordinary CE and full/mini/forward policies | Original | Same |
| Certificate prefix and readout | Original | Same |
| Held-out evaluation and parent selection | Separate original comparison | Neither runs |

This is a configuration-only test using the already reviewed native executable.
CM30's Lean `DecodedConsequences.code_of_program` and `choose_quotient_sound`
use an arbitrary observed row set and explicit structural hypotheses; they do
not require a4096-row prefix or assert that enlarging it discovers a useful
code. The native component already supports8190 equations plus two gauges.
CD8 supplies convex ordinary CE after fixed feature discovery; the existing
complex bounds and replay remain unchanged. No new Lean statement is needed
for this supported parameter change. No arithmetic answer feature is added.

`protocol.json` pins the input configuration, unchanged executable/source and
the single changed field before execution. `run.sh` rejects any source or binary
mismatch, saves the exact configurations and runner, checks all other fields
for equality, retains every55-stage refusal/error and independently replays
actual model states and CE trajectories. Both commands use the64GiB cap and
check a16GiB disk reserve. A process failure retains its files and resource log.
Generated evidence stays ignored under `runs/`.

```sh
# From the repository root at the prospective experiment commit:
bash -n experiments/convex_stress_quotient_coverage/run.sh
mkdir -p experiments/convex_stress_quotient_coverage/runs
bash experiments/convex_stress_quotient_coverage/run.sh \
  experiments/convex_stress_quotient/cohort/runs/original-v1/recurrence-fibonacci-k1-c2-lgt/catalogue/configuration.json \
  .kanban/artifacts/232/freeze-v1/quotient-v1 \
  experiments/convex_stress_quotient_coverage/runs/coverage-v1 \
  "$(git rev-parse HEAD)"
```

The separate pilot has not run yet. It cannot change the original232 outcome
or the reviewed35-task comparison. A successful discovery would still require
the refined head, complete original evaluation and independent review before
any new benchmark success is claimed.
