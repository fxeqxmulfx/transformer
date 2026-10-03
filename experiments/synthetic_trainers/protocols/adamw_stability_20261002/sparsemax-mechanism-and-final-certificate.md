# Sparsemax saturation and the actual final accuracy certificate

The requested scope is the saturation mechanism and final accuracy. This
adds real-valued Lean theorems about the existing causal `sparseWeights`,
and kernel-checked counts of every actual exported final prediction. It
does not formally execute the complete floating-point PyTorch trajectory.
The local manuscript is arXiv:2211.11052v1, §§3.1 and 4. Sparsemax with
learned Q/K scores and prime-193 division are explicitly repository
extensions, not the paper's shared positional simplex training model.

The [saturation proof](../../../../src/Transformer/GPTMini/Sparsemax/Basic.lean)
uses the existing convex routing objective at its original score scale:
`sum(a²)/4 - sum(a*score)/2`. If an allowed winner exceeds every other
allowed score by at least one, the exact minimizer is its basis vector.
With a strict gap greater than one, continuity of the finitely many score
coordinates gives a neighborhood on which the entire sparsemax row is
constant. Its full Fréchet derivative over the real score vector is zero.
Any outer loss through that row has zero score derivative when its other
inputs are fixed; no loss smoothness or assumed Jacobian is needed.
The equality boundary is excluded from the derivative theorem.

The [outer-loss counterexample](../../../../src/Transformer/GPTMini/Sparsemax/Failure.lean)
uses scores `(2, 0)`, a desired row `(0, 1)`, and squared error. The actual
sparsemax row is `(1, 0)`, the loss is 2, and its full score derivative is
zero, while scores `(0, 2)` attain loss zero. Thus a convex inner inference
program does not guarantee a useful corrective score gradient for the
outer training objective. This does not prove that GPTMini globally stalls:
other rows, value paths, optimizer moments and weight decay still matter.

The read-only final-checkpoint CPU diagnostic uses the same native
300,000-update weights as the prediction certificate. It observes:

| Fixed-weight train batch | Layer | Strict unit-gap rows / supervised rows | Q gradient L2 | K gradient L2 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 0 | 1,586 / 4,096 | 1.170031e-7 | 1.611088e-7 |
| 512 | 1 | 265 / 4,096 | 6.274309e-8 | 9.758482e-8 |
| 48 | 0 | 147 / 384 | 1.120431e-7 | 1.635368e-7 |
| 48 | 1 | 26 / 384 | 7.565528e-8 | 1.283185e-7 |

Every observed singleton also has a strict gap, and its implemented
float32 score gradient is exactly zero. Instrumented and ordinary CPU
logits, loss and every parameter gradient agree exactly. Aggregate Q/K
gradients are nevertheless nonzero. These are two **training** batches
selected for sampler positions 300,001 and 300,010, both with fixed weights
from update 300,000 and no intervening optimizer updates. The measurements
do not establish that saturation causes the reported 34.056563% test accuracy.
The [earlier 280k diagnostic](sparsemax-routing-gradient-probe.md) remains
separate and uses different weights.

The [actual prediction archive](../../baselines/adamw_sparsemax_final_certificate_20261003/CPU-export-validation.json)
contains every train and held-out CPU prediction, plus EOS flags. The final
checkpoint SHA256 is
`c3f296b2f9ffd388451f4beefb358b184ad32da404658424c6685c9a935ac601`.
Model parameters, corpus and every frozen training/proof/paper fingerprint
are checked before export. The 37,056 distinct inputs exhaust all `x/y`
with `0 <= x < 193` and `0 < y < 193`. An independent NoTorch verifier
reconstructs both exact corpus fingerprints, checks disjoint coverage,
matches every packed Lean record against its CSV row, and checks the inverse
oracle against the multiplication correctness predicate.

The [kernel certificates](../../../../src/Transformer/GPTMini/Sparsemax/Certificate/Counts.lean)
use plain `decide` on every record, including operand bounds, EOS and
`y * predicted mod 193 = x`. Their exact counts depend on **no axioms**:
train 9,264/9,264 and held-out 9,465/27,792, with EOS correct everywhere.
The [rational accuracy results](../../../../src/Transformer/GPTMini/Sparsemax/Certificate/Results.lean)
prove train accuracy 1, held-out accuracy 9465/27792, the strict percentage
rounding interval `(34.055%, 34.065%)`, and failure of the 99% final target.
CPU and the archived GPU result have equal aggregate correct counts; no
per-record CPU/GPU prediction identity is claimed. Lean checks the supplied
table. The checkpoint-to-table PyTorch forward pass is an executed,
hash-linked measurement, not a Lean theorem about PyTorch semantics.

The affected-module and full builds pass. The full tree retains 157 `sorry`,
zero proved declarations resting on `sorry`, and zero extra axioms. All
new modules are warning-free and reachable from `Transformer.lean`.
Definitions, library statements, typeclass assumptions and transitive axioms
are retained in the [inspection log](../../baselines/adamw_sparsemax_final_certificate_20261003/statements-definitions-and-axioms.log).
Every changed statement was checked against the local source sections and
its explicitly stated extension. No frozen source, optimizer, scientific
budget or stability gate changes; the scheduled softmax pair remains active.

Recheck the supplied data and Lean proofs without a model checkpoint:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_sparsemax_final_certificate
lake build Transformer.GPTMini.Sparsemax
lake build
lake env lean scripts/Axioms.lean
python3 scripts/index.py
```

Re-exporting predictions or repeating the fixed-weight derivative diagnostic
requires the separately transferred ignored checkpoint, the pinned source
versions and fresh output paths. The archive retains both exact executed
programs and measurement receipts; neither inspection performs updates or
initializes CUDA.
