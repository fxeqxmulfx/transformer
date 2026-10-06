# Genuine sparsemax atomic training

Can answer-only training find the attained convex-mixture optimum from Lean,
escape a flat raw Q/K state, and transfer the method to Basis?

The physical model implements `matchingHeadOutput` and `matchingMixtureSample`
at Lean commit `83d985f`: independent shared Q/K/value token tables, causal
sparsemax and nonnegative unit-mass head coefficients. Heads are generated
from the current answer loss; no attention targets or preset head bank are
supplied. There is no FFN, position embedding, residual or output projection.

| Runs | Difference | Budget |
| --- | --- | --- |
| `order-columns-seed0` through `seed7` | Exact global physical-head pricing, corrective convex simplex QP | 1,000 outer updates each |
| `order-adamw-seed0` through `seed7` | Raw Q/K/value AdamW from the same uniform physical head | 1,000 updates each |
| `saturated-columns-seed0` through `seed7` | Exact pricing from Q=(1,1), K=(-1,1), routing only to token one | 1,000 outer updates each |
| `saturated-adamw-seed0` through `seed7` | Raw AdamW from the same strictly saturated sparsemax head | 1,000 updates each |
| `outside-columns` | Answer targets `(2,-2)` outside the cap-one output box | 1,000 outer updates |
| `order-search-seed0`, `saturated-search-seed0` | The same numerical Q/K pricing used on Basis, from both flat starts | 128 outer updates, four restarts x 65 pricing points each |
| `basis-columns-{depth,recall,parity}-seed0` | Four-dimensional free Q/K, cap four, approximate pricing | 128 outer updates, four restarts x 65 pricing points each |

The order task uses contexts `[0,1]` and `[1,0]`, final row and one output
channel, with targets `(1,1/2)`. Lean proves a uniform-query error floor of
`1/8` even with free values, and a genuine order-sensitive head attains zero.
Both methods start at that uniform floor: Q=K=0, values=(3/4,3/4). Splits
contain the same two formal observations; this is not a generalization test.

The saturated variants instead set Q=(1,1), K=(-1,1), with the same values.
Both contexts then read only token one's value. The score gap is two,
strictly inside sparsemax's one-route region, and its score Jacobian is zero.
Free values still cannot distinguish the two answers without a route change.
Seeds change the batch permutation; these prescribed initial heads are the
same across seeds, not eight independent random initializations.

For this task the complete physical price has an exact analytic oracle:
K=(-1/2,1/2), values=(-1,1), and Q=(-sign(g1),-sign(g0)) attain
`-abs(g0)-abs(g1)`, the lower bound from `matchingHeadOutput_bounds`.
It searches the full physical family on these two points. Its certificate
uses the same output gradient and loss scale as `matchingMixture_gap_certificate`.
The outside target has unavoidable optimal squared-error sum two.

Basis retains the standard easy data and answer loss, 20,000 training rows,
512 validation/test rows and the usual batches and stopping target. Original
values minimize their linear price exactly conditional on Q/K. Randomized
projected Adam searches Q/K; this is a nonconvex inner problem. The found-head
gap is reported separately from the global output-box certificate, which is
valid only for the current training batch and can be loose. Active-weight
optimization is convex, but this finite approximate solver is not guaranteed
to find the full-family global minimum. Storage allows 129 generated heads.
The single content-only layer also has expressivity limits distinct from the
ordinary two-layer softmax GPTMini used in `basis` and `basis_ansr`.

```sh
./make.py check experiments/convex_atomic
./make.py run experiments/convex_atomic
./make.py report experiments/convex_atomic
```

All 38 runs finished on 2026-10-06 UTC. The generated
[results.json](results.json) preserves their summaries and the structural
analysis; full checkpoints, histories and manifests remain in `runs/`.

| Formal task | Best squared-error sum | Numerical global gap bound after the budget |
| --- | ---: | ---: |
| Uniform start, exact column pricing, eight batch orders | 2.0954e-31 at update 2 | 3.3307e-16 |
| Strictly saturated start, exact column pricing, eight batch orders | 2.0954e-31 at update 2 | 3.3307e-16 |
| Raw AdamW, either start, eight batch orders each | 0.125 after all 1,000 updates | Not computed |
| Outside target `(2,-2)` | 2 at update 1 | 0 |
| Numerical pricing, either start, seed zero | 1.1752e-21 after 128 updates | 3.1196e-12 |

The searched variants do not use the analytic order oracle. The same Q/K
search as Basis also escapes both flat states, with 66,560 pricing forward
evaluations per run. Its output-box bound is nearly tight here despite the
oracle itself being approximate. It retains twelve active heads; no numerical
Caratheodory compression is implemented. Exact pricing uses at most three.
These are floating-point checks of the real theorem, not verified numerical
interval certificates.

| Basis task | Best update | Validation sequence accuracy | Test sequence accuracy | Test answer loss |
| --- | ---: | ---: | ---: | ---: |
| Depth | 48 | 41.60% | 40.23% | 0.36427 |
| Recall | 96 | 0% | 0% | 8.11267 |
| Parity | 48 | 56.05% | 54.69% | 0.38203 |

Each Basis run completed 128 outer updates and 66,560 pricing forwards.
None reached the 99% target. Their reported global bounds refer only to the
final training batch, not to the full training/validation criterion or the
selected checkpoint. Corrective optimization on a fresh small batch can
discard useful prior atoms; full-dataset convex convergence is not shown.

`analyze.py` groups supervised observations by query token and visible token
multiset, which completely determine every real content-only head response.
Conflicting labels give a model-class error floor independent of optimization.
On depth, these groups allow at most 82.74% validation and 83.44% test token
accuracy, with mean cross-entropy floors 0.25723 and 0.24723. Parity has no
such finite-split conflicts; that does not show this physical family spans
all functions of its signature.

The analysis also constructs two valid Basis MQAR examples by exchanging
two written values. The raw answer oracle changes the query's answer from
token 403 to 475, but its query and visible multiset stay identical. The
selected trained block produces identical float64 logits and token 518 in
both cases. Thus it cannot recognize all valid recall assignments. A context
encoder that preserves key/value binding is needed; increasing this head bank
alone cannot resolve that invariance.

```sh
cd python
uv run --locked python ../experiments/convex_atomic/analyze.py
```

The first 36 runs preceded corrections to two Lean declaration names in
source docstrings; executable behavior is unchanged. Their exact lab sources
and locked environment are preserved in `runs/engine_before_citation_fixes.tar.gz`
for strict checkpoint continuation. The two searched witnesses use the final
source text. Validation: all 189 lab tests pass, including eleven new physical,
certificate, training and resume checks; all experiment descriptions validate.
