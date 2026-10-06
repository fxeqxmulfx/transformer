# Convex matching with preserved key/value binding

Can a causal encoder retaining each token's predecessor remove the MQAR
binding obstruction while preserving convex atomic-mixture training?

`PairedMatching` splits its freely learned Q/K width into two roles. One
key half uses the current token, the other its predecessor. Queries and
original values use the current token. This stores token tables rather than
a dictionary of every token pair. The encoder supplies context, not target
attention routes. Mixture coefficients still form a probability simplex;
new physical Q/K/value heads are selected from the entire numerical box.
Raw head search remains nonconvex, and finite search is not a global oracle.

| Runs | Encoder | Pricing / training | Budget |
| --- | --- | --- | --- |
| `paired-exact-seed0..3` | Current and preceding keys, total width 4 | Exact two-example binding price, corrective simplex QP | 1,000 outer updates |
| `paired-search-seed0..3` | Same | Four numerical restarts, 64 Q/K steps, exact conditional values | 128 outer updates; 66,560 pricing forwards |
| `content-search-seed0..3` | Original content-only, width 4 | Same numerical search | Same 128 updates |
| `content-adamw-seed0..3` | Original content-only, width 4 | Raw AdamW | 1,000 updates |
| `basis-content-recall-seed0` | Original content-only, width 8 | Same numerical search | 512 outer updates; 266,240 pricing forwards |
| `basis-paired-recall-seed0` | Current and preceding keys, total width 8 | Same numerical search | Same 512 updates |

The scalar contexts are `[0,1,3,2,4,1]` and `[0,1,4,2,3,1]` with answer
targets `(0,1)`. Any content-only head or mixture has identical outputs on
the two; squared-error sum is at least 1/2. A physical paired head selects
the original value immediately after key one and attains zero. Both splits
use the same two observations. The four seeds change batch order and
numerical pricing, not the prescribed constant initialization.

Basis keeps its 20,000 easy MQAR training rows, 512 validation rows and 512
test rows, data seed one, batch 64, and original answer cross entropy. Both
arms use cap four, identical parameter counts, head capacity 513 and four
CPU threads with eager CUDA training. A measured two-update pilot with the
paired model took 8.14/6.74 seconds on CPU and 3.68/3.12 seconds on the GTX
1050; the pilot is a timing measurement, not a training result.

All 18 runs finished on 2026-10-06. The encoder removes the structural
obstruction, but numerical head search still fails easy Basis recall.
The aggregate, including the constructed capacity probe and archived
softmax comparison, is saved in [results.json](results.json).

| Scalar binding runs | Best squared-error sum, across four seeds | Finding |
| --- | --- | --- |
| Paired, exact pricing | `6.16e-32` | Fits at the first recorded evaluation, update 10; final global gap `2.22e-16` |
| Paired, numerical pricing | `3.85e-31` to `1.45e-29` | All seeds fit without route labels or the analytic oracle |
| Content-only, numerical pricing | `0.5` | Reaches the formally proved sharp structural floor |
| Content-only, raw AdamW | `0.5` | Cannot overcome the same invariance |

The seven new tests check the physical causal forward, both key roles'
gradients, original values, exact global pricing, answer training and exact
checkpoint continuation. The small exact-pricing test fits within three
updates. The numerical search's reported global bounds are separate output
box bounds, not certificates that its head search is globally optimal.
For content-only search the loose bound remains two, even though the
structural theorem already proves that error one half is optimal.

| Basis model / optimizer | Updates | Wall time | Test token accuracy | Test sequence accuracy |
| --- | ---: | ---: | ---: | ---: |
| Content-only width 8 / numerical columns | 512 | 1,622 s | 0.415% | 0% |
| Paired width 8 / numerical columns | 512 | 1,843 s | 0.732% | 0% |
| Archived softmax GPTMini / AdamW, seed 0 | 1,650 | 207 s | 100% | 100% |
| Archived softmax GPTMini / AdamW, seed 1 | 2,450 | 287 s | 99.854% | 98.828% |
| Archived softmax GPTMini / AdamW, seed 2 | 1,800 | 214 s | 99.780% | 98.242% |

Both column models select one active head. Their selected validation losses
are 7.868 and 7.382, and test losses 7.874 and 7.405. On the valid raw-oracle
swap example with answers 403 and 475, both trained models still give
identical logits. Context-sensitive matching is available in the paired
family, but its numerical optimizer did not discover it on this witness.

The three archived softmax runs are from [basis](../basis), commit
`bd63e50`, and all train/validation/test split fingerprints match this
experiment exactly. They pass the 99% validation sequence criterion in
207–287 seconds on one compiled CPU thread, well within the column models'
measured wall times on CUDA. They are larger two-block width-64 GPTMini
models with four softmax heads, QKNorm, XSA, FFN and RoPE, rather than a
matched replacement of this width-eight head. Equal FLOPs were not measured.
At update 500, the nearest recorded point below 512, they have 11.06–11.72%
validation token accuracy and zero sequence accuracy. They require more
ordinary updates while spending much less elapsed time. One column update
contains 520 physical pricing forwards before correction and evaluation.

The separate capacity probe constructs a **single width-eight cap-four
paired head**, exactly the trained paired model's width and cap. Four
integer predecessor channels encode 256 keys with common squared norm 30
and distinct inner products at most 29. Actual causal sparsemax therefore
selects the original value immediately after the matching key. Its freely
stored original value rows put +4 on their own output class and -4 on the
others; no expanded pair dictionary or target attention labels are used.

That constructed head has 100% token and sequence accuracy on all 512
validation and 512 test examples, and correctly distinguishes the swapped
answers with logit difference eight. Its cross entropy, 0.168475, matches
the elementary universal floor `log(1 + (vocab - 1) * exp(-8))` for logits
in the cap-four box. It demonstrates capacity; its Q/K and values were
constructed, not learned, and its state never enters training or pricing.
The probe's 256 codes use the exact decoder proved in
`PairedRecallCodes.lean`, a subset of 384 signed permutations of (1,2,3,4).

The formal repair is in commit `4ef16e5`: old-head permutation invariance
and the sharp floor, actual paired causal scores/output, continuity, box
bounds, convex mixture training, attained head pricing, exact binding
pricing, compact key-code geometry and one-head recall under a unique
matching-predecessor premise. Commit `a155424` adds an answer-fitting
necessity theorem showing that a fitting feasible mixture contains a head
distinguishing the two assignments even when original values remain freely learned.

The unique-predecessor premise covers easy MQAR here. Latest-write lookup,
arbitrary unseen language, global numerical pricing and convergence over
the full training dataset remain unresolved. Correction currently uses a
fresh batch; both that limitation and search over flat sparsemax regions
remain candidates for the failed large-task training. These runs do not
isolate their individual contributions.

Validation: all 196 Python tests pass; all 18 experiment definitions check.
Every run's recorded lab-source hashes match the committed implementation.
Lean passes the full build, audit, generated index and forbidden checks:
157 existing `sorry`, zero results resting on them, zero extra axioms,
zero vacuous claims and zero placeholders.

Use `./make.py check experiments/convex_binding` and
`./make.py report experiments/convex_binding` to inspect the study. With
the local runs present, regenerate the analysis from `python/` using
`uv run --locked python ../experiments/convex_binding/analyze.py`.
