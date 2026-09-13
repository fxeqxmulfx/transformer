# todo3 — defects in the released ALM implementation

`todo2.md` §0 lists what the ALM authors *asserted without proof*.  This file
lists what is wrong with the code they released, as code.  The two are
different kinds of debt and the same finding can appear in both: §0 wants a
theorem, this file wants a bug report.

Source: `transformer-vm/` (gitignored), the repository the companion post
releases, read 2026-09-14.  Every number below was measured, not estimated;
each item says how.  Where a check was run outside this repository it is a
self-contained C++ or Python program against their own headers — `hull2d_cht.h`
compiles standalone with `g++ -std=c++17`, and `HardAttentionHead` versus their
own `BruteAttentionHead` is their own equivalence claim, tested on their own
key construction (`graph/core.py`, `_to_2d_key`, `KEY_OFFSET = 0`) at their own
scale (`weights.py`, `HARD_K = 1e10`, times `sqrt(d_head) = sqrt 2`).

Ordered by how far each one is from the claim it breaks.

## 1. No path in the release runs softmax attention

The README's first sentence is "A standard softmax-ReGLU transformer whose
weights are computed **analytically**", and the first post shows a
`VanillaTransformer` with a `forward` that softmaxes over a causal mask.

There is no `forward` in the repository — `grep -rn 'def forward'` returns
nothing.  The only entry point is `generate_with_cache`
(`model/transformer.py:41`), which calls `HullKVCache`, which is hard-max.  The
C++ engine that produces the advertised ~30k tok/s is hard-max as well
(`model/transformer.cpp:300-306`), and its `--brute` flag is a linear scan for
the same `argmax`, not a softmax.  Softmax exists once, in
`attention/standard_cache.py`, reachable only behind `--nohull`, and no test
touches it.

So the claim that this is a standard softmax transformer is supported by no
executed line.  §0's first item explains why it could not be: at `HARD_K = 1e10`
the softmax stops agreeing with the hard-max around `10^6` tokens, which is the
regime the post advertises.

## 2. Latest-write is implemented outside the model

The post says the latest write wins because "we add a small position-dependent
perturbation to each key".  In the released system it wins because the data
structure carries an integer sequence number: `HullMeta::last_seq`
(`attention/hull2d_cht.h:46-56`), set from a counter incremented per layer per
token in `transformer.cpp` and passed through `hull_cache.py:51`.

That is not a transformer operation.  Nothing in the weights computes it, and a
real softmax head has no access to it.  The perturbation it replaces is
numerically dead well inside the advertised regime: it is added to `-k²` in
float64 and survives only while `p log² p · k² ≲ 1.4·10^15`.

    k = 10^4 : lost at p ≈ 10^6      k = 10^5 : lost at p ≈ 10^4
    k = 10^6 : lost at p ≈ 10^3

Checked by evaluating `_to_2d_key` in float64 and comparing the keys of two
consecutive writes to the same logical key for bit equality.  A WASM stack
pointer passes `k = 65536` on the first frame, so this is the ordinary case,
not a corner.

## 3. The hull cache is not equivalent to the attention it replaces

`HullKVCache` is offered as a faster way to get the same answer.  It does not
always get the same answer as their own `BruteAttentionHead`.  Both were run on
the same insert/query stream, `10^5` steps, keys uniform in a range, queries
uniform (so ~63 % miss — high, but a miss is a designed-for event: the hat
filter at `wasm/interpreter.py:428-433` exists to absorb one):

    perturbation   scale     disagreements
    0.3            HARD_K        1
    0              HARD_K     3581      (5.6 % of misses)
    0.3            1             0
    0              1             0

The cause is the scale, not the hull: `HullHalf::query` decides the argmax by
comparing `m = qx/qy` against breakpoints in `long double`
(`hull2d_cht.h:277-280`), while the brute scan — and any real attention head —
decides it by comparing rounded float64 scores.  On an exact tie, which a miss
at `q` produces between the keys `q-d` and `q+d`, the two procedures disagree.

The perturbation of §2 is what suppresses this, by breaking the ties.  So it
carries a second load the post never mentions, and it degrades exactly as §2
predicts — same run, perturbation on, disagreements against key magnitude:

    keys ≤ 10^3   10^4   10^5   10^6   10^7
             0      0      1     11     38

Correctness therefore rests on a separation hypothesis that the code neither
states nor checks.  In Lean that hypothesis is already written down — it is the
`sep` of `ALM.FloatHull.cmp_of_sep`, used by `fp_query_branch`.

## 4. Float64 ends at 2^26 tokens, and nothing says so

The parabolic key needs the gap `(q-k)² ≥ 1` to be visible beside `q²`, so
exact retrieval needs `ulp(n²) < 1`, i.e. `n < 2^26 ≈ 6.7·10^7`.  Measured on
`HardAttentionHead` with `key = position`, sampled queries:

    n        10^4   10^5   10^6   10^7   10^8
    wrong       0      0      0      0   2.1 %

A direct scan of the score puts the first failure at `q ≈ 7.7·10^7`; at
`q = 6·10^7` the gap is already exactly one ulp.  The Sudoku demo is ~5.4·10^6
tokens (3 minutes at ~30k tok/s), a factor of 13 below the wall.

Neither post states a bound, and the code has no assertion.  `int seq` in the
C++ head overflows at `2^31`, which is past this wall and therefore moot, but
it is equally unguarded.

## 5. Space is linear in the trace and never released

Measured RSS of one `HardAttentionHead` fed `key = position`: **128 bytes per
token**, exactly linear to `2·10^6` tokens.  The upper hull retains every point
— keys on a parabola are all vertices — and the lower hull retains 2.

At least thirteen lookups in `wasm/interpreter.py` use `key=position`
(lines 304, 330, 334, 415-417, 481; the five-value one costs three heads), so
the floor is ~1.7 KB per generated token, nothing of it reclaimable.  The
Sudoku demo is then several GB of hulls alone.

The companion concedes this in half a sentence ("the memory requirements for
computation grow with the number of tokens").  It is worth stating plainly: the
claim being made is "a real computer updates a compact state with roughly
constant work per instruction", and the construction answers with constant time
and linear space.  A RAM machine's state does not grow with its runtime.

## 6. The headline demos have no test

`tests/` runs the graph evaluator — exact Python arithmetic, not the weights —
on `hello`, `collatz` and `fibonacci`.  The only test that builds a transformer
and generates from it is `test_model_inference_hello`, capped at 5000 tokens.
`sudoku` and `min_cost_matching`, the two programs both posts lead with, are in
`examples/manifest.yaml` and in no test.

So the "millions of steps, 100 % accuracy" regime — where items 1, 3 and 4
begin to bite, three orders of magnitude above the test cap — carries no
regression at all.

## 7. Smaller things

- `hull2d_cht.h:25-28` documents the opposite of what it does: "TieBreak::LATEST
  is accepted but behaves like AVERAGE", while `HullMeta::resolve` implements
  LATEST via `vlast`.  The whole latest-write mechanism is behind a comment
  saying it is not there.
- `add_line` discards a dominated line together with its `HullMeta`, and
  `isect` (`hull2d_cht.h:139`) calls a line unnecessary on `x->p >= y->p`, so
  equality discards too.  This is correct only while the keys are in strictly
  convex position; `clear_key` puts points at `ky ≈ -10^30` into the same hull
  and the invariant is nowhere stated or checked.
- `transformer.cpp:299` shadows the `Clock::now()` variable `tb` with a
  `TieBreak tb` inside the head loop.  Harmless as scoped, but the timing
  arithmetic three lines later reads the outer one.

## What to do with this

Nothing here is a theorem, and none of it belongs in `src/`.  Its use is that
it fixes the hypotheses: §3 and §4 say which separation and which magnitude a
float-level statement about their head has to assume, and `ALM.FloatHull` and
`ALM.FloatGrid` are where those assumptions already live.  The cheapest item
that turns into Lean is §4 — see the corresponding bullet in `todo2.md` §0.
