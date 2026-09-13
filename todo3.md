# todo3 — defects in the released ALM implementation, and the fixes Lean dictates

`todo2.md` §0 lists what the ALM authors *asserted without proof*.  This file
lists what is wrong with the code they released, as code — and, since the
implementation is to be corrected from the Lean side, what each defect turns
into once the corresponding theorem is read as an instruction.

Every item therefore carries a **Fix**, and every fix names the theorem in
`src/Transformer/ALM/` that licenses it.  The fixes are measured, not proposed:
§0 gives the before/after on their own code.

Source: `transformer-vm/` (gitignored), the repository the companion post
releases, read 2026-09-14.  Every number below was measured; each item says
how.  Where a check was run outside this repository it is a self-contained C++
or Python program against their own headers — `hull2d_cht.h` compiles
standalone with `g++ -std=c++17`, and `HardAttentionHead` versus their own
`BruteAttentionHead` is their own equivalence claim, tested on their own key
construction (`graph/core.py`, `_to_2d_key`, `KEY_OFFSET = 0`) at their own
scale (`weights.py`, `HARD_K = 1e10`, times `sqrt(d_head) = sqrt 2`).

## 0. One change fixes 1–4: keep the score on the integer grid

Items 1, 2, 3 and 4 below look like four defects.  They are one, and Lean names
it.  `ALM.FloatGrid.fp_eval_exact_of_grid` says a float score routine is
*exact* as soon as two conditions hold: the computed score lands on the integer
grid, and its error is below one unit.  `fp_exact_of_grid` discharges both for
integer keys at an integer query — `2qk − k²` is an integer, and float64 holds
integers exactly below `2^53`.  Exactness is free; it has to be thrown away.

The released code throws it away twice, but not equally:

- `weights.py:431-432, 491-492` multiply the **query** by `HARD_K · √2 ≈
  1.41·10^10`.  `HARD_K` is a softmax temperature (`weights.py:22`).  The hull
  path is an argmax, where a positive scale changes no comparison and buys
  nothing — it only takes `2qk − k²` off the grid.  **This one costs range**,
  0.86 of a bit, and it is the whole of the measurable damage.
- `graph/core.py`, `_to_2d_key`, adds `LATEST_ALPHA · inv_log_pos(p)` to `ky`.
  Off the grid by construction, but bounded by `0.433` against a key gap of
  `1`, so it decides every comparison correctly and **costs nothing
  numerically** — the table below shows it moving no wall at all.  It has to go
  for the grid theorem to apply and for the softmax path of §1 to have a gap,
  not because it is doing harm today.

Remove both and the grid condition holds exactly.  Measured on their own heads,
`10^5` insert/query steps, hull versus their own brute reference:

    key range        10^3   10^4   10^5   10^6   10^7   2^26
    as shipped          0      0      1     11     38     14
    on the grid         0      0      0      0      0      0

and latest-write is still correct on every hit in both columns, because it
never came from the perturbation in the first place (§2).

The exactness wall moves with it.  Scanning for the first query that loses to
its own neighbour `q ± 1`:

    perturbation + HARD_K (as shipped)      q = 52 301 885   (2^25.64)
    HARD_K alone                            q = 52 301 885   (2^25.64)
    on the grid                             q = 94 906 266   (2^26.50)

`94 906 266 = ⌈√(2^53)⌉` exactly — the grid bound is attained to the unit, so
`fp_eval_exact_of_grid` is not merely sufficient here, it is sharp.  The
scaling costs 0.86 of a bit; the perturbation costs nothing, being already
below the resolution of the numbers it is added to.

The hull's own arithmetic needs no change.  `ALM.FloatHead.half_le_key_dist_mid`
proves that an integer key is at least `1/2` from every breakpoint, so
`fp_head_output_of_int` asks only `u · M < 1/2`; the breakpoints run in
`long double` (`u = 2^-64`) on keys below `2·10^8`, which is `u · M ≈ 10^-11`.
`ALM.FloatHull.fp_longDouble` says the same about the erase loop up to `2^56`.
The defect was never in the hull — it was in the score handed to it.

**Fix.** Do not scale the query on the hard-max path; scale only where a
softmax is actually taken.  Delete the `LATEST_ALPHA` term from `_to_2d_key`.
Then assert `n < 94 906 266` (§4) and the head is exact by
`ALM.FloatGrid.fp_walk_collects_of_grid`, which under exactly these two
conditions — grid-valued scores, breakpoints good to half a unit — proves the
merge loops of `HullHalf::query` collect the argmax set and nothing else.

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
executed line.

**Fix.** The claim is recoverable, and §0 is what recovers it.  `todo2.md` §0
observes that at `HARD_K = 1e10` the softmax stops agreeing with the hard-max
around `10^6` tokens — but only because the perturbation manufactures score
gaps of order `10^-16`.  On the grid the gap is `1`
(`ALM.Lattice`, used by `ALM.SparseSoftmax.sparse_softmax_output_close_of_gap`),
so `softmax_mass_outside_le` bounds the loser mass by `(n−1) e^{−β}` with
`β = HARD_K · √2`: nil at every `n` the wall of §0 permits.  The fix is
therefore to remove the perturbation first, then write the `forward` and test
it — not to raise `HARD_K`.

## 2. Latest-write is implemented outside the model

The post says the latest write wins because "we add a small position-dependent
perturbation to each key".  In the released system it wins because the data
structure carries an integer sequence number: `HullMeta::last_seq`
(`attention/hull2d_cht.h:46-56`), set from a counter incremented per layer per
token in `transformer.cpp` and passed through `hull_cache.py:51`.

This is the right design, and the Lean side endorses it: `Meta.resolveLatest`
is what `ALM.SoftmaxLatestMass` proves the log's answer against.  The defect is
not the mechanism but the description — and the consequence, which the post
does not draw, that a transformer without it cannot run this interpreter.

Because it is not a transformer operation.  Nothing in the weights computes it,
and a real softmax head has no access to it.  The perturbation it replaces is
numerically dead well inside the advertised regime: it is added to `−k²` in
float64 and survives only while `p log² p · k² ≲ 1.4·10^15`.

    k = 10^4 : lost at p ≈ 10^6      k = 10^5 : lost at p ≈ 10^4
    k = 10^6 : lost at p ≈ 10^3

Checked by evaluating `_to_2d_key` in float64 and comparing the keys of two
consecutive writes to the same logical key for bit equality.  A WASM stack
pointer passes `k = 65536` on the first frame, so this is the ordinary case,
not a corner.

**Fix.** Delete the perturbation and keep `last_seq`; the measurement in §0
confirms latest-write is unaffected, because the aggregate was already doing
all the work.  `ALM.SoftmaxLatestMass.softmax_head_resolves_latest_of_int` is
the licence: with integer keys the tie gap is `1` with no perturbation
supplied, and the latest payload is recovered from an integer sequence number
carried in the aggregate — which is `HullMeta` exactly.

That theorem also fixes the honest statement of the claim.  Its bound is
`((n−2) e^{−β}/2)·C + ‖V_b − V_c‖/2`, and the second term does not shrink with
`β`.  A softmax head cannot resolve latest-write at all; it returns the mean of
the tied payloads, and no temperature changes that.

There is no way out of this inside the construction, and the reason is
structural rather than numerical:

- **CALM has no write.**  `fetch` (`graph/core.py:328`) declares a lookup gate;
  the key expression is then evaluated at *every* position, so every token
  writes to every gate at every step.  There is no channel object to attach a
  single-writer check to — the gate is the channel — and nothing is conditional.
  `clear_key` is not a conditional write: it subtracts `BIG = 1e30` from `ky`,
  so the write happens and is made to lose.
- **Duplicate keys are the mechanism, not the corner case.**  Every mutable
  cell in the WASM interpreter is a repeated write under one key: memory bytes
  at `key = memory_write_address` (`wasm/interpreter.py:423-427`), locals at
  `key = LOCAL_STRIDE·call_depth + 4·immediate + byte_index` (`:388-394`),
  stack slots at `key = stack_depth` (`:397-412`).  A single-writer restriction
  would forbid storing to the same address twice, which is to forbid mutable
  memory.
- **The hat filter does not cover it.**  `memory_byte_dirty_position` is the
  stored *address*, not a position, so the filter at `:429-433` rejects a read
  whose address does not match — but two writes to the same address both match,
  and their mean passes as a valid byte.  Latest-write is load-bearing for
  memory correctness and unprotected.
- **The integer route does not scale.**  Folding recency into the key as
  `k·S + p` with `S > n` needs `(k·S)² < 2^53`, i.e. `k·S < 9.49·10^7`: at a
  `10^6`-token trace that is 95 distinct logical keys.  Latest-write cannot be
  bought back on the grid.

And the sentence is not merely false, it is unaffordable.  Recency in the key
has to fit inside the width-`1` score gap between distinct integer keys, so `n`
recencies need spacing `≤ 1/n`, and that spacing must stay visible beside a
score of size `k² ~ n²`.  One float then carries two quantities three decades
apart and the wall becomes `n³ < 2^mantissa` instead of `n² < 2^53`.  Measured,
as the largest trace on which the sentence holds:

    even spacing 1/n, float64                          262 143
    their spacing 0.3/(p log² p), float64               30 730
    even spacing 1/n, long double (64-bit mantissa)   2 516 581
    for comparison, §0's fix, recency not in the key  94 906 266

The cube-root law is confirmed at both mantissa widths, so binary128 would be
needed — `n ≈ 2^37.7` — merely to beat what float64 already gives without
recency in the key.  Making the claim true costs a factor of 362, and their own
spacing is a further factor of 8.5 worse than an evenly spaced one.  The Sudoku
demo at 5.4·10^6 tokens does not run in any row of that table.

So the disjunction has one branch.  The post has to drop "the latest one then
scores strictly highest": latest-write is not in the weights and cannot be put
there.  Their own reference semantics already says so — `evaluator.py:158-165`
resolves a tie by the highest sequence number, never by the perturbation, which
is thus absent from the specification, absent from both C++ heads, and present
only in the emitted weights.

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
at `q` produces between the keys `q−d` and `q+d`, the two procedures disagree.

Every disagreement observed was on a miss; on a stored key both heads were
right at every magnitude tested.  That is a caveat on the severity, not on the
defect: two procedures asserted equal are not equal, and the interpreter leans
on misses deliberately.

**Fix.** §0 — the divergence is gone at every key magnitude once the score is
on the grid, with no change to the hull.  What remains is that
`hull_cache.py` and `standard_cache.py` are two implementations of one
specification with no differential test between them; the grid condition is
what makes such a test meaningful, since `ALM.FloatGrid.fp_walk_collects_of_grid`
then says the two must agree exactly rather than approximately.

## 4. Float64 ends at 94 906 266 tokens, and nothing says so

The score `2qk − k²` is an integer and float64 holds integers exactly only
below `2^53`, so exact retrieval needs `n < √(2^53) = 94 906 266`.  Measured on
`HardAttentionHead` with `key = position`, sampled queries:

    n        10^4   10^5   10^6   10^7   10^8
    wrong       0      0      0      0   2.1 %

with the first failure of the shipped arithmetic at `q = 52 301 885` and of
grid arithmetic at `q = 94 906 266` (§0).  The Sudoku demo is ~5.4·10^6 tokens
(3 minutes at ~30k tok/s), a factor of 10 below the shipped wall.

Neither post states a bound, and the code has no assertion.  `int seq` in the
C++ head overflows at `2^31`, which is past this wall and therefore moot, but
it is equally unguarded.

**Fix.** One assertion, at insert, in both `hull_cache.py` and
`hull2d_cht.h`: `n < 94906266`.  This is the item that "ends in a number", and
the number is now exact rather than an order of magnitude: it is
`⌈√(2^53)⌉`, the hypothesis `S.δ < 1` of
`ALM.FloatGrid.fp_eval_exact_of_grid` failing at the point where the integer
grid of float64 stops having spacing `1`.  Above it the construction is not
slow or approximate, it is wrong, and it fails silently.

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

**Fix.** None available from the current Lean corpus, and this is the item
where that matters most.  `ALM.HullCost.hullQuery_cost_total` prices a *query*
and `hullQuery_collects` says what it collects; nothing prices the *cache*.
The statement worth proving before any code changes is a lower bound — that an
append-only lookup machine whose every read is a `key = position` lookup must
retain `Ω(n)` state — because if it is true, this is not a defect to fix but a
property of ALM to state, and `ALM.FixedDim.OVHard_needs_growing_dimension` is
the nearest thing already in the repository to that kind of argument.

## 6. The headline demos have no test

`tests/` runs the graph evaluator — exact Python arithmetic, not the weights —
on `hello`, `collatz` and `fibonacci`.  The only test that builds a transformer
and generates from it is `test_model_inference_hello`, capped at 5000 tokens.
`sudoku` and `min_cost_matching`, the two programs both posts lead with, are in
`examples/manifest.yaml` and in no test.

So the "millions of steps, 100 % accuracy" regime — where items 1, 3 and 4
begin to bite, three orders of magnitude above the test cap — carries no
regression at all.

**Fix.** The differential test §3 asks for, run at the wall of §4 rather than
at 5000 tokens: hull against brute against `standard_cache`, on `key =
position`, at `n = 10^8`.  On the grid all three must agree exactly up to
`94 906 266` and the assertion must fire above it.  That single test covers
items 1, 3 and 4 at once.

## 7. Smaller things

- `hull2d_cht.h:25-28` documents the opposite of what it does: "TieBreak::LATEST
  is accepted but behaves like AVERAGE", while `HullMeta::resolve` implements
  LATEST via `vlast`.  The whole latest-write mechanism is behind a comment
  saying it is not there.
- `add_line` discards a dominated line together with its `HullMeta`, and
  `isect` (`hull2d_cht.h:139`) calls a line unnecessary on `x->p >= y->p`, so
  equality discards too.  This is correct only while the keys are in strictly
  convex position; `clear_key` puts points at `ky ≈ −10^30` into the same hull
  and the invariant is nowhere stated or checked.
- `transformer.cpp:299` shadows the `Clock::now()` variable `tb` with a
  `TieBreak tb` inside the head loop.  Harmless as scoped, but the timing
  arithmetic three lines later reads the outer one.

**Fix.** The first and third are one line each.  The second is the one place
where §0 does not reach: `clear_key` subtracts `BIG = 1e30` from `ky`
(`graph/core.py`), which is off the grid by four orders of magnitude more than
`HARD_K` was, so a cleared entry satisfies no hypothesis of
`ALM.FloatGrid`.  It is harmless only because such an entry always loses.  To
keep the grid argument whole, the cleared marker has to leave the score: the
interpreter already owns a ReGLU hat filter (`wasm/interpreter.py:428-433`)
that discards an out-of-range lookup, and a cleared entry is the same kind of
event.  That is a design change, not a patch, and it is the only one on this
list.

## Order of work

1. §0, both halves — one conditional in `weights.py` and one deleted term in
   `graph/core.py`.  Fixes 1, 2, 3 and half of 4, and the measurement above
   says so on their own code.
2. §4's assertion, and §6's differential test to hold it in place.
3. §7's two one-liners.
4. §2's second half — a text change, not a code change: the post stops
   claiming a softmax head does latest-write.  The single-writer alternative is
   closed, for the reasons in §2.
5. §5 — prove the lower bound first, since it decides whether there is anything
   to fix.

What is *not* on this list is the Lean side.  None of these findings belongs in
`src/`: their use is that they fix hypotheses, and the hypotheses they fix
(`S.δ < 1`, `u · M < 1/2`, the unit lattice gap) are already stated in
`ALM.FloatGrid`, `ALM.FloatHead` and `ALM.FloatHull`.  What §0 shows is that
those three conditions are not idealizations of the implementation — they are a
complete specification of what it has to stop doing.
