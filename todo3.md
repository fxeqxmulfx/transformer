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
  numerically** — the table below shows it moving no wall at all.  It looks
  removable, because the data structure carries a sequence number that does the
  same job (§2).  It is not: §2a rebuilds the weights without it and all four
  programs stop, in both caches.  The perturbation is the margin that keeps two
  writes to one key apart after the key has been through a matvec, and nothing
  else in the release supplies one.

Remove the scale and the grid condition holds on every key; the perturbation
stays, and the score stays off the grid by less than half a unit.

Measured on their own heads,
`10^5` insert/query steps, hull versus their own brute reference:

    key range        10^3   10^4   10^5   10^6   10^7   2^26
    as shipped          0      0      1     11     38     14
    on the grid         0      0      0      0      0      0

and latest-write is still correct on every hit in both columns — on their own
heads, with keys handed in exactly.  That is what makes the perturbation look
redundant, and §2a is what shows the difference between handing a key in and
computing it in the model.

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
softmax is actually taken — `vm-rs/patches/on-the-grid.patch`, measured in §0a
and §8a.  Keep the `LATEST_ALPHA` term; §2a is why.  Then assert
`n < 94 906 266` (§4) and the head is exact up to that half-unit by
`ALM.FloatGrid.fp_walk_collects_of_grid`, which under exactly these two
conditions — grid-valued scores, breakpoints good to half a unit — proves the
merge loops of `HullHalf::query` collect the argmax set and nothing else.

### 0a. On the released model the scale is not what binds

The fix needs no weight surgery to test.  An argmax is invariant under
`q → q/s` for any `s > 0`, so dividing the query by `|qᵧ|` at query time is the
same change applied at the only place it can change anything.  That is
`--grid` in `vm-rs/alm-vm`, and against the released `model.bin` it

* reproduces **every** reference trace token for token, `hello` through
  `sudoku`'s 1 055 417 tokens — the invariance confirmed on the shipped
  weights rather than argued;
* removes **no** off-the-grid query.  The counts are identical in both modes:
  26 in `hello`, 788 in `addition`, 2863 in `collatz`, 265 in `fibonacci`,
  8093 in `min_cost_matching`, 65 727 in `sudoku`.

The reason is visible in the worst `ulp(score)/|qᵧ|` of a run — the quantity
the wall thresholds at `1`, reported per program by the driver:

    program              as shipped   at unit scale   ratio
    hello                   310.989         512.000   0.607
    collatz                 310.989         512.000   0.607
    fibonacci               310.989         512.000   0.607
    addition               1243.955        1024.000   1.215
    min_cost_matching      2487.911        2048.000   1.215
    sudoku                 4975.822        4096.000   1.215

The scale helps three programs and hurts three, and it can do nothing else:
the last column takes exactly two values, `2^33/s = 0.607` and
`2^34/s = 1.215`, according to where the score falls in its binade.  It is a
shuffle, not a loss.  What it
moves one way is the threshold, and that is the 22 % of §4a — a real cost, but
a quarter of a binade, not the difference between exact and wrong.

What binds instead is §4b.  A ratio of `512` says 512 consecutive key values
score identically; `sudoku` reaches `4096`, on the key
`[3334915682, −2.7804156515123814e18]`, the parabolic embedding of
`1 667 457 841`.  These heads are keyed on 32-bit WebAssembly values, and no
rescaling of the query touches the magnitude of the key.  So §0's claim holds
as stated — grid-valued scores are exact and the scale throws that away — but
on *these programs* the scale was never the binding constraint, and removing
it alone would leave every crossing in place.  Measured:
`vm-rs/alm-vm/tests/reference.rs::unit_scale_queries_reproduce_them_as_well`.

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

### 2a. Measured: the perturbation is a noise margin, and deleting it stops the machine

The fix above is a prediction, and it is wrong.  Rebuilt with
`LATEST_ALPHA = 0` and nothing else changed — same plan, same allocation,
`--plan plan.yaml` — the machine stops:

    cache    hello                  addition
    hull     503:  00 ≠ 57          1231:  2b ≠ 03
    brute    606:  01 ≠ 6f          2591:  00 ≠ 01

`collatz` and `fibonacci` fail too; 0 passed, 4 failed, in both caches.  Both
caches carry `last_seq`, both heads are flagged latest from the weight file,
and they fail at *different* tokens — which is already the shape of the answer.

Three further measurements say what the perturbation was doing.

1. **Its magnitude carries no information.**  Rebuilt at `LATEST_ALPHA` =
   `10^-6`, `0.1` and `0.49`, all four programs reproduce their traces token
   for token.  Six orders of magnitude and nothing changes: it is a margin, not
   a signal.

2. **The expression graph does not need it.**  `evaluator.py`'s brute
   attention resolves a tie by the highest sequence number, exactly as §2
   describes.  Run `hello` through it with `LATEST_ALPHA = 0` and the output is
   still correct.  So at the level of the graph, §2 is right: latest-write is
   the sequence number's job and the perturbation is redundant.

3. **The transformer's keys are not the graph's keys.**  Over `hello`'s first
   400 tokens, the brute head's gap from the winner to the best strictly-lower
   score, divided by `|qᵧ|`:

       weights              gaps in (0, 10^-9)   smallest positive gap
       as shipped                            0   1.015e-5
       LATEST_ALPHA = 0                    305   4.316e-15

   Two writes to the *same logical key* do not arrive at the head with the same
   key, because the key is a matvec through the residual stream and the matvec
   rounds.  With the perturbation they are separated by at least `10^-5` and
   the later one wins by construction.  Without it they are separated by
   `10^-15` in whichever direction the rounding fell, and `last_seq` is never
   consulted — there is no exact tie left to break.  On the released weights
   the latest tie-break never fires at all: **zero** ties on latest-flagged
   heads across `hello`, against 3 939 once the perturbation is gone.

So the perturbation is not a redundant restatement of `last_seq`.  It orders
nearly-equal keys *before* the comparison; `last_seq` catches only the residue
where the rounding happened to come out identical.  On the released model that
residue is empty, which means the release runs on the perturbation alone and
`HullMeta::last_seq` is dead code in it.  Its admissible window is

    rounding error of the key path  <  α · Δ inv_log_pos(p)  <  1/2

— the lower bound because it must dominate the matvec's noise, the upper
because it must not cross the unit gap between distinct integer keys.  Both
`10^-6` and `0.3` sit inside it; `0` does not.

**Fix, revised.**  Keep the perturbation.  §2's structural argument stands
untouched — a softmax head returns the mean of the tied payloads and cannot
resolve latest-write — but its operational conclusion does not: `last_seq` is
not a replacement for the perturbation, it is a fallback that the released
model never reaches.  What §2's float64 table then bounds is not a defect to
be removed but the trace length: `0.3/(p log² p)` shrinks with position while
the matvec's noise does not, so past some length the ordering is lost silently
and the answer becomes whichever the rounding prefers.  Where that length is
has not been measured — on the six reference programs the shipped model shows
no gap below `10^-5` — and it is the sharpest open question in this file.

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

### 3a. The hull head is exact only if no three keys are collinear

This is a second, independent gap between the two heads, found by porting the
hull to Rust (`vm-rs/alm-hull`) and running the two against each other on
*arbitrary* 2D keys rather than parabolic ones.  The envelope keeps vertices.
A key that is maximal at exactly one query and interior everywhere else is
erased the moment a later line covers it, and its payload — `vsum`, `vlast`,
`last_seq` — goes with it.  Both tie-breaks then answer with the wrong entry:
on the three collinear keys of the test, `Latest` gives 101 where the brute
head gives 102, and `Average` gives 100.5 where the brute head gives 101.

The machine is safe from this, but by accident of the embedding, not by
anything the code checks: `k ↦ (2k, −k²)` is strictly convex, so no three key
points are ever collinear.  Neither post nor code states the hypothesis, and
`HullKVCache` is offered as a general drop-in.

**Fix.** State it.  The hull head is equivalent to the brute head *for keys in
general position*; anything else needs the brute head or a hull that keeps
collinear vertices.  Witness and characterization:
`vm-rs/alm-hull/tests/differential.rs`.

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

The wall is in the representation, not in the arithmetic, and that settles
what a wider float would buy: nothing.  Scanning each query against its
neighbours for the first `q` whose own key stops being the strict argmax:

    rounded float64 scoring                first failure at q = 94 906 266
    exact scoring on the same stored pts   first failure at q = 94 906 266

the same number, to the unit.  The coordinate `−k²` stops being an exact
double at `k = 94 906 267`; at `q = 94 906 266` the stored coordinate of `q+1`
is `9 007 199 515 875 288` against a true `9 007 199 515 875 289`, and the
missing unit is exactly the gap that made `q` the argmax — so exact rational
arithmetic on the stored points ties them too.  Shewchuk expansions, an 80-bit
`long double` accumulator, `f128`: none of them recover a coordinate that was
rounded before it arrived.  Only a wider *embedding* moves the wall, and that
is a change to the weights.  (`f128` in Rust is nightly-only —
`#![feature(f128)]`, tracking issue 116909 — and software-emulated on x86-64:
measured 72× slower than `f64`.)  Measurement and test:
`vm-rs/alm-hull/src/exact.rs::the_wall_is_the_key_coordinate_not_the_arithmetic`.

Neither post states a bound, and the code has no assertion.  `int seq` in the
C++ head overflows at `2^31`, which is past this wall and therefore moot, but
it is equally unguarded.

### 4a. The shipped model queries at a scale, and the scale moves the wall

`94 906 266` is the unit-scale number.  The released `model.bin` scales its
hard-attention queries by `√2·10^10 = 14142135623.730951`, which multiplies the
score and the gap to the runner-up together — so the failure point moves, but
not by the factor, only by the residue of the cancellation.  `ulp` is relative
and the margin is not, and the residue is 22 %:

    query scale         first q whose winner is unseparable
    1                              94 906 266
    √2·10^10                        73 966 031

The second number is the one that bounds the shipped model.  It also means an
absolute test — "is the winning score past `2^53`" — is the wrong test in both
directions: it fires at token 799 of a 1034-token `hello` run whose every token
is correct, and it would still fire on a program that stays safely inside the
wall.  The scale-free form is `ulp(score) > |qᵧ|`: one integer step of the key
is `|qᵧ|` in the score, so a winner whose float64 neighbourhood is wider than
that step cannot be told from its runner-up.  That is the hypothesis
`S.δ < 1` of `ALM.FloatGrid.fp_eval_exact_of_grid` stated where the rounding
actually happens.  Implemented as `off_the_grid` in
`vm-rs/alm-hull/src/grid.rs`, wired through every head, reported per run by
`alm-vm`.

A guard of this shape must also know that a key which *cannot* win is not a
problem.  The compiler disables one by subtracting `BIG = 1e30` from its second
coordinate (`graph/core.py:8,314`, the `clear_key` mechanism); those keys sit
far off the grid by construction and an insert-time test counts thousands of
them per program while meaning nothing by it.  The winning score is the only
thing worth testing.

### 4b. A head keyed on a 32-bit value is past the wall from the first token

The wall is usually presented as a limit on trace length.  It is not: it is a
limit on whatever the head is keyed on, and this machine keys heads on
WebAssembly values, which run to `2^32 = 4.29·10^9` — 58× past the scaled wall
of `7.4·10^7` before a single token is generated.

Taken verbatim from a `hello` run, the first crossing:

    query  [4.763922097235979e18, 14142135623.730951]
    key    [673720322.0, -1.1347476806894592e17]

which is the parabolic embedding `v ↦ (2v, −v²)` of `v = 336 860 161`, not of
a position.  At that magnitude and scale the float64 maximum of `2qk − k²`
over the neighbourhood `v−60 … v+60` is attained at `v−3`, `v−1` and `v+1`,
and `v` is **not** among them: a query for `v` does not return `v`.  The
unresolvable neighbourhood is `ulp/scale ≈ 19.44` keys wide.  (Another
crossing in `collatz` and `fibonacci` is keyed on `538 976 288 = 0x20202020`,
four ASCII spaces — the same class.)

The released traces are all correct anyway, because the 32-bit values actually
present in one head are never within ~20 of each other.  Nothing arranges
that: it is a property of the programs, not of the construction, and no part
of the compiler checks it.  Test:
`vm-rs/alm-hull/tests/differential.rs::a_head_keyed_on_a_32_bit_value_is_past_the_wall_from_the_start`.

**Fix.** Not `n < 94906266` at insert — that bounds the wrong quantity and uses
the wrong constant.  Two things, at query, in both `hull_cache.py` and
`hull2d_cht.h`:

1.  `ulp(best_score) <= abs(qy)` on the winner, which is scale-free, survives
    `clear_key`, and catches §4b as well as §4 (measured: 26 crossings in
    `hello`, 788 in `addition`, 2863 in `collatz`, all real).
2.  A compile-time check that no head is keyed on an unbounded 32-bit value,
    or a documented statement that the compiler does not guarantee correctness
    for programs whose values collide inside one head.

Above the wall the construction is not slow or approximate, it is wrong, and
it fails silently.

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

## 8. The cumulative sum is not exact, and it is the main path

The post: "multiplying that average by `position` recovers the exact cumulative
sum".  It does not.  The head returns `vsum * (1.0/count)`
(`hull2d_cht.h`, `HullMeta::resolve`) and the FFN multiplies by `position`
(`graph/core.py`, `fetch_sum`), so the machine computes
`fl(fl(s · fl(1/p)) · p)` and asks it to be `s`.  Measured over byte-valued and
0/1-valued payloads, `p` up to `2·10^6`:

    round trips that fail        25.8 %      (first: s = 3, p = 5 → 3.0000000000000004)
    at p ≈ 10^6, non-integer     37.3 %

This is `stack_depth`, `cursor` and `call_depth` (`wasm/interpreter.py:316`) —
the instruction pointer among them — and `cursor` goes straight into a query as
`5·cursor + 1` (`:320`).  So on the main path the query handed to the lookup is
not an integer, and the hypothesis `q : ℤ` of `ALM.FloatGrid.fp_exact_of_grid`
does not hold of the running machine.  Everything §0 proves about exactness
applies to the keys and not to these queries.

What saves it is margin, and the margin is exactly one ulp wide:

    query off by 0 ulp    first key that loses to its neighbour = 94 906 266
    query off by 1 ulp                                           94 906 266
    query off by 2 ulp                                           50 331 647

The error out of `fetch_sum` is one ulp, so the machine runs on the last row it
can afford, and `5·cursor + 1 + i` adds one or two more roundings on top.

### 8a. Measured, and most of it was the scale's own rounding

Both halves of the paragraph above are predictions from reading the source.
The port measures them, by dividing every hard-attention query by its own
`|qᵧ|` — which leaves the value being looked up — and comparing that to the
nearest integer.  Done against the released `model.bin` it reports 22 % to
32 % non-integer queries and a worst offset of exactly one ulp in all six
programs, and **both of those numbers are artefacts**: on the released weights
`qᵧ` is the query scale `s = √2·10^10`, the weight rows were rounded *after*
being multiplied by `s`, and the reconstruction divides by `s` again.  Two
roundings the machine does not make are being counted.

The honest measurement is on weights rebuilt without the scale
(`vm-rs/patches/on-the-grid.patch`, §0), where `qᵧ = 1` and the query is the
number:

    program       not integers             share   worst offset        at
    hello             1 815 / 43 344        4.2 %  5.684e-14  1 ulp   260.99999999999994
    addition          7 807 / 183 120       4.3 %  1.137e-13  1 ulp   675.9999999999999
    fibonacci        10 769 / 382 284       2.8 %  9.095e-13  1 ulp   4316.000000000001
    collatz          95 027 / 1 872 654     5.1 %  9.095e-13  2 ulp   2573.000000000001
    min_cost_m.     421 519 / 7 485 408     5.6 %  1.819e-12  1 ulp   13876.000000000002
    sudoku        3 135 417 / 44 327 430    7.1 %  3.638e-12  1 ulp   16805.999999999996

Two things change against the shipped-model figures, in opposite directions.

The share drops by a factor of five, from 22–32 % to 2.8–7.1 %.  So roughly
four fifths of the queries that miss an integer on the released model miss it
because of the scale and not because of `fetch_sum` — which is an argument for
§0's first half that §0 does not make, and a better one than the 0.86 of a bit
it does make.  The `25.8 %` predicted at the top of this section was counting
the round trip in isolation; the running machine, unscaled, is well inside it.

The worst offset gets worse.  `collatz` reaches **two ulp**, at the query
`2573.000000000001`, which is the third row of the margin table above and not
the second: the wall for a query off by two ulp is `50 331 647`, half of
`94 906 266`.  The one-ulp uniformity reported on the shipped model was the
scale's doing — at magnitude `10^10` an ulp is `1.9·10^-6` and swallows the
composition of roundings that is visible at magnitude `10^3`.  So the sentence
this section wanted to close with is not "never past one ulp"; it is that the
composition of `fetch_sum`'s division and `5·cursor + 1 + i` reaches two ulp on
the fourth program anyone runs, and nothing bounds it at two.  Instrument:
`vm-rs/alm-hull/src/query.rs`; measurement:
`vm-rs/alm-vm/tests/reference.rs`.

**No fix, and that is the point.**  The division is not a mistake: cumulative
sums by uniform attention are the only constant-depth route, since the obvious
recurrence `c(p) = c(p−1) + δ(p)` needs the previous position's *computed* value
and a transformer layer can only read the previous layer, so the recurrence
costs one layer per token.  The average is forced, the division is forced, and
the inexactness is forced with it.

What changes is what has to be proved.  Exactness is not available on this path;
what is true is exactness **within a margin**, and `ALM.FloatHull.cmp_of_sep` is
already the right tool — it wants `δ₁ + δ₂ < |a − b|`, here `2k·ε + rounding <
1/2` with `ε = ulp(q)`.  The measurement above is that statement's content.
This is the most valuable thing in this file to formalize, and `todo2.md` §0
now carries it: it is the one place where the machine is not exact and cannot
be made so.

The other ReGLU identities degrade gracefully under the same perturbation,
which is worth stating in their favour.  `_make_multiply(a,b) = reglu(a,b) −
reglu(a,−b) = a·b` is exact for any real `b`.  `stepglu(a,b) = a·step(b ≥ 0)`
and the hat filter `reglu(x,d+1) − 2reglu(x,d) + reglu(x,d−1) = x·[d = 0]` are
exact for integer arguments and, at `b = −ε` or `d = ε`, return `a(1 − ε)`
rather than flipping: a relative error of one ulp, not a wrong branch.  The
construction is 1-ulp-Lipschitz rather than brittle.  Nobody has bounded the
composition of those errors, and that is the open question this section leaves.

## Order of work

1. §4's query-time guard, `ulp(best_score) <= abs(qy)`, and §6's differential
   test to hold it in place.  This is first because it is the only item that
   reports whether a given run was answered or guessed, and because running it
   is what turned up §4a and §4b.
2. §0's first half — `patches/on-the-grid.patch`, five lines of `weights.py`.
   It is what makes the grid theorem apply at all, and §8a is the measured
   argument for it: four fifths of the machine's non-integer queries are the
   scale's own rounding.  §0a is the argument against urgency: on the released
   programs it changes no answer and removes no crossing.  §0's second half is
   withdrawn — see §2a.
3. §7's two one-liners.
4. §2's second half — a text change, not a code change: the post stops
   claiming a softmax head does latest-write.  The single-writer alternative is
   closed, for the reasons in §2.  What replaces the deletion §2 asked for is
   §2a's open question: measure the trace length at which the perturbation
   drops below the key path's own rounding, because that is where latest-write
   fails silently.
5. §5 — prove the lower bound first, since it decides whether there is anything
   to fix.
6. §8 — nothing to fix in the code; the work is the margin theorem, and it is
   the one that decides how much of §0's wall the machine actually keeps.

What is *not* on this list is the Lean side.  None of these findings belongs in
`src/`: their use is that they fix hypotheses, and the hypotheses they fix
(`S.δ < 1`, `u · M < 1/2`, the unit lattice gap) are already stated in
`ALM.FloatGrid`, `ALM.FloatHead` and `ALM.FloatHull`.  What §0 shows is that
those three conditions are not idealizations of the implementation — they are a
complete specification of what it has to stop doing.
