# vm-rs — a Rust port of `transformer-vm`

A port of the Percepta `transformer-vm` release (Apache-2.0, vendored under the
gitignored `transformer-vm/`) whose purpose is to carry the corrections the Lean
formalization licenses.  `todo3.md` is the fix list; this is where the fixes go.

## Crates

| crate       | what it is                                                |
|-------------|-----------------------------------------------------------|
| `alm-hull`  | the 2D hard-attention KV cache (`attention/hull2d_cht.h`)  |
| `alm-model` | the transformer itself (`model/transformer.py`, `.cpp`)    |
| `alm-vm`    | the driver: load `model.bin`, generate                     |
| `alm-compile` | the compiler: the graph, the weights, the programs       |
| `alm-schedule` | the scheduler: the MILP, and `plan.yaml` out of HiGHS    |

Nothing here runs on a tensor framework, and the reason is the shape of the
computation rather than a preference: generation is one token at a time with no
batch, so the whole of the linear algebra is four matrix-vector products a
layer against a residual stream 38 wide, and the attention is a convex-hull
query rather than an inner product at all.  A `[1, 38] x [38, 114]` product is
a hundred nanoseconds of arithmetic; a tensor type's dispatch, allocation and
round trip cost several times that, and it was measured at 3.5x.  The element
type is `f64` everywhere, which is not a detail either: every exactness claim
in the construction is a claim about float64.

The only dependency left outside the standard library is HiGHS, under
`alm-schedule`, and it is not on any path a run takes.

## What is already different from the original

* **Breakpoints are exact.**  The C++ hull stores every breakpoint as a rounded
  `long double` and compares the rounded values.  Rust has no 80-bit float, so
  rather than dropping to 53 bits the breakpoints here are kept as rationals and
  compared by cross-multiplication, exactly (`alm-hull/src/exact.rs`).  This is
  `ALM.FloatHull.cmp_of_sep` with `delta_1 = delta_2 = 0`, and it is strictly
  sharper than the original: `alm-hull` separates breakpoints one ulp of a
  `2^106` product apart, which `long double` cannot.

## What building it has already turned up

* **The hull head is exact only if no three key points are collinear.**  The
  envelope keeps vertices; a key that maximises the score at exactly one query
  and is interior everywhere else is erased on insertion, and its payload goes
  with it — so both tie-breaks can answer with the wrong entry.  The parabola
  `k -> (2k, -k^2)` is strictly convex and has no three collinear points, which
  is why the machine gets away with it, but neither post nor code states the
  hypothesis.  Witness and characterization: `alm-hull/tests/differential.rs`.

* **The float64 wall is a wall of the representation.**  Scoring exactly —
  `dot_cmp` in `alm-hull/src/exact.rs`, Dekker products and a Shewchuk
  expansion sign — does not move it.  The first query whose own key stops
  being the strict argmax is `q = 94 906 266` under rounded scoring and
  `q = 94 906 266` under exact scoring, the same number to the unit, because
  the coordinate `-k^2` is rounded before the comparison ever sees it.  No
  wider accumulator helps; only a wider embedding would, and that is a change
  to the weights.  `todo3.md` section 4.

* **The shipped model's query scale moves that wall 22 % earlier.**  Hard
  attention queries at `sqrt(2) * 10^10`, which multiplies the score and the
  gap to the runner-up together but does not cancel, because `ulp` is relative
  and the gap is not.  At unit scale the first unseparable query is
  `94 906 266`; at the model's scale it is `73 966 031`.  So the right guard is
  not `score > 2^53` but `ulp(score) > abs(qy)` — scale-free, and quiet about
  the keys the compiler deliberately disables with `BIG = 1e30`.  Implemented
  in `alm-hull/src/grid.rs` and reported per run by `alm-vm`.

* **A head keyed on a 32-bit value is past the wall from the first token.**
  The wall bounds whatever the head is keyed on, not the trace length, and this
  machine keys heads on WebAssembly values, which reach `4.29e9`.  The first
  crossing in `hello` is the key `[673720322, -1.1347476806894592e17]` — the
  parabolic embedding of `336 860 161` — where the float64 maximum over the
  neighbourhood is attained at `v-3`, `v-1` and `v+1` and not at `v`.  The
  reference programs pass because their values are never within ~20 of each
  other in one head; nothing arranges that.  `todo3.md` section 4b.

* **38 is the optimum, and the released schedule is one of them.**  The
  scheduler is a mixed-integer program — 14 392 variables over 34 498 rows for
  the shipped graph's 159 gates and 186 dimensions — and porting it makes the
  claim checkable in two directions at once.  The released schedule satisfies
  every row and costs exactly the `milp_d_model: 38` it reports; and HiGHS
  proves in 9.6 s that no seven-layer schedule of this graph costs less.  What
  it finds is a *different* schedule of the same cost, whose deepest dependence
  width is 30 against the release's 31, which builds a working `model.bin`.
  The port's own arithmetic agrees with the solver's to the unit, and that is
  the check that caught the one real bug in it: an indicator whose upper row
  had its big-M on the wrong side, which let a solver claim a residual stream
  of 26 for a schedule needing 40.  The test is
  `no_bit_of_the_program_is_free_to_flip`.

* **Removing the query scale is free, and on these programs it buys nothing.**
  `--grid` divides each query by `abs(qy)` before scoring, which an argmax
  cannot notice, and the released weights agree: every reference trace comes
  back token for token, `hello` through `sudoku`.  It also removes no
  off-the-grid query — the counts are identical in both modes — because the
  worst `ulp(score)/margin` is 512 at unit scale against 311 as shipped in
  `hello`, and 1024 against 1244 in `addition`.  The scale can only multiply
  that ratio by `2^33/s = 0.607` or `2^34/s = 1.215`, so it shuffles rather
  than loses; what it shifts one way is the threshold.  `todo3.md` section 0a.

* **Four fifths of the non-integer queries are the query scale's own
  rounding.**  The exactness argument assumes an integer query; the machine
  recovers a cumulative sum by multiplying an average back by its count, which
  does not round-trip.  Measured on the released weights that costs 22 % to
  32 % of all queries — but on those weights the reconstruction divides by the
  scale the weight rows were rounded *after* being multiplied by, so it counts
  two roundings the machine never makes.  On weights rebuilt without the scale
  the share is 2.8 % to 7.1 %, and the worst offset reaches **two** ulp
  (`collatz`, at `2573.000000000001`) rather than the one ulp the shipped
  model reports.  Two ulp is the row of `todo3.md` section 8's margin table
  that halves the wall, to `50 331 647`.  `todo3.md` section 8a.

## Patches to the vendored compiler

`transformer-vm/` is gitignored, so a correction that belongs to the Python
side cannot live there.  `patches/` holds them as unified diffs against the
release, each with a header saying which `todo3.md` section it implements and
what rebuilding with it measured.  Apply one with

```
git apply --directory=transformer-vm vm-rs/patches/on-the-grid.patch
cd transformer-vm && uv run python -m transformer_vm.build --plan plan.yaml --save-weights=model.bin
```

Rebuilding against the recorded plan takes about 1.5 s; without `--plan` the
MILP scheduler runs again and takes minutes.

* **The compiler does not build the same model twice.**  Ten builds of
  `model.bin` from the released plan give four distinct files.  The erased slots
  of a half-layer are held in a `set` of ints, the loop that zeroes them
  iterates it directly, and that fixes which source slot is packed into which
  passthrough head — but a CPython set of ints iterates in table order, and the
  table order depends on the insertion order, which comes from a set of
  `Dimension` objects hashed by address.  The four files differ only in the
  order of layer 5's FFN passthrough neurons, so they compute the same function;
  what they cost is the ability to check a build against its source, and to
  check this port against the original.  Two `sorted()` calls fix it:
  `patches/reproducible-build.patch`.  `todo3.md` section 9.

* **The key perturbation is load-bearing, and it is a noise margin.**  The
  post says the latest write wins because of a small position-dependent term
  added to each key; the release also carries a sequence number in the cache,
  which looks like the same thing done properly.  Rebuild the weights with
  that term set to zero and all four cheap programs stop, under both caches and
  at different tokens.  The reason is that two writes to one logical key do not
  reach the head with the same key — the key is a matvec and the matvec rounds
  — so the sequence number never sees a tie to break.  Measured by
  `alm-hull/src/gap.rs` and reported per run under `--brute`: with the term,
  the closest runner-up in `hello` is `6.0e-6` key steps and nothing is inside
  `1e-9`; without it, 1 402 queries are.  Any coefficient from `1e-6` to `0.49`
  works, which is what a margin looks like.  `todo3.md` section 2a.

## Running it

`plan.yaml`, `model.bin` and the program traces are build artefacts, not files
in this repository — and none of them needs a Python to make any more.
`alm-compile` is a port of `graph/core.py`, `wasm/interpreter.py`,
`model/weights.py`, `scheduler/milp.py` and the whole of `compilation/`, so the
five binaries below are the entire build:

```
cargo build --release
target/release/alm-schedule -o plan.yaml
target/release/alm-build plan.yaml model.bin
target/release/alm-cc  --all --out data
target/release/alm-ref --all --data data
target/release/alm-vm model.bin data/hello.txt data/addition.txt
```

Nothing there reads the vendored release: run it with `transformer-vm/` moved
out of the way and the same `plan.yaml`, the same 1 188 074-byte `model.bin`
and the same eighteen `data/` files come out.  What makes that possible is
`programs/` — the six `.c` sources, the manifest and `runtime.h`, copied from
the release under its own Apache-2.0 licence, because the programs are the
machine's input and were never part of the compiler this port replaces.

`alm-schedule` builds the mixed-integer program and gives it to HiGHS, which
`highs-sys` compiles from source, so there is no solver to install; `--lp`
writes the program in CPLEX LP format instead, for a second opinion.  Handed
the release's own `plan.yaml` instead, `alm-build` writes the same 1 188 074
bytes as the Python builder; `--grid` builds the `on-the-grid.patch` variant
and `--mask` the masking one, so both `todo3.md` experiments can be run
without a Python at all.  `alm-cc` runs clang with the release's own flags and
writes the token prefix; `alm-ref` executes it and writes the trace the model
is checked against.

The checking does not want the vendored checkout either.  `reference/` holds
the two things needed for it: the released `plan.yaml`, which is an input and
the one file of the release that is not a build product, and `sha256sums`,
which is what the released `model.bin` and the eighteen `data/*.txt` hash to.
Every test builds the artefact it is about and compares the digest — byte
identity in sixty-four characters instead of ten megabytes — so `cargo test`
on a bare clone runs everything.  The one thing it can still skip for is a
clang that cannot target wasm32.

`alm-vm`'s command line is the C++ driver's: `--brute`, `--trace[=N]`,
`--args=STR`, `--max=N`, plus `--grid`, which has no counterpart there.

The check that keeps the port honest is byte identity at every layer:
`alm-graph-dump` against `alm-compile/tests/pydump.py` is 1 938 lines of
dimensions, expressions and float64 bit patterns with no difference;
`alm-wasm-dump --lower` against `tests/lowerdump.py` is 12 298 lines of decoded
and lowered instructions with no difference; the weights are `model.bin`
itself; the eighteen released `data/*.txt`, `*_spec.txt` and `*_ref.txt` are
reproduced byte for byte from the C sources in `tests/programs.rs`, digest
against digest; and the
released `plan.yaml` is written back out line for line from the ported
schedule's own analysis, down to pyyaml's line folding.  The one field that
cannot be reproduced is each layer's `attention:` order, which the Python
takes from a `set` of objects hashed by address, so it is compared as a set.

## Status

All four cheap reference programs reproduce their traces token for token,
under both caches — and now from artefacts this port built itself, weights and
programs and reference traces alike:

```
hello      1 034 tok, 149 ops    Hello World!
addition   4 362 tok, 718 ops    19134
collatz   44 589 tok, 9 009 ops  7 22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1
fibonacci  9 104 tok, 892 ops    55
```

Against the C++ engine on the same 59 089 tokens, back to back on one machine:

```
            total     proj     hull     head     misc
C++         4.05s    2.491    1.316    0.217    0.023
alm-vm      3.88s    2.370    1.207    0.230    0.068
```

The port is the faster of the two, and it is faster in the part that was
supposed to cost it: the hull answers in 1.21s against 1.32s while comparing
breakpoints *exactly*, where the original compares rounded `long double`s.
What is left on the other side of the ledger is `misc`, which is the
per-query diagnostics this port carries and the original has no counterpart
for.

The lead holds as the traces grow.  On the sudoku trace, eighteen times the
tokens:

```
            total     proj     hull     head     misc
C++        79.08s   44.455   29.637    4.568    0.417
alm-vm     75.65s   42.397   27.263    4.793    1.196
```

Both engines keep the envelope in the same shape of container, and it took
two goes to get there.  The C++ uses one `std::multiset`, ordered by slope
and searched by breakpoint through a heterogeneous comparator; the port began
with a vector in slope order, which searches and walks faster but shifts a
tail on every insertion.  The vector was the quicker of the two on every
trace here — 26.1s against 27.1s on this one — and it was the wrong
container.  `HullBuild.lean` charges a build two searches per key and one
unit per erase and proves it `O(n log n)` from that tariff, which is only
true where erasing at a known cursor is amortized constant.  Erasing from a
vector costs the tail it moves, so `buildCost_le` was a theorem about a
container the engine did not have.

`alm-hull/src/tree.rs` is the one it describes: a red-black tree in an arena
addressed by `u32`, with the cursors `BTreeSet` does not offer on stable Rust
(`btree_cursors`, rust#107540).  A run of erases walks the envelope once
instead of searching again for each line it drops, nothing ever moves, and
the ends are held rather than walked to, so an insertion at either end costs
no descent at all — which leaves one search per key where the C++ spends two.
What it buys is that the numbers stop depending on the order the keys arrive
in: `alm-stress` builds 262 144 keys in descending order in 0.072s, where the
vector took 102.9s and grew quadratically.  Nothing in the engine promises
the near-sorted order the traces happen to have, since `cache.rs` reads the
keys from a learned projection.

So the vector is gone, and with it the `BTreeMap` envelope tried before it,
which answered the sudoku trace in 81.8s — worse than either.  What the tree
is checked against is the definition rather than another container:
`alm-hull/src/envelope.rs` takes the maximum over the given lines by brute
force at every probe and asserts the envelope agrees with it under every
arrival order, and `alm-hull/src/tree.rs` audits the red-black invariants,
the cached ends and the cursor walk against a sorted vector of what was put
in.

The projections are not merely as fast as the C++ — they are the same
arithmetic.  `transformer.cpp` sums each row left to right into one accumulator
(and only calls BLAS on macOS); `Dense::apply` writes that loop out, so on the
whole of `hello` and `addition` — 37 756 layer steps, 1 434 728 residual-stream
values — every `f64` matches the C++ bit for bit.  A tensor framework did not:
its `matmul` splits the reduction, and the two disagreed by an ulp often enough
to move a quarter of the hull queries off the integers.
