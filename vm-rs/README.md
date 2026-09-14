# vm-rs — a Rust port of `transformer-vm`

A port of the Percepta `transformer-vm` release (Apache-2.0, vendored under the
gitignored `transformer-vm/`) whose purpose is to carry the corrections the Lean
formalization licenses.  `todo3.md` is the fix list; this is where the fixes go.

## Crates

| crate       | what it is                                              | burn |
|-------------|---------------------------------------------------------|------|
| `alm-hull`  | the 2D hard-attention KV cache (`attention/hull2d_cht.h`) | no |
| `alm-model` | the transformer itself (`model/transformer.py`, `.cpp`)   | yes |
| `alm-vm`    | the driver: load `model.bin`, generate                    | —  |

`burn` is pinned at 0.21, on the `ndarray` backend with `f64` elements.  The
element type is not a detail: every exactness claim in the construction is a
claim about float64, and `NdArray<f64, i64, i8>` is the only backend in burn
that holds one.  burn covers the model layer only — the forward pass is a chain
of 36-wide matvecs with no batch — while the hull is ordinary Rust.

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

## Running it

`model.bin` and the program traces are build artefacts of the original Python
and are not in this repository:

```
cd transformer-vm
uv sync --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv run python -m transformer_vm.build --save-weights=model.bin
uv run python -c 'from transformer_vm.compilation.compile_wasm import ensure_data; ensure_data()'
```

Then

```
cargo run --release -p alm-vm -- ../transformer-vm/model.bin     ../transformer-vm/transformer_vm/data/hello.txt
```

The command line is the C++ driver's: `--brute`, `--trace[=N]`, `--args=STR`,
`--max=N`.  `model.bin` itself is out of scope here — it is the output of the
compiler (`graph/`, `scheduler/milp.py`, `compilation/`, `model/weights.py`),
not of the model, and the Lean conclusions this port exists to carry are all
about the runtime.

## Status

All four cheap reference programs reproduce their traces token for token,
under both caches:

```
hello      1 034 tok, 149 ops    Hello World!
addition   4 362 tok, 718 ops    19134
collatz   44 589 tok, 9 009 ops  7 22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1
fibonacci  9 104 tok, 892 ops    55
```

Against the C++ engine on the same 59 089 tokens:

```
            total     proj     hull     head
C++         3.31s    2.012    1.107    0.176
alm-vm     11.40s    6.758    4.362    0.190
```

The head matches once it is sparse, as the original's is.  What is left is
3.4x on the projections — burn's per-call overhead on a 38-wide residual
stream generated one token at a time, with no batch to amortize it — and 3.9x
on the hull, which is what the exact rational breakpoints cost against the
original's rounded `long double`.  Neither changes an answer.
