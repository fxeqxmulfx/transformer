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

## Status

`alm-hull` is complete and differentially tested against a brute-force head.
`alm-model` reads `model.bin` and runs the forward pass on burn; `alm-vm` is
a stub.
