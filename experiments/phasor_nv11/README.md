# A four-layer Phasor for NV11

Can the Phasor that failed to generalize in E1 solve `a³ + ab mod 97` with different weights?
On the original half split, fitting one class shift per first operand gives all 4,705 held-out
answers. The saved weights run through DiffGPT's existing `complex-phase-score-lerope` forward
pass with its tied embedding, width 128, four heads, 171 complex FFN units and four layers.
The input is the original two-token sequence `[a, b]`. The prediction is the largest of the
97 ordinary logits at the second token. [NV11 and E1](../../review/flat_spectrum.md#nv11-what-no-single-feature-carries)

## What is fitted

| Fit | Supplied structure | Estimated from the 4,704 training triples |
| --- | --- | --- |
| `fit` | A constructed multiplication Phasor and the CM121 family of row shifts | One of 97 shifts for each first operand, by categorical maximum likelihood |
| `fit-class` | The larger class `g(a) + h(b) + m a b` | Both sets of shifts and the coupling, using the existing PT6 finite-field solve |
| `score --ablate-shifts` | The saved model from either fit | No fitting. Set both shift embeddings to the unit phase. |

For the first fit, let `L(a,b,c)` be the multiplication Phasor's actual logit for class `c`.
For each observed `a` choose the shift `s` that maximizes

```text
sum over its training rows (a,b,y) of L(a,b,(y-s) mod 97).
```

This minimizes cross-entropy over cyclic translations of those logits since all shifts have
the same softmax denominator. The fit sees the cubic training labels and the backbone logits.
Multiplication labels are used only for scoring the control. The estimate is written into
the first operand's embedding. The network then multiplies phases internally, so its tied
logits, including the own-token residual, are evaluated afresh after fitting. The second
fit learns the coupling as well and can handle simultaneous row and column shifts.
[CM121 symmetry](../../review/convex/moments.md#cm121-a-class-shift-in-every-cell-is-a-symmetry-of-the-budget-sets) and [PT6 completion](../table_part/README.md)

## How the weights compute a logit

Write `ψ(x) = exp(2πix/97)`. In the tied embedding, each residue has a plane for its answer
code, planes for its fitted row and column shifts, and seven powers `ψ(2ʲ m a)` when read
as the first operand. Its seven binary digits are stored as signs when read as the second.
At position 1, one rotary attention head and one uniform head have different weights on the
two positions. The output projection takes their linear combination to copy the first
operand's planes. All four mHC streams carry the same state.

The first FFN writes eight complex numbers. For each bit of `b` it writes the corresponding
row power if that bit is one, or the unit phase otherwise. The eighth number carries
`ψ(g(a)) ψ(h(b))`. Three more FFNs multiply the eight leaves in pairs. Up to positive RMS
scales the result has phase `ψ(g(a) + h(b) + m a b)` in the tied answer plane. A common
embedding bias controls interference from the own-token residual. Only the saved model
weights and the two token IDs enter the forward pass.

## Measured result

The dataset builder checks the original SHA-256 fingerprints of both splits. Fitting took
only its training object. Both methods produced a byte-identical block of 898,448 FP32
parameters. The native scalar FP64 reference scored every cell from the reloaded block.
A correct answer requires a strictly positive margin over every other logit.

| Weights | Training NV11 | Held-out NV11 | Held-out multiplication | Minimum held-out NV11 margin |
| --- | ---: | ---: | ---: | ---: |
| Fitted by either method | 4,704 / 4,704 | 4,705 / 4,705 | 51 / 4,705 | 0.255467 |
| Both shifts set to the unit phase | 46 / 4,704 | 51 / 4,705 | 4,705 / 4,705 | −404.815925 |

The two labels coincide exactly on row `a = 0`, which contains 51 held-out pairs. The positive
margin establishes that four layers at these widths can represent NV11. Learning uses a
supplied modular multiplication circuit or a specified finite-field class. Generalization
from random initialization with the original AdamW recipe remains open. The measured
precision is the upstream CPU reference. GPU mixed-precision execution is unverified.

## Reproduce

The build reads the unmodified forward, layout and RNG source from DiffGPT Git object
`5a37c515639f23037d191b0659a5a0853984328d`. It expects a clone at `.lake/reproduction/gpt`
under the repository root, or an existing clone named by `PHASOR_GPT_REPO`. For a fresh checkout
the following commands reproduce the experiment.

```sh
mkdir -p .lake/reproduction
git clone https://github.com/fxeqxmulfx/diff-gpt.git .lake/reproduction/gpt
cd experiments/phasor_nv11
mkdir -p runs
cargo test --locked --release
cargo build --locked --release
export RAYON_NUM_THREADS=12
target/release/phasor-nv11 fit runs/learned.f32
target/release/phasor-nv11 fit-class runs/class.f32
cmp runs/learned.f32 runs/class.f32
target/release/phasor-nv11 score runs/learned.f32
target/release/phasor-nv11 score runs/learned.f32 --ablate-shifts
target/release/phasor-nv11 predict runs/learned.f32 3 5
```

The last command prints `42`. Each weights file is the little-endian FP32 parameters block
in the upstream native layout. Its SHA-256 on the reported run was
`7ba9272ae9d19a3adab342716ed7be07016ec250ab29b1dc2e09772731a27c4d`.
