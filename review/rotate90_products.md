# Exact products and a constructive division checkpoint

M9.4b separates a proved algebraic ingredient from a numerically checked
complete model checkpoint. Neither is a gradient-attraction theorem. The
architecture remains the tied Rotate90 SwiGLU model ported in `c59d4426`.
The expression is checked against its actual forward function.

## Proved compiler

`Neural.Products.silu_sub_neg` proves `SiLU(t) - SiLU(-t) = t` over the
reals. Thus subtracting two SwiGLU channels with opposite gates and the
same value yields their product exactly. `productWeights` places the
opposite gate and output matrices, the common value matrix and the shared
embedding into their original free Euclidean coordinates.

`pairedBranch_eq_products` proves a sum of arbitrary products of linear
forms. `gatedState_productWeights` retains the original skip and its first
RMSNorm, and `gatedLogit_productWeights` retains the second RMSNorm and the
same transposed E at the readout. Nothing assumes an unproved model
reduction or changes the classification loss. The theorem describes a
family of weights; it does not say that the optimizer preserves the paired
gate constraints or reaches them. Two individual hidden channels are used
per product, not two pairs per product.

## Checked character construction

The script `experiments/rotate90_analytic/expressivity.py` chooses primitive
root 5 modulo 97 and the faithful character
`chi(5^k) = (cos(2 pi k/96), sin(2 pi k/96))`. In zero-based coordinates,
E has a constant `c=0.1` at coordinate 0. Nonzero rows have the character
at coordinates 2 and 4; row zero instead has a one at coordinate 6. Every
other coordinate is zero. All rows have squared norm `1+c^2`.

J sends those even coordinates into disjoint odd ones, so the input state
has squared norm `2(1+c^2)`. Write its normalized value as r and
`kappa = eps + 2(1+c^2)/128`, so `r0 = c/sqrt(kappa)`.
The branch uses the following products and output coefficients:

| Product | Output coordinate | Coefficient |
| --- | ---: | ---: |
| r0 r0 | 1 | -kappa/c |
| r0 r2 | 2 | -kappa/c |
| r0 r4 | 4 | -kappa/c |
| r0 r3 | 3 | -kappa/c |
| r0 r5 | 5 | -kappa/c |
| r2 r3 | 2 | kappa |
| r4 r5 | 2 | kappa |
| r4 r3 | 4 | kappa |
| r2 r5 | 4 | -kappa |

The first five cancel the unwanted skip coordinates. The last four form
the real and imaginary parts of `chi(a) conjugate(chi(b))`. The constant
and zero flag already have their intended skip values. For the original
domain `b != 0`, the intended state is therefore `E[a/b]`, with the zero
numerator pathway retained. Nine opposed pairs need 18 individual gates;
the remaining 25 of H=43 stay present with zero weights.

Final RMSNorm multiplies this state by the same positive factor for every
answer. Its tied readout uniquely favors the intended character, or the
zero flag when the numerator is zero. The expected minimum margin is
`(1-cos(2 pi/96))/sqrt(eps+(1+c^2)/128)`. The complete construction and this
selection argument are not yet fully formalized in Lean; the five compiler
theorems above are. The executable checks cover every equation independently
through the original model's forward path, in addition to comparing states
and logits with the intended formula.

## Numerical evidence and limits

All 9,312 source equations, including 96 zero numerators, classify correctly
on CPU float64, CPU float32 and CUDA float32. The maximum absolute parameter
is one; the largest output coefficient is about 0.157913. Thus the uniform
1.7 AdamW parameter box proved in `Neural.AdamW` does not exclude this
correct classifier. This does not identify an optimizer trajectory to it.

Float64's maximum state and logit errors are 1.89e-15 and 2.31e-14. CPU
float32 gives 4.77e-7 and 4.29e-6; CUDA float32 gives 5.96e-7 and 4.77e-6.
Minimum observed margins are positive, about 0.02409. CE is about 2.42165;
correct classification does not assert zero finite-logit cross-entropy or
that the checkpoint minimizes the actual regularized training objective.

The JSON artifact records dtype, device, runtime, source hashes, parameter
hashes, all examples and zero-row counts. The witness is constructed using
the known arithmetic rule, not learned from half the equations, and is
never supplied as training initialization. Its single-character solution
is distinct from the multi-character geometry of the recorded trained
checkpoints. The learned character selection and random-start attraction
are still open parts of the original goal.

## Validation at the user-requested stop

All five new compiler theorems build without warnings and have only
`propext`, `Classical.choice` and `Quot.sound` as axioms. The complete build
passes 12,777 jobs. The audit remains 39 direct `sorry`, zero indirect rests
and zero extra axioms. The regenerated index has 3,680 modules and 18,906
theorems, zero vacuous statements and zero placeholders. The forbidden and
whitespace checks pass, and both analysis scripts compile.

The earlier complete Python suite passed 339 tests in 804.625 seconds,
with three archived GPU comparisons skipped. The final `./run.py verify`
completed all Lean stages, then its repeated Python phase was interrupted
at the user's explicit stop. That final combined command is unfinished,
not a successful verification result. The standalone constructive witness
completed all three dtype/device checks before the stop.
