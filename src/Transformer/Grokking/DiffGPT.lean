module

public import Transformer.Grokking.DiffGPT.StepWrap
public import Transformer.Grokking.DiffGPT.PolarDiagonal
public import Transformer.Grokking.DiffGPT.Compare
public import Transformer.Grokking.DiffGPT.ProjectionKink
public import Transformer.Grokking.DiffGPT.MergedLevels
public import Transformer.Grokking.DiffGPT.CompareFill
public import Transformer.Grokking.DiffGPT.WrapFill
public import Transformer.Grokking.DiffGPT.StepHalfPlane
public import Transformer.Grokking.DiffGPT.UnseenRow
public import Transformer.Grokking.DiffGPT.StepChord
public import Transformer.Grokking.DiffGPT.StepPhase
public import Transformer.Grokking.DiffGPT.StepSign
public import Transformer.Grokking.DiffGPT.StepPole
public import Transformer.Grokking.DiffGPT.ScoreShift
public import Transformer.Grokking.DiffGPT.ScoreChain
public import Transformer.Grokking.DiffGPT.StepNumbers
public import Transformer.Grokking.DiffGPT.ScoreNumbers
public import Transformer.Grokking.DiffGPT.StepLevel
public import Transformer.Grokking.DiffGPT.StepCarry
public import Transformer.Grokking.DiffGPT.StepDomain

/-! `review/diffgpt/steps.md`: the DiffGPT task *Fibonacci mod 97*, a recurrence modulo `p` that
the model reads and writes as integer steps. M and Theorem A are those of the flat-spectrum
note, `Transformer.Grokking.FlatSpectrum`. The step of an additive
recurrence modulo `p` in a difference encoding (`StepWrap`, `PolarDiagonal`, `Compare`,
`ProjectionKink`, `MergedLevels`, `CompareFill`, `WrapFill`, `StepHalfPlane`, `UnseenRow`): the
step `(b + x) % p − x` is the residue `b`, a kernel of rank one in the two steps before it,
and a wrap. The sign of the step is the comparison kernel `C` (`1` for `i ≤ j`, `−1` below)
with reversed columns, and `C + Cᴴ = 2`. A kernel `K` with `K + Kᴴ = 2c`, `c > 0`, has a polar
factor with a positive diagonal, so by Theorem A it is not a minimum of M once a diagonal cell
is unseen; for `C` the same holds for the cells just below the diagonal. A matrix with two
equal rows `i`, `i'` and two equal columns `j`, `j'` is a minimum of M among the matrices equal
to it off `(i, j)`, by a kink of the nuclear norm between two projections. So with one unseen
cell at the boundary of `C`, or one unseen pair with `b + x = p − 1` or `b + x = p`, the table
with that cell flipped is a minimum of M and no minimum holds the true sign. In the steps
`d₁`, `d₂` the sum is `b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0]`, so
`exp (iπ (b + x) / p) = exp (2πi d₁ / p) · exp (3πi d₂ / p)` has rank one and the wrap is the
half-plane `Im (exp (iπ / 2p) · …) < 0` of it. A row or a column with no seen cell is zero in
every minimum of M, as `‖P X‖_S1² + ‖(1 − P) X‖_S1² ≤ ‖X‖_S1²` for a projection `P` of the
rows. The phase of order `2p` of the step, which tells `b` from `b − p`, needs no threshold
(`StepChord`, `StepPhase`): as `exp (iα) + exp (iβ) = 2 cos ((α − β) / 2) · exp (i (α + β) / 2)`,
the chord `S = exp (iπ (b + x) / p) + exp (−iπ / 2p)` is `2 cos φ · exp (i (φ − π / 2p))` with
`φ = π (b + x + ½) / 2p`, and `cos φ` is positive exactly without a wrap. So
`exp (iπ d₃ / p) = exp (iπ / 4p) · exp (−iπ d₂ / 2p) · S / ‖S‖`, and `S` is, up to a factor of
modulus one, the sum of one phase of each of the two steps read. The sign of the step is the
real kernel of rank two `ℓ = Im (exp (iπ / 2p) · …)`, which is never zero, divided by `|ℓ|`
(`StepSign`). Moving the base point of the chord by `2γ`, `|γ| < π / 4p`, turns every
normalized chord by `exp (iγ)` and changes no branch (`StepPole`): one angle places the wrap of
all cells. With a free score for each token in place of the phases (`ScoreShift`,
`ScoreChain`) the wrap is the comparison `p − 3 d₂ − 2p [d₂ < 0] ≤ 2 d₁`, and scores fit the
seen cells when they compare there as these do. Raising the scores of rows and columns closed
upwards along the seen cells keeps every seen comparison, so some fitting scores give the
other answer at an unseen cell exactly when no chain of seen cells links its two tokens; at a
cell with equal scores, and at a cell with integer scores one apart, a chain is the cell
itself. With the numbers of the tokens given (`StepNumbers`, `ScoreNumbers`): at every frequency
`ω` the phase `exp (iω (b + x))` is `exp (2iω d₁) · exp (3iω d₂)` times one of two constants by
the sign of `d₂`, and for `0 < ω`, `ω p ≤ π` the wrap is `0 < sin (ω (b + x + ½ − p))`; scores
that are linear in vectors of numbers of the rows and columns and fit the seen cells answer as
those cells do at every cell in the convex hull of the seen cells of one answer. With levels
in place of steps (`StepLevel`) the value to write is the entry of the kernel of `a + b` and is
determined by it, while two steps share a residue; a step `d` is read at `p − |d|` of the `p²`
states and a level at `p`. A codec that carries its rounding to the next row (`StepCarry`)
writes a step with one of two tokens, by a bit of the steps before it, and with one token on a
grid that holds the values. The step is the only integer of its residue that keeps the series
in its domain `[0, p)` (`StepDomain`): the wrap is a function of the residue and of the level,
which the decoder holds. -/

@[expose] public section
