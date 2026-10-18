/-
# Phase model over a finite abelian group

The phase model of division (`Transformer.PhaseModel`) with a finite abelian
group `G` of order `n` in place of the cyclic group `ℤ/n`: phases
`ψ ∈ ℝ^G`, the rows `e_x - e_y - e_{x-y}` of the division table of `G`, and
the characters `χ : G → ℂ`, whose phases `arg χ(x)` take the place of
`2πjx/n`.  The zeros of `V_full` are the phases of the characters up to
periods.  For `n ≥ 4`, below `n²/8` every local minimizer of `V_full` is the
phases of a character up to periods and every other critical point is a
strict saddle, so for almost every start below `n²/8` gradient descent
converges to the phases of a character.

| Module | Contents |
| --- | --- |
| `PhaseModel.Abelian.Defs` | integer rows, the rows of division, the energy `V_S`, the phases of a character, the periods `2πℤ^G` |
| `PhaseModel.Abelian.Basic` | rows and periods; only `0` is orthogonal to every row; the characters keep every relation and are, up to periods, the only zeros of `V_full`; the norms of the rows |
| `PhaseModel.Abelian.Coordinates` | the gradient of `V_full` and the diagonal of its Hessian in coordinates; the chord inequality summed over `G` |
| `PhaseModel.Abelian.Basin` | critical points of `V_full` within `n/8` of `0` with a nonnegative Hessian diagonal are periods |
| `PhaseModel.Abelian.Characters` | the same near every character; there the other critical points are strict saddles |
| `PhaseModel.Abelian.Fourier` | the orthogonality of the characters, Parseval's identity and a cubic identity: some character lies within `V_full/n` of every phases |
| `PhaseModel.Abelian.Landscape` | below `n²/8`: local minimizers of `V_full` are characters, other critical points strict saddles |
| `PhaseModel.Abelian.Convergence` | the periods span `ℝ^G`; every orbit of gradient descent on a phase energy with integer rows converges, to a critical point; from almost every start below `n²/8`, gradient descent on `V_full` converges, to a character up to periods |
-/

import Transformer.PhaseModel.Abelian.Defs
import Transformer.PhaseModel.Abelian.Basic
import Transformer.PhaseModel.Abelian.Coordinates
import Transformer.PhaseModel.Abelian.Basin
import Transformer.PhaseModel.Abelian.Characters
import Transformer.PhaseModel.Abelian.Fourier
import Transformer.PhaseModel.Abelian.Landscape
import Transformer.PhaseModel.Abelian.Convergence
