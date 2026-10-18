/-
# Phase model

Gradient descent on phases `ψ ∈ ℝⁿ`, one per token, fitted to integer
relations `⟪m_r, ψ⟫ ∈ 2πℤ` through the energy
`V(ψ) = Σ_r (1 - cos ⟪m_r, ψ⟫)`.  For division in a cyclic group of order
`n`, with phases indexed by discrete logarithms, a set `S` of pairs gives
`V_S(ψ) = Σ_{(x, y) ∈ S} (1 - cos (ψ_x - ψ_y - ψ_{x-y}))`, whose zeros are
the phases with `ψ_x - ψ_y - ψ_{x-y} ∈ 2πℤ` for every `(x, y) ∈ S`.  On the
full table, and on every `S` whose rows generate those of the full table
over `ℤ`, these are the phases `ψ_x = 2πjx/n` of the characters up to
periods, and every zero is nondegenerate.  For `n ≥ 4`, below `n²/8` every
local minimizer of `V_full` is the phases of a character up to periods and
every other critical point is a strict saddle.  The phase energy is
analytic, so every orbit of gradient descent converges, and for almost every
start below `n²/8` it converges to the phases of a character.  The same
landscape and convergence hold over every finite abelian group in place of
`ℤ/n` (`PhaseModel.Abelian`), and the landscape of `ℤ/n` is that of `Fin n`,
whose characters are the `x ↦ e^{2πijx/n}`.  The landscape fails for
partial tables: already for `n = 5` some `S` generating the full table has a
local minimizer below `n²/8` that is not a character (`PhaseModel.Spurious`).

| Module | Contents |
| --- | --- |
| `PhaseModel.Defs` | the phase energy, integer rows, the rows of division, the periods `2πℤⁿ`, the cube `[0, 2π]ⁿ` |
| `PhaseModel.Basic` | zeros and invariance; analyticity; the gradient, the Hessian and its Lipschitz constant; nondegenerate zeros; `e^{i(x + 2πk)} = e^{ix}`, `conj e^{iθ} = e^{-iθ}` and the chord inequality |
| `PhaseModel.Periods` | integer rows and periods; translates into the cube; the energy of division |
| `PhaseModel.Descent` | limit points are critical; every orbit converges, to a critical point; convergence from nondegenerate limit points; almost every orbit converges to a local minimizer when every critical point is nondegenerate |
| `PhaseModel.Division` | the full division table: only `0` is orthogonal to every row; the characters are zeros of every `V_S` and, up to periods, the only zeros of `V_full`; the norms of the rows |
| `PhaseModel.Generating` | sets of rows generating the full table over `ℤ`: the zeros of `V_S` are the characters up to periods, all nondegenerate, with the Łojasiewicz inequality of exponent `1/2`; convergence from a character as a limit point |
| `PhaseModel.Lift` | critical points near a zero: relations within `π` of `2πℤ` vanish at a critical point, and within `π` inclusive at a local minimizer |
| `PhaseModel.Coordinates` | the gradient of `V_full` and the diagonal of its Hessian in coordinates; the chord inequality summed over `Fin n` |
| `PhaseModel.Basin` | critical points of `V_full` within `n/8` of `0` with a nonnegative Hessian diagonal are periods |
| `PhaseModel.Characters` | the same near every character; there the other critical points are strict saddles |
| `PhaseModel.Orthogonality` | the characters `x ↦ e^{2πijx/n}` of `Fin n`, all of them, with the phases `2πjx/n` up to periods; their orthogonality |
| `PhaseModel.Fourier` | Parseval's identity and a cubic identity: some character lies within `V_full/n` of every phases |
| `PhaseModel.Landscape` | below `n²/8`: local minimizers of `V_full` are characters, other critical points strict saddles |
| `PhaseModel.Convergence` | from almost every start below `n²/8`, gradient descent on `V_full` converges, to a character up to periods |
| `PhaseModel.Abelian` | the same over a finite abelian group: its characters, their orthogonality, the landscape of `V_full` below `n²/8` for a group of order `n`, and the convergence to a character from below `n²/8` |
| `PhaseModel.Twisted` | critical points with positive cosines are local minimizers; twisted states, phases in `(π/3)ℤⁿ` with defects `0, ±1` modulo `6` and balanced rows, are local minimizers of generating partial tables |
| `PhaseModel.Spurious` | for `n = 5`, a partial table generating the full one with a local minimizer of energy `3 < 5²/8` that is not a character |

The theorems of `Coordinates`, `Basin`, `Characters`, `Orthogonality`,
`Fourier` and `Landscape` are those of `PhaseModel.Abelian` at `G = Fin n`,
which imports none of them.  The convergence results instantiate
`Transformer.Landscape.SaddleAvoidance` with the lattice of periods as translations and
the cube as a compact set meeting every orbit of the lattice.
-/

import Transformer.Landscape.PhaseModel.Defs
import Transformer.Landscape.PhaseModel.Basic
import Transformer.Landscape.PhaseModel.Periods
import Transformer.Landscape.PhaseModel.Descent
import Transformer.Landscape.PhaseModel.Division
import Transformer.Landscape.PhaseModel.Generating
import Transformer.Landscape.PhaseModel.Lift
import Transformer.Landscape.PhaseModel.Coordinates
import Transformer.Landscape.PhaseModel.Basin
import Transformer.Landscape.PhaseModel.Characters
import Transformer.Landscape.PhaseModel.Orthogonality
import Transformer.Landscape.PhaseModel.Fourier
import Transformer.Landscape.PhaseModel.Landscape
import Transformer.Landscape.PhaseModel.Convergence
import Transformer.Landscape.PhaseModel.Abelian
import Transformer.Landscape.PhaseModel.Twisted
import Transformer.Landscape.PhaseModel.Spurious
