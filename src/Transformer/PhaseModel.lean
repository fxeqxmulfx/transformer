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
periods, and every zero is nondegenerate.

| Module | Contents |
| --- | --- |
| `PhaseModel.Defs` | the phase energy, integer rows, the rows of division, the periods `2πℤⁿ`, the cube `[0, 2π]ⁿ` |
| `PhaseModel.Basic` | zeros and invariance; the gradient, the Hessian and its Lipschitz constant; nondegenerate zeros |
| `PhaseModel.Periods` | integer rows and periods; translates into the cube; the energy of division |
| `PhaseModel.Descent` | limit points are critical; convergence from nondegenerate limit points; almost every orbit converges to a local minimizer when every critical point is nondegenerate |
| `PhaseModel.Division` | the full division table: only `0` is orthogonal to every row; the characters are zeros of every `V_S` and, up to periods, the only zeros of `V_full`; the norms of the rows |
| `PhaseModel.Generating` | sets of rows generating the full table over `ℤ`: the zeros of `V_S` are the characters up to periods, all nondegenerate; convergence from a character as a limit point |

The convergence results instantiate `Transformer.SaddleAvoidance` with the
lattice of periods as translations and the cube as a compact set meeting
every orbit of the lattice.
-/

import Transformer.PhaseModel.Defs
import Transformer.PhaseModel.Basic
import Transformer.PhaseModel.Periods
import Transformer.PhaseModel.Descent
import Transformer.PhaseModel.Division
import Transformer.PhaseModel.Generating
