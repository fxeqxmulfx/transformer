/-
# Phase model over a finite abelian group — definitions

`Landscape.PhaseModel.Defs` fits phases to division in a cyclic group of order `n`.
Here the group is any finite abelian group `G`, written additively, with
one phase `ψ_x ∈ ℝ` per element, `ψ ∈ ℝ^G`.  The relation of the row
`x / y = x - y` is `ψ_x - ψ_y - ψ_{x-y} ∈ 2πℤ`, with integer row
`e_x - e_y - e_{x-y}` (`divisionRow`), and the energy of a set `S` of rows
is `divisionEnergy S`, the phase energy `Σ_r (1 - cos ⟪m_r, ψ⟫)` of those
rows.  A character `χ : G → ℂ` takes values of modulus `1`; its phases are
its arguments `ψ_x = arg χ(x)` (`characterPhase`).  For `G = ℤ/n` the
characters are `x ↦ e^{2πijx/n}`, whose phases `2πjx/n` of
`Landscape.PhaseModel.characterPhase` agree with these up to `2πℤ`.  The periods are
the lattice `2πℤ^G` (`periods`).
-/

import Transformer.Landscape.PhaseModel.Defs
import Mathlib.Analysis.Fourier.FiniteAbelian.PontryaginDuality
import Mathlib.Analysis.SpecialFunctions.Complex.Arg

open Real

namespace Transformer.Landscape.PhaseModel.Abelian

variable {G : Type*}

/-- The vector of `ℝ^G` with the integer coordinates `a`. -/
def intVec (a : G → ℤ) : EuclideanSpace ℝ G :=
  WithLp.toLp 2 fun i => (a i : ℝ)

/-- **The row of `x / y = x - y`** in a finite abelian group `G`:
`e_x - e_y - e_{x-y}`, for the pair `p = (x, y)`. -/
def divisionRow [AddCommGroup G] [DecidableEq G] (p : G × G) : G → ℤ :=
  Pi.single p.1 1 - Pi.single p.2 1 - Pi.single (p.1 - p.2) 1

/-- **The energy of a set `S` of rows of the division table of `G`**:
`V_S(ψ) = Σ_{(x, y) ∈ S} (1 - cos (ψ_x - ψ_y - ψ_{x-y}))`
(`divisionEnergy_apply`). -/
noncomputable def divisionEnergy [AddCommGroup G] [Fintype G] [DecidableEq G]
    (S : Finset (G × G)) : EuclideanSpace ℝ G → ℝ :=
  phaseEnergy fun p : S => intVec (divisionRow p.1)

/-- **The phases of a character** `χ` of `G`: `ψ_x = arg χ(x)`. -/
noncomputable def characterPhase [AddCommGroup G] (χ : AddChar G ℂ) :
    EuclideanSpace ℝ G :=
  WithLp.toLp 2 fun x => Complex.arg (χ x)

/-- The lattice `2πℤ^G` of periods. -/
def periods (G : Type*) : Set (EuclideanSpace ℝ G) :=
  {v | ∀ i, ∃ k : ℤ, v.ofLp i = k * (2 * π)}

end Transformer.Landscape.PhaseModel.Abelian
