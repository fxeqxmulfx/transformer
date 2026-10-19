module

public import Transformer.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Phase model — definitions

A phase model fits one phase `ψ_i` on the circle to each token `i`, so that
a finite set of integer relations `⟪m_r, ψ⟫ ∈ 2πℤ` holds. Gradient descent
runs on the energy `V(ψ) = Σ_r (1 - cos ⟪m_r, ψ⟫)` (`phaseEnergy`) over
`ℝⁿ`, which vanishes exactly where every relation holds and is invariant
under the lattice `2πℤⁿ` (`periods`); every point has a translate in the
cube `[0, 2π]ⁿ` (`cube`).

For division in a cyclic group of order `n`, with phases indexed by discrete
logarithms, the relation of the row `x / y = x - y` is
`ψ_x - ψ_y - ψ_{x-y} ∈ 2πℤ`, with integer row `e_x - e_y - e_{x-y}`
(`divisionRow`): the relation `χ(a) = χ(b) χ(a/b)` of a character `χ`. The
energy of a set `S` of rows is `divisionEnergy S`. The character
`x ↦ e^{2πijx/n}` of `ℤ/n` has the phases `ψ_x = 2πjx/n`
(`characterPhase`).
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel

/-- **The phase energy** `V(ψ) = Σ_r (1 - cos ⟪m_r, ψ⟫)` of a finite family of
rows `m_r`. -/
noncomputable def phaseEnergy {E R : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [Fintype R] (m : R → E) (ψ : E) : ℝ :=
  ∑ r, (1 - cos (inner ℝ (m r) ψ))

/-- The vector of `ℝⁿ` with the integer coordinates `a`. -/
def intVec {n : ℕ} (a : Fin n → ℤ) : EucSpace n :=
  WithLp.toLp 2 fun i => (a i : ℝ)

/-- **The row of `x / y = x - y`** in discrete logarithms modulo `n`:
`e_x - e_y - e_{x-y}`, for the pair `p = (x, y)`. -/
def divisionRow {n : ℕ} (p : Fin n × Fin n) : Fin n → ℤ :=
  Pi.single p.1 1 - Pi.single p.2 1 - Pi.single (p.1 - p.2) 1

/-- **The energy of a set `S` of rows of the division table**:
`V_S(ψ) = Σ_{(x, y) ∈ S} (1 - cos (ψ_x - ψ_y - ψ_{x-y}))`
(`divisionEnergy_apply`). -/
noncomputable def divisionEnergy {n : ℕ} (S : Finset (Fin n × Fin n)) : EucSpace n → ℝ :=
  phaseEnergy fun p : S => intVec (divisionRow p.1)

/-- **The phases of the character** `x ↦ e^{2πijx/n}` of `ℤ/n`:
`ψ_x = 2πjx/n` for `x = 0, …, n - 1`. -/
noncomputable def characterPhase (n : ℕ) (j : ℤ) : EucSpace n :=
  WithLp.toLp 2 fun x => 2 * π * j * (x : ℕ) / n

/-- The lattice `2πℤⁿ` of periods. -/
def periods (n : ℕ) : Set (EucSpace n) :=
  {v | ∀ i, ∃ k : ℤ, v.ofLp i = k * (2 * π)}

/-- The cube `[0, 2π]ⁿ`. -/
def cube (n : ℕ) : Set (EucSpace n) :=
  {v | ∀ i, v.ofLp i ∈ Set.Icc 0 (2 * π)}

end Transformer.Landscape.PhaseModel
