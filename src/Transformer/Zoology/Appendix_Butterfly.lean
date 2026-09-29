/-
# Butterfly factors and their shifted-diagonal decomposition

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`, equation
`eq: butterfly-split`, and Proposition `prop: butterfly-decomposition`.
Using `Fin 2 × Fin m` instead of `Fin (2m)` merely exposes the two halves.
The algebraic decomposition is proved, and a counterexample shows why the
shifts omitted in one later proof display cannot be dropped.
-/

import Mathlib

namespace Transformer.Zoology

/-- A vector of two halves, each with `m` coordinates.
Source: Appendix `def: butterfly`, size `2m`. -/
abbrev HalfVector (m : ℕ) := Fin 2 → Fin m → ℝ

/-- The four diagonals in a butterfly factor's `2 × 2` block matrix.
Source: Appendix `def: butterfly`. -/
structure ButterflyFactor (m : ℕ) where
  upperLeft : Fin m → ℝ
  upperRight : Fin m → ℝ
  lowerLeft : Fin m → ℝ
  lowerRight : Fin m → ℝ

/-- Apply the butterfly factor to a vector.
Source: Appendix `def: butterfly`. -/
def ButterflyFactor.apply {m : ℕ} (f : ButterflyFactor m)
    (u : HalfVector m) : HalfVector m :=
  fun half t =>
    if half = 0 then f.upperLeft t * u 0 t + f.upperRight t * u 1 t
    else f.lowerLeft t * u 0 t + f.lowerRight t * u 1 t

/-- Exchange the two halves, corresponding to the half-length cyclic shift
in equation `eq: butterfly-split`. -/
def swapHalves {m : ℕ} (u : HalfVector m) : HalfVector m :=
  fun half t => if half = 0 then u 1 t else u 0 t

/-- The diagonal term `D₁u` in equation `eq: butterfly-split`. -/
def ButterflyFactor.mainDiagonal {m : ℕ} (f : ButterflyFactor m)
    (u : HalfVector m) : HalfVector m :=
  fun half t => if half = 0 then f.upperLeft t * u 0 t
    else f.lowerRight t * u 1 t

/-- The diagonal preceding a half-length shift, `D₂Su`.
Source: Appendix equation `eq: butterfly-split`. -/
def ButterflyFactor.upperDiagonal {m : ℕ} (f : ButterflyFactor m)
    (u : HalfVector m) : HalfVector m :=
  fun half t => if half = 0 then f.upperRight t * u 0 t else 0

/-- The diagonal following a half-length shift, `SD₃u`.
Source: Appendix equation `eq: butterfly-split`. -/
def ButterflyFactor.lowerDiagonal {m : ℕ} (f : ButterflyFactor m)
    (u : HalfVector m) : HalfVector m :=
  fun half t => if half = 0 then f.lowerLeft t * u 0 t else 0

/-- Every even-size butterfly factor is the sum of its main diagonal and
two shifted off-diagonals.  The paper states the power-of-two case; the
identity itself needs only two equal-size halves.
Source: Appendix Proposition `prop: butterfly-decomposition`. -/
theorem butterfly_factor_decomposition {m : ℕ} (f : ButterflyFactor m)
    (u : HalfVector m) :
    f.apply u = fun half t =>
      f.mainDiagonal u half t +
      f.upperDiagonal (swapHalves u) half t +
      swapHalves (f.lowerDiagonal u) half t := by
  funext half t
  fin_cases half <;>
    simp [ButterflyFactor.apply, ButterflyFactor.mainDiagonal,
      ButterflyFactor.upperDiagonal, ButterflyFactor.lowerDiagonal,
      swapHalves]; ring

/-- Removing both shifts from the off-diagonal terms changes the operator.
This refutes the unshifted display `B x = D₁ ⊙ u + D₂ ⊙ u + D₃ ⊙ u` in the
proof of Appendix Lemma `prop: butterfly-hyena`; the preceding Proposition
`prop: butterfly-decomposition` itself has the shifts correctly. -/
theorem butterfly_unshifted_display_false :
    ∃ f : ButterflyFactor 1, ∃ u : HalfVector 1,
      f.apply u 0 0 ≠
        f.mainDiagonal u 0 0 +
        f.upperDiagonal u 0 0 + f.lowerDiagonal u 0 0 := by
  let f : ButterflyFactor 1 := {
    upperLeft := fun _ => 0
    upperRight := fun _ => 1
    lowerLeft := fun _ => 0
    lowerRight := fun _ => 0
  }
  let u : HalfVector 1 := fun half _ => if half = 0 then 0 else 1
  refine ⟨f, u, ?_⟩
  norm_num [ButterflyFactor.apply, ButterflyFactor.mainDiagonal,
    ButterflyFactor.upperDiagonal, ButterflyFactor.lowerDiagonal, f, u]

end Transformer.Zoology
