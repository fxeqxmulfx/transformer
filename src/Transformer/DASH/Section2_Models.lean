/-
# DASH — Shampoo and grafting

Modoranu et al., arXiv:2602.02016v2, §2, Algorithm 1
(`algorithm:default-shampoo`). Computations are over real matrices.
Inverse-root solvers and hardware timing are treated separately.
-/

import Transformer.Muon.Section2_PolarIdentities

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- The left exponential moving average in Algorithm 1,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
def leftEma (β : ℝ) (L : Matrix (Fin m) (Fin m) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin m) ℝ :=
  β • L + (1 - β) • (G * G.transpose)

/-- The right exponential moving average in Algorithm 1,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
def rightEma (β : ℝ) (R : Matrix (Fin n) (Fin n) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  β • R + (1 - β) • (G.transpose * G)

/-- The preconditioned direction from the computed left and right inverse
roots, arXiv:2602.02016v2, §2.2–2.3. -/
def preconditionedGradient (P : Matrix (Fin m) (Fin m) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) (Q : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ := P * G * Q

/-- Frobenius norm used by grafting and matrix scaling,
arXiv:2602.02016v2, §2.3 and §3.4. -/
def frobeniusNorm (X : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  Real.sqrt (Muon.squaredFrobenius X)

/-- Adam grafting: retain the Shampoo direction and match the reference
direction's norm, arXiv:2602.02016v2, §2.3, “Grafting”.
At zero Shampoo direction the total division gives zero, not a unit direction. -/
def graft (U P : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  (frobeniusNorm P / frobeniusNorm U) • U

/-- Parameter update after grafting, arXiv:2602.02016v2, §2.3, “Grafting”. -/
def graftedStep (η : ℝ) (θ U P : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ := θ - η • graft U P

/-- A right Gram matrix is positive semidefinite,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
theorem rightGram_posSemidef (G : Matrix (Fin m) (Fin n) ℝ) :
    (G.transpose * G).PosSemidef := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_conjTranspose_mul_self G

/-- A left Gram matrix is positive semidefinite,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
theorem leftGram_posSemidef (G : Matrix (Fin m) (Fin n) ℝ) :
    (G * G.transpose).PosSemidef := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial,
    Matrix.transpose_transpose] using Matrix.posSemidef_conjTranspose_mul_self G.transpose

/-- EMA preserves positive semidefiniteness on the left,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
theorem leftEma_posSemidef (β : ℝ) (L : Matrix (Fin m) (Fin m) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hL : L.PosSemidef) :
    (leftEma β L G).PosSemidef :=
  (hL.smul hβ).add ((leftGram_posSemidef G).smul (sub_nonneg.mpr hβ'))

/-- The EMA hypotheses are satisfiable, arXiv:2602.02016v2, §2.2. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef := by
  exact ⟨by norm_num, by norm_num, Matrix.PosSemidef.zero⟩

/-- EMA preserves positive semidefiniteness on the right,
arXiv:2602.02016v2, §2.2, `algorithm:default-shampoo`. -/
theorem rightEma_posSemidef (β : ℝ) (R : Matrix (Fin n) (Fin n) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hR : R.PosSemidef) :
    (rightEma β R G).PosSemidef :=
  (hR.smul hβ).add ((rightGram_posSemidef G).smul (sub_nonneg.mpr hβ'))

/-- The right EMA hypotheses are satisfiable, arXiv:2602.02016v2, §2.2. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef := by
  exact ⟨by norm_num, by norm_num, Matrix.PosSemidef.zero⟩

/-- Frobenius norm scales by the absolute scalar,
arXiv:2602.02016v2, §2.3, the grafting normalization. -/
theorem frobeniusNorm_smul (c : ℝ) (X : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusNorm (c • X) = |c| * frobeniusNorm X := by
  rw [frobeniusNorm, Muon.squaredFrobenius_smul, Real.sqrt_mul (sq_nonneg c),
    Real.sqrt_sq_eq_abs, frobeniusNorm]

/-- Grafting matches the reference norm when the Shampoo norm is positive.
The source's division requires this condition, which excludes a zero direction.
Source: arXiv:2602.02016v2, §2.3, “Grafting”. -/
theorem graft_norm (U P : Matrix (Fin m) (Fin n) ℝ) (hU : 0 < frobeniusNorm U) :
    frobeniusNorm (graft U P) = frobeniusNorm P := by
  rw [graft, frobeniusNorm_smul, abs_of_nonneg
    (show 0 ≤ frobeniusNorm P / frobeniusNorm U from
      div_nonneg (Real.sqrt_nonneg _) hU.le)]
  exact div_mul_cancel₀ _ hU.ne'

/-- Positive Shampoo norms exist, arXiv:2602.02016v2, §2.3. -/
example : 0 < frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [frobeniusNorm, Muon.squaredFrobenius]

/-- Grafting cannot normalize a zero Shampoo update to a nonzero reference
norm, arXiv:2602.02016v2, §2.3, “Grafting”. -/
theorem zero_graft_counterexample :
    frobeniusNorm (graft (0 : Matrix (Fin 1) (Fin 1) ℝ) 1) = 0 ∧
      frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 := by
  norm_num [graft, frobeniusNorm, Muon.squaredFrobenius]

end Transformer.DASH
