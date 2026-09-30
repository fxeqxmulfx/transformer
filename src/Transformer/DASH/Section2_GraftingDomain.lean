/-
# DASH — the nonzero-gradient domain of regularized grafting

arXiv:2602.02016v2, §2.2–2.3. Positive regularization makes the exact
inverse roots invertible. Consequently a nonzero gradient produces a
nonzero Shampoo direction and the norm-matching scale is well defined.
-/

import Transformer.DASH.Section3_InverseRootCorrectness

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- A nonzero matrix has positive Frobenius norm, the denominator used by
grafting. Source: arXiv:2602.02016v2, §2.3, `s_t=‖P_t‖_F/‖U_t‖_F`. -/
theorem frobeniusNorm_pos (X : Matrix (Fin m) (Fin n) ℝ) (hX : X ≠ 0) :
    0 < frobeniusNorm X := by
  have hex : ∃ i j, X i j ≠ 0 := by
    contrapose! hX
    ext i j
    exact hX i j
  obtain ⟨i, j, hij⟩ := hex
  apply Real.sqrt_pos.2
  exact lt_of_lt_of_le (sq_pos_of_ne_zero hij) (Muon.entry_sq_le_squaredFrobenius X i j)

/-- Nonzero matrix hypotheses are satisfiable, arXiv:2602.02016v2, §2.3. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  intro h
  have := congrFun (congrFun h 0) 0
  norm_num at this

/-- Exact positive-definite EVD inverse roots have two-sided inverses.
Source: arXiv:2602.02016v2, §3.1, spectral inverse-root construction. -/
theorem evdInverseRoot_invertible (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : A.PosDef) (p : ℕ) :
    ∃ B : Matrix (Fin n) (Fin n) ℝ,
      evdInverseRoot A hA p * B = 1 ∧ B * evdInverseRoot A hA p = 1 := by
  refine ⟨spectralPower (evdFrame A hA.1) hA.1.eigenvalues (-(-(1 / (p : ℝ)))), ?_⟩
  exact spectralPower_opposite _ _ _ (evdFrame_orthogonal A hA.1) hA.eigenvalues_pos

/-- Inverse-root invertibility assumptions are satisfiable,
arXiv:2602.02016v2, §3.1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef := by
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- Invertible left and right preconditioners preserve the zero-gradient
condition. Source: arXiv:2602.02016v2, §2.3, `U=L^(-1/4)GR^(-1/4)`. -/
theorem preconditionedGradient_eq_zero_iff
    (L L' : Matrix (Fin m) (Fin m) ℝ) (R R' : Matrix (Fin n) (Fin n) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) (hL : L' * L = 1) (hR : R * R' = 1) :
    preconditionedGradient L G R = 0 ↔ G = 0 := by
  constructor
  · intro h
    have hinv : L' * preconditionedGradient L G R * R' = G := by
      calc
        L' * preconditionedGradient L G R * R' = (L' * L) * G * (R * R') := by
          simp only [preconditionedGradient, Matrix.mul_assoc]
        _ = G := by rw [hL, hR, Matrix.one_mul, Matrix.mul_one]
    rw [h, Matrix.mul_zero, Matrix.zero_mul] at hinv
    exact hinv.symm
  · intro h
    simp [h, preconditionedGradient]

/-- The inverse-preconditioner assumptions are satisfiable,
arXiv:2602.02016v2, §2.3. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ) * 1 = 1 ∧
    (1 : Matrix (Fin 1) (Fin 1) ℝ) * 1 = 1 := by simp

/-- For any nonzero gradient, the actual regularized Shampoo direction
matches the reference norm after grafting. This derives the nonzero
denominator from regularization instead of assuming it.
Source: arXiv:2602.02016v2, §2.2–2.3, Algorithm 1 and Adam grafting. -/
theorem regularized_graft_norm (L : Matrix (Fin m) (Fin m) ℝ)
    (R : Matrix (Fin n) (Fin n) ℝ) (G P : Matrix (Fin m) (Fin n) ℝ) (ε : ℝ)
    (hL : L.PosSemidef) (hR : R.PosSemidef) (hε : 0 < ε) (hG : G ≠ 0) :
    frobeniusNorm (graft
      (preconditionedGradient
        (evdInverseRoot (L + ε • 1) (regularized_posDef L ε hL hε) 4) G
        (evdInverseRoot (R + ε • 1) (regularized_posDef R ε hR hε) 4)) P) =
      frobeniusNorm P := by
  obtain ⟨L', _, hL'⟩ := evdInverseRoot_invertible (L + ε • 1) (regularized_posDef L ε hL hε) 4
  obtain ⟨R', hR', _⟩ := evdInverseRoot_invertible (R + ε • 1) (regularized_posDef R ε hR hε) 4
  apply graft_norm
  apply frobeniusNorm_pos
  exact fun h => hG ((preconditionedGradient_eq_zero_iff _ L' _ R' G hL' hR').mp h)

/-- Regularized-grafting hypotheses are satisfiable,
arXiv:2602.02016v2, §2.2–2.3. -/
example : (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧
    (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧ (0 : ℝ) < 1 ∧
    (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  refine ⟨Matrix.PosSemidef.zero, Matrix.PosSemidef.zero, by norm_num, ?_⟩
  intro h
  have := congrFun (congrFun h 0) 0
  norm_num at this

end Transformer.DASH
