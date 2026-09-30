/-
# DASH — algebraic correctness of exact inverse roots

arXiv:2602.02016v2, §2.2 and §3.1–3.3. These identities verify the
inverse roots used by Shampoo, including the exact EVD implementation.
-/

import Transformer.DASH.Section2_History

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Opposite spectral powers are two-sided matrix inverses on a positive
spectrum. Source: arXiv:2602.02016v2, §3.1, `A^p=QΛ^pQᵀ`. -/
theorem spectralPower_opposite (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (exponent : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) :
    spectralPower Q s exponent * spectralPower Q s (-exponent) = 1 ∧
      spectralPower Q s (-exponent) * spectralPower Q s exponent = 1 := by
  constructor <;> rw [spectralPower_mul Q s _ _ hQ hs]
  · simpa [spectralPower] using spectralMatrix_one Q hQ
  · simpa [spectralPower] using spectralMatrix_one Q hQ

/-- Opposite-power assumptions are satisfiable, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- An integer power of a spectral power multiplies its real exponent.
Source: arXiv:2602.02016v2, §3.1–3.3, the defining root identity. -/
theorem spectralPower_natPow (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (exponent : ℝ) (k : ℕ) (hQ : Orthogonal Q)
    (hs : ∀ i, 0 ≤ s i) :
    spectralPower Q s exponent ^ k = spectralPower Q s (exponent * k) := by
  rw [spectralPower, spectralMatrix_pow Q _ hQ, spectralPower]
  congr 1
  funext i
  rw [← Real.rpow_natCast, ← Real.rpow_mul (hs i)]

/-- Integer-root assumptions are satisfiable, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Raising an inverse `p`th root to its order yields the inverse matrix.
Positive order and eigenvalues are required; symmetry alone is insufficient.
Source: arXiv:2602.02016v2, §3.1–3.2, `A^(-1/p)`. -/
theorem spectral_inverseRoot_identity (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (p : ℕ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hp : 0 < p) :
    spectralMatrix Q s * spectralPower Q s (-(1 / (p : ℝ))) ^ p = 1 ∧
      spectralPower Q s (-(1 / (p : ℝ))) ^ p * spectralMatrix Q s = 1 := by
  have hexp : (-(1 / (p : ℝ))) * p = (-1 : ℝ) := by
    have hp' : (p : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
    field_simp
  rw [spectralPower_natPow Q s _ p hQ (fun i => (hs i).le), hexp]
  have hone : spectralMatrix Q s = spectralPower Q s 1 := by simp [spectralPower]
  rw [hone]
  exact spectralPower_opposite Q s 1 hQ hs

/-- Inverse-root hypotheses are satisfiable, arXiv:2602.02016v2, §3.1–3.2. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧ 0 < (4 : ℕ) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The canonical EVD inverse root used by the Shampoo update is an actual
inverse root of its original matrix, not an unchecked spectral surrogate.
Source: arXiv:2602.02016v2, §2.2, Algorithm 1, and §3.1, EVD. -/
theorem evdInverseRoot_correct (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : A.PosDef) (p : ℕ) (hp : 0 < p) :
    A * evdInverseRoot A hA p ^ p = 1 ∧ evdInverseRoot A hA p ^ p * A = 1 := by
  have h := spectral_inverseRoot_identity (evdFrame A hA.1) hA.1.eigenvalues p
    (evdFrame_orthogonal A hA.1) hA.eigenvalues_pos hp
  simpa only [evdInverseRoot, ← evd_reconstruction A hA.1] using h

/-- Exact-EVD inverse-root assumptions are satisfiable,
arXiv:2602.02016v2, §2.2 and §3.1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef ∧ 0 < (4 : ℕ) := by
  refine ⟨?_, by norm_num⟩
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

end Transformer.DASH
