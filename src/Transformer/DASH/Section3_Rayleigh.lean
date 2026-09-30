/-
# DASH — the Rayleigh quotient in an eigenbasis

arXiv:2602.02016v2, §3.4. Rayleigh quotients are weighted averages
of eigenvalues; strict inequality requires a component outside the
maximal eigenspace, not merely an inexactly computed vector.
-/

import Transformer.DASH.Section3_Scaling

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The actual matrix Rayleigh quotient, arXiv:2602.02016v2, §3.4. -/
def rayleigh (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  dotProduct x (A *ᵥ x) / dotProduct x x

/-- Squared Euclidean length in the eigenbasis, arXiv:2602.02016v2, §3.4. -/
def coordinateEnergy (z : Fin n → ℝ) : ℝ := ∑ i, z i ^ 2

/-- Rayleigh quotient expressed in eigenvalue coordinates,
arXiv:2602.02016v2, §3.4. -/
def spectralRayleigh (s z : Fin n → ℝ) : ℝ :=
  (∑ i, s i * z i ^ 2) / coordinateEnergy z

/-- Nonzero coordinates have positive squared length,
arXiv:2602.02016v2, §3.4, the Rayleigh denominator. -/
theorem coordinateEnergy_pos (z : Fin n → ℝ) (hz : z ≠ 0) : 0 < coordinateEnergy z := by
  have hex : ∃ i, z i ≠ 0 := by
    contrapose! hz
    funext i
    exact hz i
  obtain ⟨i, hi⟩ := hex
  exact Finset.sum_pos' (fun j _ => sq_nonneg (z j))
    ⟨i, Finset.mem_univ i, sq_pos_of_ne_zero hi⟩

/-- Nonzero vectors exist, arXiv:2602.02016v2, §3.4. -/
example : (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 := by
  intro h
  have := congrFun h 0
  norm_num at this

/-- Orthogonal frames preserve Euclidean inner products,
arXiv:2602.02016v2, §3.1 and §3.4. -/
theorem orthogonal_dotProduct (Q : Matrix (Fin n) (Fin n) ℝ)
    (z w : Fin n → ℝ) (hQ : Orthogonal Q) :
    dotProduct (Q *ᵥ z) (Q *ᵥ w) = dotProduct z w := by
  have h := Matrix.dotProduct_transpose_mulVec Q z (Q *ᵥ w)
  rw [Matrix.mulVec_mulVec, hQ.1, Matrix.one_mulVec] at h
  exact (dotProduct_comm _ _).trans h.symm

/-- Inner-product assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The coordinate formula equals the actual matrix quotient,
arXiv:2602.02016v2, §3.4, `R_A(x) = xᵀAx / xᵀx`. -/
theorem rayleigh_spectral (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (hQ : Orthogonal Q) :
    rayleigh (spectralMatrix Q s) (Q *ᵥ z) = spectralRayleigh s z := by
  have hqt : Q.transpose *ᵥ (Q *ᵥ z) = z := by
    rw [Matrix.mulVec_mulVec, hQ.1, Matrix.one_mulVec]
  have hmul : spectralMatrix Q s *ᵥ (Q *ᵥ z) = Q *ᵥ (fun i => s i * z i) := by
    have hd : Matrix.diagonal s *ᵥ z = fun i => s i * z i := by
      funext i
      exact Matrix.mulVec_diagonal s z i
    simp only [spectralMatrix, Muon.singularMatrix, ← Matrix.mulVec_mulVec, hqt, hd]
  rw [rayleigh, hmul, orthogonal_dotProduct Q _ _ hQ, orthogonal_dotProduct Q _ _ hQ]
  unfold spectralRayleigh coordinateEnergy dotProduct
  congr 1
  · apply Finset.sum_congr rfl
    intro i hi
    ring
  · apply Finset.sum_congr rfl
    intro i hi
    ring

/-- Coordinate-formula assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- A Rayleigh quotient never exceeds an upper bound on the spectrum.
Source: arXiv:2602.02016v2, §3.4, the largest-eigenvalue estimate. -/
theorem spectralRayleigh_le (s z : Fin n → ℝ) (μ : ℝ)
    (hz : z ≠ 0) (hμ : ∀ i, s i ≤ μ) : spectralRayleigh s z ≤ μ := by
  apply (div_le_iff₀ (coordinateEnergy_pos z hz)).mpr
  calc
    (∑ i, s i * z i ^ 2) ≤ ∑ i, μ * z i ^ 2 :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hμ i) (sq_nonneg _)
    _ = μ * coordinateEnergy z := by rw [coordinateEnergy, Finset.mul_sum]

/-- Rayleigh upper-bound hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 ∧
    ∀ i : Fin 1, (fun _ : Fin 1 => (1 : ℝ)) i ≤ 1 := by
  constructor
  · intro h
    have := congrFun h 0
    norm_num at this
  · norm_num

/-- For a positive semidefinite matrix every nonzero-vector Rayleigh
quotient is nonnegative, arXiv:2602.02016v2, §3.4. -/
theorem spectralRayleigh_nonneg (s z : Fin n → ℝ) (hs : ∀ i, 0 ≤ s i) :
    0 ≤ spectralRayleigh s z :=
  div_nonneg (Finset.sum_nonneg fun i _ => mul_nonneg (hs i) (sq_nonneg _))
    (Finset.sum_nonneg fun i _ => sq_nonneg (z i))

/-- Nonnegative spectra exist, arXiv:2602.02016v2, §3.4. -/
example : ∀ i : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (1 : ℝ)) i := by norm_num

/-- Any nonzero vector in an eigenspace attains its eigenvalue, including
any vector in a repeated maximal eigenspace. The source's strict inequality
for every vector other than “the largest eigenvector” needs this correction.
Source: arXiv:2602.02016v2, §3.4, the paragraph defining `v_PI`. -/
theorem rayleigh_eigenvector (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ)
    (μ : ℝ) (hx : dotProduct x x ≠ 0) (heigen : A *ᵥ x = μ • x) :
    rayleigh A x = μ := by
  rw [rayleigh, heigen]
  have hnum : dotProduct x (μ • x) = μ * dotProduct x x := by
    simp only [dotProduct, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hnum]
  exact mul_div_cancel_right₀ μ hx

/-- Eigenvector assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : dotProduct (fun _ : Fin 1 => (1 : ℝ)) (fun _ => (1 : ℝ)) ≠ 0 ∧
    (1 : Matrix (Fin 1) (Fin 1) ℝ) *ᵥ (fun _ => (1 : ℝ)) =
      (1 : ℝ) • (fun _ : Fin 1 => (1 : ℝ)) := by
  norm_num [dotProduct]

end Transformer.DASH
