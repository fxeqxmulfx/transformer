/-
# DASH — exact eigenbasis description of normalized Power Iteration

arXiv:2602.02016v2, §3.4–3.5. Normalization changes vector length but
preserves the Rayleigh quotient. A nonzero dominant component prevents
the iterates from reaching zero.
-/

import Transformer.DASH.Section3_PowerIteration

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Unnormalized Power Iteration in the eigenbasis,
arXiv:2602.02016v2, §3.4–3.5, repeated matrix-vector products. -/
def powerCoordinates (s z : Fin n → ℝ) (k : ℕ) : Fin n → ℝ :=
  fun i => s i ^ k * z i

/-- Spectral matrix-vector multiplication acts coordinatewise.
Source: arXiv:2602.02016v2, §3.1 and §3.4, Power Iteration in an eigenbasis. -/
theorem spectralMatrix_mulVec (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (hQ : Orthogonal Q) :
    spectralMatrix Q s *ᵥ (Q *ᵥ z) = Q *ᵥ (fun i => s i * z i) := by
  have hqt : Q.transpose *ᵥ (Q *ᵥ z) = z := by
    rw [Matrix.mulVec_mulVec, hQ.1, Matrix.one_mulVec]
  simp only [spectralMatrix, Muon.singularMatrix, ← Matrix.mulVec_mulVec, hqt]
  congr 1
  funext i
  exact Matrix.mulVec_diagonal s z i

/-- Spectral multiplication assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- A nonzero scalar multiple has the same Rayleigh quotient.
Source: arXiv:2602.02016v2, §3.4, normalization of PI vectors. -/
theorem rayleigh_smul (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ)
    (c : ℝ) (hc : c ≠ 0) : rayleigh A (c • x) = rayleigh A x := by
  unfold rayleigh
  rw [Matrix.mulVec_smul]
  have hdot (x y : Fin n → ℝ) : dotProduct (c • x) (c • y) = c ^ 2 * dotProduct x y := by
    simp only [dotProduct, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hdot, hdot]
  exact mul_div_mul_left _ _ (pow_ne_zero 2 hc)

/-- Nonzero scaling hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (2 : ℝ) ≠ 0 := by norm_num

/-- An orthogonal frame sends a nonzero coordinate vector to a nonzero
vector. Source: arXiv:2602.02016v2, §3.1 and §3.4. -/
theorem orthogonal_mulVec_ne_zero (Q : Matrix (Fin n) (Fin n) ℝ)
    (z : Fin n → ℝ) (hQ : Orthogonal Q) (hz : z ≠ 0) : Q *ᵥ z ≠ 0 := by
  intro h
  apply hz
  have heq : Q.transpose *ᵥ (Q *ᵥ z) = z := by
    rw [Matrix.mulVec_mulVec, hQ.1, Matrix.one_mulVec]
  rw [h, Matrix.mulVec_zero] at heq
  exact heq.symm

/-- Nonzero orthogonal-image assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 := by
  constructor
  · simp [Orthogonal, Muon.OrthonormalColumns]
  · intro h
    have := congrFun h 0
    norm_num at this

/-- Every normalized PI vector is a nonzero scalar multiple of the exact
power vector, provided one positive-eigenvalue coordinate starts nonzero.
The source's missing starting-vector condition is explicit.
Source: arXiv:2602.02016v2, §3.4–3.5, PI estimation. -/
theorem powerIterate_coordinates (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (j : Fin n) (hQ : Orthogonal Q)
    (hsj : 0 < s j) (hzj : z j ≠ 0) (k : ℕ) :
    ∃ c : ℝ, c ≠ 0 ∧ powerIterate (spectralMatrix Q s) (Q *ᵥ z) k =
      c • (Q *ᵥ powerCoordinates s z k) := by
  induction k with
  | zero =>
    refine ⟨1, by norm_num, ?_⟩
    have hzero : powerCoordinates s z 0 = z := by
      funext i
      simp [powerCoordinates]
    simp [powerIterate, hzero]
  | succ k ih =>
    obtain ⟨c, hc, heq⟩ := ih
    have hcoord : (fun i => s i * powerCoordinates s z k i) = powerCoordinates s z (k + 1) := by
      funext i
      simp only [powerCoordinates, pow_succ]
      ring
    have hz : powerCoordinates s z (k + 1) ≠ 0 := by
      intro h
      have hne : powerCoordinates s z (k + 1) j ≠ 0 :=
        mul_ne_zero (pow_ne_zero _ hsj.ne') hzj
      exact hne (congrFun h j)
    have hqz := orthogonal_mulVec_ne_zero Q _ hQ hz
    have hy : c • (Q *ᵥ powerCoordinates s z (k + 1)) ≠ 0 := smul_ne_zero hc hqz
    have henergy : 0 < dotProduct
        (c • (Q *ᵥ powerCoordinates s z (k + 1)))
        (c • (Q *ᵥ powerCoordinates s z (k + 1))) := by
      simpa only [coordinateEnergy, dotProduct, pow_two] using coordinateEnergy_pos _ hy
    have hnorm : Real.sqrt (dotProduct
        (c • (Q *ᵥ powerCoordinates s z (k + 1)))
        (c • (Q *ᵥ powerCoordinates s z (k + 1)))) ≠ 0 :=
      (Real.sqrt_pos.2 henergy).ne'
    refine ⟨_, mul_ne_zero (inv_ne_zero hnorm) hc, ?_⟩
    rw [powerIterate, heq]
    simp only [powerStep, Matrix.mulVec_smul, spectralMatrix_mulVec Q _ _ hQ,
      hcoord, smul_smul]

/-- The nonvanishing-coordinate assumptions are satisfiable,
arXiv:2602.02016v2, §3.4–3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) 0 ∧
    (fun _ : Fin 1 => (1 : ℝ)) 0 ≠ 0 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The normalized implementation returns the exact power-vector Rayleigh
quotient at every step. Source: arXiv:2602.02016v2, §3.4–3.5. -/
theorem powerIterate_rayleigh (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (j : Fin n) (hQ : Orthogonal Q)
    (hsj : 0 < s j) (hzj : z j ≠ 0) (k : ℕ) :
    rayleigh (spectralMatrix Q s) (powerIterate (spectralMatrix Q s) (Q *ᵥ z) k) =
      spectralRayleigh s (powerCoordinates s z k) := by
  obtain ⟨c, hc, heq⟩ := powerIterate_coordinates Q s z j hQ hsj hzj k
  rw [heq, rayleigh_smul _ _ c hc, rayleigh_spectral Q s _ hQ]

/-- Rayleigh-iteration assumptions are satisfiable,
arXiv:2602.02016v2, §3.4–3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) 0 ∧
    (fun _ : Fin 1 => (1 : ℝ)) 0 ≠ 0 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
