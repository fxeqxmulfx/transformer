/-
# AdaFisher: normalization-layer factors and exact Fisher blocks

arXiv:2405.16397v3, §3.2, Proposition 3.1, and Appendix A.2.
Scale and shift have one parameter per channel. Their exact Fisher blocks
are channel-by-channel matrices, not channel-pair-by-channel-pair matrices.
-/

import Transformer.AdaFisher.Section2_Fisher

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {c t n : ℕ}

/-- The affine part of a normalization layer, Appendix A.2, Proposition 3.1. -/
def normalizationAffine (ν β : Fin c → ℝ) (h : Fin t → Fin c → ℝ) :
    Fin t → Fin c → ℝ := fun x i => ν i * h x i + β i

/-- Scale gradient summed over normalization sites, Appendix A.2. -/
def scaleGradient (h s : Fin t → Fin c → ℝ) : Fin c → ℝ :=
  fun i => ∑ x, h x i * s x i

/-- Shift gradient summed over normalization sites, Appendix A.2. -/
def shiftGradient (s : Fin t → Fin c → ℝ) : Fin c → ℝ :=
  fun i => ∑ x, s x i

/-- The normalization activation factor specified in Proposition 3.1.
The source calls these both pre-normalized and normalized activations;
`h` denotes the activations actually entering the affine scale/shift. -/
def normalizationH (h : Fin t → Fin c → ℝ) : Matrix (Fin c) (Fin c) ℝ :=
  fun i j => (∑ x, h x i * h x j) / t

/-- The normalization sensitivity factor of Proposition 3.1. -/
def normalizationS (s : Fin t → Fin c → ℝ) : Matrix (Fin c) (Fin c) ℝ :=
  fun i j => (∑ x, s x i * s x j) / t

/-- The shift activation factor in Proposition 3.1 is the all-ones matrix. -/
def normalizationBiasH (c : ℕ) : Matrix (Fin c) (Fin c) ℝ :=
  outer (fun _ => 1) (fun _ => 1)

/-- Empirical scale Fisher has all spatial cross terms, Appendix A.2.
This corrects the proof's dimension-changing Kronecker equality: the exact
block is `c × c` and uses channelwise products, with two site indices. -/
theorem normalization_scale_fisher_exact (w : Fin n → ℝ)
    (h s : Fin n → Fin t → Fin c → ℝ) (i j : Fin c) :
    secondMoment w (fun k => scaleGradient (h k) (s k)) i j =
      ∑ k, ∑ x, ∑ y, w k * (h k x i * s k x i) * (h k y j * s k y j) := by
  simp only [secondMoment, scaleGradient, Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k hk
  rw [Finset.sum_comm]

/-- Exact shift Fisher, Appendix A.2. The source drops cross-site terms
without an assumption that would make them vanish. -/
theorem normalization_shift_fisher_exact (w : Fin n → ℝ)
    (s : Fin n → Fin t → Fin c → ℝ) (i j : Fin c) :
    secondMoment w (fun k => shiftGradient (s k)) i j =
      ∑ k, ∑ x, ∑ y, w k * s k x i * s k y j := by
  simp only [secondMoment, shiftGradient, Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k hk
  rw [Finset.sum_comm]

/-- Counterexample to the “Bias term factorization” equality in the proof
of Proposition 3.1, Appendix A.2. One channel and two deterministic sites
give exact Fisher four, whereas the proposed product of factors is one.
The same example refutes the corresponding scale equality. -/
theorem normalization_factorization_counterexample :
    let h : Fin 2 → Fin 1 → ℝ := fun _ _ => 1
    let s : Fin 2 → Fin 1 → ℝ := fun _ _ => 1
    outer (shiftGradient s) (shiftGradient s) 0 0 = 4 ∧
    outer (scaleGradient h s) (scaleGradient h s) 0 0 = 4 ∧
    normalizationBiasH 1 0 0 * normalizationS s 0 0 = 1 ∧
    normalizationH h 0 0 * normalizationS s 0 0 = 1 := by
  norm_num [outer, shiftGradient, scaleGradient, normalizationBiasH,
    normalizationH, normalizationS, Fin.sum_univ_two]

/-- The factor diagonal is the empirical mean of squared activations,
§3.2, Proposition 3.1 and Appendix A.3. -/
theorem normalizationH_diagonal (h : Fin t → Fin c → ℝ) (i : Fin c) :
    normalizationH h i i = (∑ x, h x i ^ 2) / t := by
  simp only [normalizationH, pow_two]

/-- The normalization KF diagonal is nonnegative, Proposition 3.1. -/
theorem normalizationH_diagonal_nonneg (h : Fin t → Fin c → ℝ) (i : Fin c) :
    0 ≤ normalizationH h i i := by
  rw [normalizationH_diagonal]
  exact div_nonneg (Finset.sum_nonneg fun x _ => sq_nonneg _) (Nat.cast_nonneg _)

end Transformer.AdaFisher
