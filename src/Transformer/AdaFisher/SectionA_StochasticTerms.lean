/-
# AdaFisher: corrected squared-update and harmonic bounds

arXiv:2405.16397v3, Proposition 3.4, Appendix A.2, C1 term in `eq:bound_1`.
The printed proof uses a matrix norm lower bound to bound every reciprocal
entry. The required assumption is an entrywise lower bound δ>0, supplied
by damping, and the resulting estimate contains δ⁻².
-/

import Transformer.AdaFisher.SectionA_Descent
import Mathlib.NumberTheory.Harmonic.Bounds

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- Positive-time learning-rate schedule in Proposition 3.4;
zero-based step t corresponds to the manuscript's t+1. -/
def stepSize (η : ℝ) (t : ℕ) : ℝ := η / Real.sqrt (t + 1 : ℕ)

/-- Squared Euclidean norm scales quadratically, Appendix A.2. -/
theorem squaredNorm_scale (r : ℝ) (x : Fin d → ℝ) :
    squaredNorm (fun i => r * x i) = r ^ 2 * squaredNorm x := by
  simp only [squaredNorm, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- An entrywise damping bound, rather than `‖F‖≥1`, controls the inverse,
Proposition 3.4, Appendix A.2, the C1 derivation. -/
theorem precondition_squaredNorm_bound (δ : ℝ) (f grad : Fin d → ℝ)
    (hδ : 0 < δ) (hf : ∀ i, δ ≤ f i) :
    δ ^ 2 * squaredNorm (precondition f grad) ≤ squaredNorm grad := by
  simp only [squaredNorm, precondition, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i hi
  have hfi : 0 < f i := lt_of_lt_of_le hδ (hf i)
  have heq : f i ^ 2 * (grad i / f i) ^ 2 = grad i ^ 2 := by
    field_simp
  calc
    _ ≤ f i ^ 2 * (grad i / f i) ^ 2 := by
      apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
      nlinarith [hf i]
    _ = _ := heq

example : (0 : ℝ) < 1 ∧ (∀ i : Fin 1, 1 ≤ (fun _ => (2 : ℝ)) i) := by norm_num

/-- The schedule squared is η²/t, Proposition 3.4, Appendix A.2. -/
theorem stepSize_sq (η : ℝ) (t : ℕ) : stepSize η t ^ 2 = η ^ 2 / (t + 1 : ℕ) := by
  rw [stepSize, div_pow, Real.sq_sqrt (by positivity)]

/-- Harmonic sum estimate used in Appendix A.2's C1 term. -/
theorem harmonic_real_bound (T : ℕ) :
    (∑ t ∈ Finset.range T, (1 : ℝ) / (t + 1 : ℕ)) ≤ 1 + Real.log T := by
  simpa [harmonic, Rat.cast_sum, Rat.cast_inv, Rat.cast_natCast, one_div] using
    harmonic_le_one_add_log T

/-- Corrected C1 estimate, Proposition 3.4, Appendix A.2, `eq:bound_1`.
For Fisher entries at least δ and squared gradient norm at most G²,
the accumulated squared update is at most η²G²δ⁻²(1+log T). The source
omits δ⁻² by asserting an unjustified entrywise lower bound of one. -/
theorem squaredUpdates_bound (η δ G : ℝ) (f g : ℕ → Fin d → ℝ)
    (hδ : 0 < δ) (hf : ∀ t i, δ ≤ f t i)
    (hg : ∀ t, squaredNorm (g t) ≤ G ^ 2) (T : ℕ) :
    (∑ t ∈ Finset.range T,
      squaredNorm (precondition (f t) (fun i => stepSize η t * g t i))) ≤
        η ^ 2 * G ^ 2 / δ ^ 2 * (1 + Real.log T) := by
  have hb (t : ℕ) : squaredNorm (precondition (f t) (g t)) ≤ G ^ 2 / δ ^ 2 := by
    apply (le_div_iff₀ (sq_pos_of_pos hδ)).mpr
    simpa [mul_comm] using (precondition_squaredNorm_bound δ (f t) (g t) hδ (hf t)).trans (hg t)
  have hs (t : ℕ) : squaredNorm (precondition (f t) (fun i => stepSize η t * g t i)) =
      stepSize η t ^ 2 * squaredNorm (precondition (f t) (g t)) := by
    rw [← squaredNorm_scale]
    congr 1
    funext i
    simp only [precondition]
    ring
  calc
    _ ≤ ∑ t ∈ Finset.range T, (η ^ 2 * G ^ 2 / δ ^ 2) * ((1 : ℝ) / (t + 1 : ℕ)) := by
      apply Finset.sum_le_sum
      intro t ht
      rw [hs t, stepSize_sq]
      have h := mul_le_mul_of_nonneg_left (hb t) (sq_nonneg (stepSize η t))
      rw [stepSize_sq] at h
      convert h using 1
      ring
    _ = (η ^ 2 * G ^ 2 / δ ^ 2) * ∑ t ∈ Finset.range T, ((1 : ℝ) / (t + 1 : ℕ)) :=
      (Finset.mul_sum _ _ _).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left (harmonic_real_bound T) (by positivity)

example : (0 : ℝ) < 1 ∧
    (∀ (t : ℕ) (i : Fin 1), 1 ≤ (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i) ∧
    (∀ t : ℕ, squaredNorm ((fun (_ : ℕ) (_ : Fin 1) => (0 : ℝ)) t) ≤ 1 ^ 2) := by
  norm_num [squaredNorm]

end Transformer.AdaFisher
