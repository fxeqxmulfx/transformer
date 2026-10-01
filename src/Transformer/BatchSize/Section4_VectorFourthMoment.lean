/-
# Dimension-explicit fourth moment of vector Brownian sums

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
No independence of the adapted coordinate amplitudes is required.
-/

import Transformer.BatchSize.Section4_ItoFourthBound
import Transformer.BatchSize.Section4_VectorItoSums

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Euclidean fourth powers are controlled by coordinate fourth
powers with a dimension factor, used for Section 4.3 (2)--(3). -/
theorem euclidean_norm_four_le {d : ℕ} (x : EucSpace d) :
    ‖x‖ ^ 4 ≤ (d : ℝ) * ∑ k : Fin d, x k ^ 4 := by
  rw [show ‖x‖ ^ 4 = (‖x‖ ^ 2) ^ 2 by ring, EuclideanSpace.real_norm_sq_eq]
  have h := Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul (R := ℝ) Finset.univ
    (r := fun k : Fin d => x k ^ 2) (f := fun _ => (1 : ℝ)) (g := fun k => x k ^ 4)
    (fun _ _ => by norm_num) (fun _ _ => by positivity) (fun _ _ => by nlinarith)
  simpa only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] using h

/-- Bounded diagonal amplitudes give finite fourth moment for the
actual vector stochastic sum, Section 4.3 (2)--(3). -/
theorem vectorItoSum_memLp_four {d : ℕ} (t : ℕ → ℝ≥0)
    (H : ℕ → BrownianSample d → EucSpace d)
    (hH : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (H j))
    (A : ℝ≥0) (hbound : ∀ j ω k, |H j ω k| ≤ A) (n : ℕ) :
    MemLp (vectorItoSum t H n) 4 (brownianNoiseLaw d) := by
  apply MemLp.of_eval_piLp
  intro k
  exact brownianItoSum_memLp_four k t (fun j ω => H j ω k)
    (fun j => (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable (hH j))
    A (fun j ω => hbound j ω k) n

/-- The vector stochastic sum has fourth moment at most
3 d^2 A^4 times elapsed time squared, uniformly over grid resolutions,
Section 4.3 (2)--(3). -/
theorem vectorItoSum_fourthMoment_bound {d : ℕ} (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (H : ℕ → BrownianSample d → EucSpace d)
    (hH : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (H j))
    (A : ℝ≥0) (hbound : ∀ j ω k, |H j ω k| ≤ A) (n : ℕ) :
    (∫ ω, ‖vectorItoSum t H n ω‖ ^ 4 ∂brownianNoiseLaw d) ≤
      3 * (d : ℝ) ^ 2 * A ^ 4 * ((t n : ℝ) - t 0) ^ 2 := by
  let V := vectorItoSum t H n
  have hVL4 := vectorItoSum_memLp_four t H hH A hbound n
  have hcoord k := (hVL4.eval_piLp k)
  have hint k : Integrable (fun ω => V ω k ^ 4) (brownianNoiseLaw d) :=
    memLp_four_integrable_pow (brownianNoiseLaw d) _ (hcoord k) 4 le_rfl
  have hpoint (ω : BrownianSample d) : ‖V ω‖ ^ 4 ≤ (d : ℝ) * ∑ k : Fin d, V ω k ^ 4 :=
    euclidean_norm_four_le (V ω)
  have h := integral_mono (hVL4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
    ((integrable_finsetSum _ (fun k _ => hint k)).const_mul (d : ℝ)) hpoint
  rw [integral_const_mul, integral_finsetSum _ (fun k _ => hint k)] at h
  calc
    _ ≤ (d : ℝ) * ∑ k : Fin d, ∫ ω, V ω k ^ 4 ∂brownianNoiseLaw d := h
    _ ≤ (d : ℝ) * ∑ _ : Fin d, 3 * (A : ℝ) ^ 4 * ((t n : ℝ) - t 0) ^ 2 := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      apply Finset.sum_le_sum
      intro k hk
      exact brownianItoSum_fourthMoment_bound k t hmono (fun j ω => H j ω k)
        (fun j => (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable (hH j))
        A (fun j ω => hbound j ω k) n
    _ = _ := by simp; ring

/-- Joint nonvacuity of vector fourth-moment hypotheses, Section 4.3:
unit diagonal amplitudes in two dimensions on unit grid times. -/
example : Monotone (fun j : ℕ => (j : ℝ≥0)) ∧
    (∀ j : ℕ, StronglyMeasurable[brownianFiltration 2 j]
      ((fun _ : ℕ => fun _ : BrownianSample 2 => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) j)) ∧
    (∀ (j : ℕ) (ω : BrownianSample 2) (k : Fin 2),
      |(fun _ : ℕ => fun _ : BrownianSample 2 => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) j ω k| ≤
        (1 : ℝ≥0)) :=
  ⟨fun i j hij => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast hij,
    fun _ => stronglyMeasurable_const, by simp⟩

end Transformer.BatchSize
