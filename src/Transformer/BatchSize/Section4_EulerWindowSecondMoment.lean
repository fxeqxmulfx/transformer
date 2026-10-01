/-
# Mean-square bounds on arbitrary Brownian Euler path windows

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The bounded drift and exact adapted vector Ito isometry control even
partial grid intervals, uniformly in the grid resolution.
-/

import Transformer.BatchSize.Section4_EulerIncrementMoments

open MeasureTheory Filter
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Arbitrary actual Euler path increments have finite second moment
and the bound 2 M^2 duration^2 + 2 d A^2 duration,
Section 4.3 (2)--(3). This also controls grid states adjacent to a
fixed observation time in the martingale-limit argument. -/
theorem eulerPathValue_increment_meanSquare {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : NNReal)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → NNReal) (hmono : Monotone t) (x₀ : EucSpace d)
    (s v : NNReal) (hsv : s ≤ v) (n : ℕ) :
    MemLp (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) 2
      (brownianNoiseLaw d) ∧
    (∫ ω, ‖eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s‖ ^ 2 ∂brownianNoiseLaw d) ≤
      2 * M ^ 2 * ((v : ℝ) - s) ^ 2 + 2 * (d : ℝ) * A ^ 2 * ((v : ℝ) - s) := by
  let q := windowGrid s v t
  let D := windowEulerDrift b a t x₀ s v n
  let H := windowEulerNoise a (eulerChain b a t x₀) t v
  let N := vectorItoSum q H n
  have hq := windowGrid_monotone s v t hmono
  have hH (j : ℕ) : StronglyMeasurable[brownianFiltration d (q j)] (H j) :=
    windowEulerNoise_adapted a ha _ t (eulerChain_adapted b a hb ha t hmono x₀) s v j
  have hHA (j : ℕ) (ω : BrownianSample d) (k : Fin d) : |H j ω k| ≤ A := by
    dsimp only [H, windowEulerNoise]
    split_ifs
    · exact haA _ k
    · simpa only [PiLp.zero_apply, abs_zero] using A.coe_nonneg
  have hH2 (j : ℕ) : MemLp (H j) 2 (brownianNoiseLaw d) := by
    apply MemLp.of_eval_piLp
    intro k
    exact MemLp.of_bound
      ((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable
        ((hH j).mono ((brownianFiltration d).le (q j)))).aestronglyMeasurable A
      (Eventually.of_forall fun ω => by simpa only [Real.norm_eq_abs] using hHA j ω k)
  have hDm : StronglyMeasurable D := by
    apply Finset.stronglyMeasurable_fun_sum
    intro j hj
    exact ((hb.comp_stronglyMeasurable (eulerChain_adapted b a hb ha t hmono x₀ j)).mono
      ((brownianFiltration d).le (t j))).const_smul ((q (j + 1) : ℝ) - q j)
  have hD2 : MemLp D 2 (brownianNoiseLaw d) := MemLp.of_bound hDm.aestronglyMeasurable
    (M * ((v : ℝ) - s)) (Eventually.of_forall (windowEulerDrift_norm_bound b a M hbM t hmono x₀ s v hsv n))
  have hN2 := vectorItoSum_memLp q hq H hH hH2 n
  have heq (ω : BrownianSample d) :
      eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s = D ω + N ω :=
    eulerPathValue_sub_eq_window b a t hmono x₀ s v hsv n ω
  constructor
  · rw [show (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) =
        D + N from funext heq]
    exact hD2.add hN2
  have hDb : (∫ ω, ‖D ω‖ ^ 2 ∂brownianNoiseLaw d) ≤ M ^ 2 * ((v : ℝ) - s) ^ 2 := by
    have h := integral_mono (hD2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (integrable_const ((M : ℝ) ^ 2 * ((v : ℝ) - s) ^ 2)) fun ω => by
        simpa only [mul_pow] using pow_le_pow_left₀ (norm_nonneg _)
          (windowEulerDrift_norm_bound b a M hbM t hmono x₀ s v hsv n ω) 2
    simpa using h
  have hHb (j : ℕ) : (∫ ω, ‖H j ω‖ ^ 2 ∂brownianNoiseLaw d) ≤ (d : ℝ) * A ^ 2 := by
    have hpoint (ω : BrownianSample d) : ‖H j ω‖ ^ 2 ≤ (d : ℝ) * A ^ 2 := by
      rw [EuclideanSpace.real_norm_sq_eq]
      calc
        _ ≤ ∑ _ : Fin d, (A : ℝ) ^ 2 := Finset.sum_le_sum fun k _ => by
          simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hHA j ω k) 2
        _ = _ := by simp
    have h := integral_mono ((hH2 j).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (integrable_const ((d : ℝ) * A ^ 2)) hpoint
    simpa using h
  have hNb : (∫ ω, ‖N ω‖ ^ 2 ∂brownianNoiseLaw d) ≤ (d : ℝ) * A ^ 2 * ((v : ℝ) - s) := by
    rw [vectorItoSum_isometry q hq H hH hH2]
    calc
      _ ≤ ∑ j ∈ Finset.range n, ((d : ℝ) * A ^ 2) * ((q (j + 1) : ℝ) - q j) :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hHb j)
          (sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hq (Nat.le_succ j))))
      _ = ((d : ℝ) * A ^ 2) * ((q n : ℝ) - q 0) := by
        rw [← Finset.mul_sum, Finset.sum_range_sub (fun j => (q j : ℝ)) n]
      _ ≤ _ := mul_le_mul_of_nonneg_left
        (sub_le_sub (NNReal.coe_le_coe.mpr (max_le hsv (min_le_left _ _)))
          (NNReal.coe_le_coe.mpr (le_max_left _ _))) (by positivity)
  simp_rw [heq]
  have h := integral_mono ((hD2.add hN2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
    (((hD2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2).add
      ((hN2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2))
    (fun ω => by
      simp only [Pi.add_apply]
      simpa only [one_smul, one_add_one_eq_two, one_mul] using
        norm_add_time_smul_sq_le (D ω) (N ω) 1 (by norm_num))
  simp only [Pi.add_apply] at h
  rw [integral_add ((hD2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2)
    ((hN2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2), integral_const_mul,
    integral_const_mul] at h
  have hout := h.trans (add_le_add
    (mul_le_mul_of_nonneg_left hDb (by norm_num : (0 : ℝ) ≤ 2))
    (mul_le_mul_of_nonneg_left hNb (by norm_num : (0 : ℝ) ≤ 2)))
  simpa only [mul_assoc] using hout

/-- The actual Euler path second-moment bound for times in either
order, Section 4.3 (2)--(3). -/
theorem eulerPathValue_increment_meanSquare_abs {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : NNReal)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → NNReal) (hmono : Monotone t) (x₀ : EucSpace d) (s v : NNReal) (n : ℕ) :
    MemLp (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) 2
      (brownianNoiseLaw d) ∧
    (∫ ω, ‖eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s‖ ^ 2 ∂brownianNoiseLaw d) ≤
      2 * M ^ 2 * |(v : ℝ) - s| ^ 2 + 2 * (d : ℝ) * A ^ 2 * |(v : ℝ) - s| := by
  rcases le_total s v with hsv | hvs
  · simpa only [abs_of_nonneg (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hsv))] using
      eulerPathValue_increment_meanSquare b a hb ha M A hbM haA t hmono x₀ s v hsv n
  · have h := eulerPathValue_increment_meanSquare b a hb ha M A hbM haA t hmono x₀ v s hvs n
    have hm : MemLp (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) 2
        (brownianNoiseLaw d) := by
      convert h.1.neg using 1
      funext ω
      exact (neg_sub _ _).symm
    refine ⟨hm, ?_⟩
    simpa only [norm_sub_rev, abs_of_nonpos (sub_nonpos.mpr (NNReal.coe_le_coe.mpr hvs)),
      neg_sub] using h.2

/-- Joint nonvacuity of the arbitrary-window second-moment hypotheses,
Section 4.3: bounded constant coefficients and a positive unit window. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    ‖(0 : EucSpace 1)‖ ≤ (1 : NNReal) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧
    Monotone (fun n : ℕ => (n : NNReal)) ∧ (1 : NNReal) ≤ 2 :=
  ⟨continuous_const, fun _ => continuous_const, by simp, (fun k => by fin_cases k; norm_num),
    fun i j hij => by change (i : NNReal) ≤ (j : NNReal); exact_mod_cast hij, by norm_num⟩

end Transformer.BatchSize
