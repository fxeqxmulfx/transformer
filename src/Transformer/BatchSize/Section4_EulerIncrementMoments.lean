/-
# Fourth moments of arbitrary Brownian Euler path increments

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The estimate is independent of grid resolution and remains valid inside
grid intervals. The adapted stochastic sum uses clamped window times.
-/

import Transformer.BatchSize.Section4_EulerWindow
import Transformer.BatchSize.Section4_EulerFourthMoment

open MeasureTheory Filter
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Bounded Euler drift accumulates at most M times the window length,
Section 4.3 (2)--(3). -/
theorem windowEulerDrift_norm_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (M : ℝ≥0) (hbM : ∀ x, ‖b x‖ ≤ M)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d)
    (s v : ℝ≥0) (hsv : s ≤ v) (n : ℕ) (ω : BrownianSample d) :
    ‖windowEulerDrift b a t x₀ s v n ω‖ ≤ M * ((v : ℝ) - s) := by
  let q := windowGrid s v t
  have hq := windowGrid_monotone s v t hmono
  have hduration : (q n : ℝ) - q 0 ≤ (v : ℝ) - s := by
    have hv : q n ≤ v := max_le hsv (min_le_left _ _)
    have hs : s ≤ q 0 := le_max_left _ _
    exact sub_le_sub (NNReal.coe_le_coe.mpr hv) (NNReal.coe_le_coe.mpr hs)
  calc
    _ ≤ ∑ j ∈ Finset.range n, ‖((q (j + 1) : ℝ) - q j) • b (eulerChain b a t x₀ j ω)‖ :=
      norm_sum_le _ _
    _ ≤ ∑ j ∈ Finset.range n, ((q (j + 1) : ℝ) - q j) * M := by
      apply Finset.sum_le_sum
      intro j hj
      have hδ : 0 ≤ (q (j + 1) : ℝ) - q j :=
        sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hq (Nat.le_succ j)))
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hδ]
      exact mul_le_mul_of_nonneg_left (hbM _) hδ
    _ = M * ((q n : ℝ) - q 0) := by
      rw [← Finset.sum_mul, Finset.sum_range_sub (fun j => (q j : ℝ)) n]
      ring
    _ ≤ _ := mul_le_mul_of_nonneg_left hduration M.coe_nonneg

/-- Arbitrary Euler path increments have finite fourth moment and
the grid-independent bound 8 M^4 duration^4 + 24 d^2 A^4 duration^2,
Section 4.3 (2)--(3). This does not assume independence of random
coefficients from each other; only future Brownian increments are used. -/
theorem eulerPathValue_increment_fourthMoment {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : ℝ≥0)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d)
    (s v : ℝ≥0) (hsv : s ≤ v) (n : ℕ) :
    MemLp (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) 4
      (brownianNoiseLaw d) ∧
    (∫ ω, ‖eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s‖ ^ 4 ∂brownianNoiseLaw d) ≤
      8 * M ^ 4 * ((v : ℝ) - s) ^ 4 + 24 * (d : ℝ) ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2 := by
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
  have hDm : StronglyMeasurable D := by
    apply Finset.stronglyMeasurable_fun_sum
    intro j hj
    exact ((hb.comp_stronglyMeasurable (eulerChain_adapted b a hb ha t hmono x₀ j)).mono
      ((brownianFiltration d).le (t j))).const_smul ((q (j + 1) : ℝ) - q j)
  have hD4 : MemLp D 4 (brownianNoiseLaw d) := MemLp.of_bound hDm.aestronglyMeasurable
    (M * ((v : ℝ) - s)) (Eventually.of_forall (windowEulerDrift_norm_bound b a M hbM t hmono x₀ s v hsv n))
  have hN4 := vectorItoSum_memLp_four q H hH A hHA n
  have heq (ω : BrownianSample d) :
      eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s = D ω + N ω :=
    eulerPathValue_sub_eq_window b a t hmono x₀ s v hsv n ω
  constructor
  · rw [show (fun ω => eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s) =
        D + N from funext heq]
    exact hD4.add hN4
  have hDb : (∫ ω, ‖D ω‖ ^ 4 ∂brownianNoiseLaw d) ≤ M ^ 4 * ((v : ℝ) - s) ^ 4 := by
    have h := integral_mono (hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
      (integrable_const ((M : ℝ) ^ 4 * ((v : ℝ) - s) ^ 4)) fun ω => by
        simpa only [mul_pow] using pow_le_pow_left₀ (norm_nonneg _)
          (windowEulerDrift_norm_bound b a M hbM t hmono x₀ s v hsv n ω) 4
    simpa using h
  have hNb : (∫ ω, ‖N ω‖ ^ 4 ∂brownianNoiseLaw d) ≤
      3 * (d : ℝ) ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2 := by
    have hv : q n ≤ v := max_le hsv (min_le_left _ _)
    have hs : s ≤ q 0 := le_max_left _ _
    have hδ : 0 ≤ (q n : ℝ) - q 0 :=
      sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hq (Nat.zero_le n)))
    have hduration : (q n : ℝ) - q 0 ≤ (v : ℝ) - s :=
      sub_le_sub (NNReal.coe_le_coe.mpr hv) (NNReal.coe_le_coe.mpr hs)
    exact (vectorItoSum_fourthMoment_bound q hq H hH A hHA n).trans (by gcongr)
  simp_rw [heq]
  have h := integral_mono ((hD4.add hN4).integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
    (((hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0)).add
      (hN4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))).const_mul 8)
    (fun ω => norm_add_four_le (D ω) (N ω))
  rw [integral_const_mul] at h
  simp only [Pi.add_apply] at h
  rw [integral_add (hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
    (hN4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))] at h
  calc
    _ ≤ 8 * ((∫ ω, ‖D ω‖ ^ 4 ∂brownianNoiseLaw d) + (∫ ω, ‖N ω‖ ^ 4 ∂brownianNoiseLaw d)) := h
    _ ≤ 8 * (M ^ 4 * ((v : ℝ) - s) ^ 4 + 3 * (d : ℝ) ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2) := by gcongr
    _ = _ := by ring

/-- Joint nonvacuity of increment-moment hypotheses, Section 4.3:
bounded constant coefficients, unit times, and a positive window. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, Continuous (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) ∧ Monotone (fun n : ℕ => (n : ℝ≥0)) ∧ (1 : ℝ≥0) ≤ 2 :=
  ⟨continuous_const, fun _ => continuous_const, by simp, by simp,
    fun i j hij => by change (i : ℝ≥0) ≤ j; exact_mod_cast hij, by norm_num⟩

end Transformer.BatchSize
