/-
# Euler interpolation restricted to a time window

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Clamping each grid time to [s,v] expresses a path increment as a bounded
drift sum and a predictable Brownian sum on exactly that time window.
-/

import Transformer.BatchSize.Section4_EulerClipping

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The original Euler grid clamped to a time window,
Section 4.3 (2)--(3). -/
def windowGrid (s v : ℝ≥0) (t : ℕ → ℝ≥0) (j : ℕ) : ℝ≥0 := max s (min v (t j))

/-- Window clamping preserves the order of an Euler grid,
Section 4.3 (2)--(3). -/
theorem windowGrid_monotone (s v : ℝ≥0) (t : ℕ → ℝ≥0) (hmono : Monotone t) :
    Monotone (windowGrid s v t) :=
  fun _ _ hij => max_le_max le_rfl (min_le_min le_rfl (hmono hij))

/-- Differences of two stopped evaluations equal a window evaluation,
Section 4.3 (2)--(3). This applies both to time and to Brownian coordinates. -/
theorem stopped_difference_eq_window {E : Type*} [AddCommGroup E]
    (F : ℝ≥0 → E) (s v : ℝ≥0) (hsv : s ≤ v) (r : ℝ≥0) :
    F (min v r) - F (min s r) = F (max s (min v r)) - F s := by
  by_cases hrs : r ≤ s
  · simp only [min_eq_right hrs, min_eq_right (hrs.trans hsv), max_eq_left hrs, sub_self]
  · have hsr := le_of_not_ge hrs
    simp only [min_eq_left hsr, max_eq_right (le_min hsv hsr)]

/-- The actual Euler drift accumulated inside a time window,
Section 4.3 (2)--(3). -/
def windowEulerDrift {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d)
    (s v : ℝ≥0) (n : ℕ) (ω : BrownianSample d) : EucSpace d :=
  ∑ j ∈ Finset.range n,
    ((windowGrid s v t (j + 1) : ℝ) - windowGrid s v t j) • b (eulerChain b a t x₀ j ω)

/-- Predictable Euler noise coefficients on a window, Section 4.3
(2)--(3). Coefficients from future grid states are zeroed on intervals
whose clamped length is zero. -/
def windowEulerNoise {d : ℕ} (a : EucSpace d → Fin d → ℝ)
    (X : ℕ → BrownianSample d → EucSpace d) (t : ℕ → ℝ≥0) (v : ℝ≥0)
    (j : ℕ) (ω : BrownianSample d) : EucSpace d :=
  if t j ≤ v then WithLp.toLp 2 (a (X j ω)) else 0

/-- Window noise amplitudes are measurable at the actual clamped
left endpoints, Section 4.3 (2)--(3). -/
theorem windowEulerNoise_adapted {d : ℕ} (a : EucSpace d → Fin d → ℝ)
    (ha : ∀ k, Continuous (fun x => a x k)) (X : ℕ → BrownianSample d → EucSpace d)
    (t : ℕ → ℝ≥0) (hX : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (X j))
    (s v : ℝ≥0) (j : ℕ) :
    StronglyMeasurable[brownianFiltration d (windowGrid s v t j)] (windowEulerNoise a X t v j) := by
  by_cases hj : t j ≤ v
  · have hleft : t j ≤ windowGrid s v t j := by
      simp only [windowGrid, min_eq_right hj]
      exact le_max_right _ _
    have hstate := (hX j).mono ((brownianFiltration d).mono hleft)
    have heq : windowEulerNoise a X t v j = fun ω => WithLp.toLp 2 (a (X j ω)) := by
      funext ω
      exact ite_eq_left hj
    rw [heq]
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d (windowGrid s v t j)
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    exact Measurable.of_eval fun k => ((ha k).comp_stronglyMeasurable hstate).measurable
  · have heq : windowEulerNoise a X t v j = fun _ => 0 := by
      funext ω
      exact ite_eq_right hj
    rw [heq]
    exact stronglyMeasurable_const

/-- A Brownian Euler path increment is exactly the drift and
predictable stochastic sum on its time window, Section 4.3 (2)--(3). -/
theorem eulerPathValue_sub_eq_window {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (s v : ℝ≥0) (hsv : s ≤ v) (n : ℕ) (ω : BrownianSample d) :
    eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s =
      windowEulerDrift b a t x₀ s v n ω +
        vectorItoSum (windowGrid s v t) (windowEulerNoise a (eulerChain b a t x₀) t v) n ω := by
  let L (u : ℝ≥0) (j : ℕ) : EucSpace d :=
    (((min u (t (j + 1)) : ℝ≥0) : ℝ) - min u (t j)) • b (eulerChain b a t x₀ j ω) +
      WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ j ω) k *
        brownianIncrement k (min u (t j)) (min u (t (j + 1))) ω)
  have hterm (j : ℕ) : L v j - L s j =
      ((windowGrid s v t (j + 1) : ℝ) - windowGrid s v t j) • b (eulerChain b a t x₀ j ω) +
        WithLp.toLp 2 (fun k => windowEulerNoise a (eulerChain b a t x₀) t v j ω k *
          brownianIncrement k (windowGrid s v t j) (windowGrid s v t (j + 1)) ω) := by
    ext k
    have ht0 := stopped_difference_eq_window (fun r : ℝ≥0 => (r : ℝ)) s v hsv (t j)
    have ht1 := stopped_difference_eq_window (fun r : ℝ≥0 => (r : ℝ)) s v hsv (t (j + 1))
    have hB0 := stopped_difference_eq_window (fun r => coordinateBrownian k r ω) s v hsv (t j)
    have hB1 := stopped_difference_eq_window (fun r => coordinateBrownian k r ω) s v hsv (t (j + 1))
    by_cases hj : t j ≤ v
    · simp only [L, windowEulerNoise, ite_eq_left hj, windowGrid, PiLp.sub_apply,
        PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, brownianIncrement]
      linear_combination b (eulerChain b a t x₀ j ω) k * (ht1 - ht0) +
        a (eulerChain b a t x₀ j ω) k * (hB1 - hB0)
    · have hvj : v ≤ t j := le_of_not_ge hj
      have hvnext := hvj.trans (hmono (Nat.le_succ j))
      simp only [L, windowEulerNoise, ite_eq_right hj, windowGrid, min_eq_left hvj,
        min_eq_left hvnext, min_eq_left (hsv.trans hvj), min_eq_left (hsv.trans hvnext),
        max_eq_right hsv, sub_self, zero_smul, brownianIncrement, mul_zero,
        PiLp.add_apply]
      simp
  have hsum : eulerPathValue b a t x₀ n ω v - eulerPathValue b a t x₀ n ω s =
      ∑ j ∈ Finset.range n, (L v j - L s j) := by
    simp only [eulerPathValue, Real.toNNReal_coe, L, Finset.sum_sub_distrib]
    abel
  rw [hsum]
  simp_rw [hterm]
  rw [Finset.sum_add_distrib]
  congr 1
  ext k
  exact map_sum (PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin d => ℝ) k) _ _

/-- Joint nonvacuity of window hypotheses, Section 4.3:
a positive window and a nonconstant adapted Brownian state. -/
example : Monotone (fun j : ℕ => (j : ℝ≥0)) ∧ (1 : ℝ≥0) ≤ 2 ∧
    (∀ _ : Fin 1, Continuous (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ j : ℕ, StronglyMeasurable[brownianFiltration 1 j]
      (fun ω => WithLp.toLp 2 (fun _ : Fin 1 => coordinateBrownian (0 : Fin 1) j ω))) := by
  refine ⟨fun i j hij => by change (i : ℝ≥0) ≤ j; exact_mod_cast hij,
    by norm_num, fun _ => continuous_const, ?_⟩
  intro j
  let : MeasurableSpace (BrownianSample 1) := brownianFiltration 1 j
  apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 1 => ℝ)).symm.continuous.comp_stronglyMeasurable
  apply Measurable.stronglyMeasurable
  exact Measurable.of_eval fun _ => ((coordinateBrownian_filtered (0 : Fin 1)).stronglyAdapted j).measurable

end Transformer.BatchSize
