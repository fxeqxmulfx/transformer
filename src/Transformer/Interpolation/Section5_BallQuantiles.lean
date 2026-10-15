/-
# Ordered quantiles and equal-mass shells on the carrying set

ArXiv:2411.04551v3, §5, `cl: balls`. The original probability is
carried by a small set. Only the intersections of consecutive balls
with that set have to be nested; the ambient balls need not be nested.
This suffices for the actual measure subtraction giving each shell
mass `1/M`. The same full-measure carrier gives the zero and one
endpoint masses used in the intermediate value arguments.

Continuous antitone mass functions attain every required cumulative
mass. Any choices of those quantiles have strictly ordered parameters,
because the prescribed masses are strictly ordered. All formulas below
refer to the original measure and actual sets and functions.
-/

import Transformer.Interpolation.Section5_BallMass

open Real MeasureTheory Metric Set Filter

namespace Transformer.Interpolation

variable {X : Type*} [MeasurableSpace X]

/-- An actual full-measure carrier is an almost-everywhere membership
condition, even when the carrier itself is not assumed measurable.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the original small support. -/
theorem ae_mem_of_null_carrier_compl (ν : Measure X) (S : Set X) (hS : ν Sᶜ = 0) :
    ∀ᵐ x ∂ν, x ∈ S := hS

/-- Nesting on the actual carrier suffices for the actual measure
comparison. Source: arXiv:2411.04551v3, §5, `cl: balls`, monotone ball mass. -/
theorem measure_mono_on_carrier (ν : Measure X) (S : Set X) (hS : ν Sᶜ = 0)
    (A B : Set X) (hAB : ∀ x ∈ S, x ∈ A → x ∈ B) : ν A ≤ ν B := by
  apply measure_mono_ae
  filter_upwards [ae_mem_of_null_carrier_compl ν S hS] with x hx
  exact hAB x hx

/-- The real mass comparison retains finiteness of the actual measure.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the real continuous mass function. -/
theorem measureReal_mono_on_carrier (ν : Measure X) [IsFiniteMeasure ν]
    (S : Set X) (hS : ν Sᶜ = 0) (A B : Set X)
    (hAB : ∀ x ∈ S, x ∈ A → x ∈ B) : ν.real A ≤ ν.real B :=
  ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_on_carrier ν S hS A B hAB)

/-- A set avoiding the actual carrier has zero original mass.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the empty end of the sweep. -/
theorem measure_zero_of_disjoint_carrier (ν : Measure X) (S : Set X) (hS : ν Sᶜ = 0)
    (A : Set X) (hA : ∀ x ∈ S, x ∉ A) : ν A = 0 := by
  apply measure_mono_null _ hS
  exact fun x hx hxs => hA x hxs hx

/-- A set containing the actual carrier has full original probability.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the full end of the sweep. -/
theorem measure_one_of_contains_carrier (ν : Measure X) [IsProbabilityMeasure ν]
    (S : Set X) (hS : ν Sᶜ = 0) (A : Set X) (hA : S ⊆ A) : ν A = 1 := by
  calc
    _ = ν univ := measure_congr (by
      filter_upwards [ae_mem_of_null_carrier_compl ν S hS] with x hx
      exact propext (iff_true_intro (hA hx)))
    _ = 1 := measure_univ

/-- Consecutive cumulative masses subtract even when the actual
ambient sets are nested only on the carrying set. Source:
arXiv:2411.04551v3, §5, property 1 of `cl: balls`. -/
theorem measure_sdiff_of_nested_on_carrier (ν : Measure X) [IsFiniteMeasure ν]
    (S : Set X) (hS : ν Sᶜ = 0) (A B : Set X) (hB : MeasurableSet B)
    (hBA : ∀ x ∈ S, x ∈ B → x ∈ A) : ν (A \ B) = ν A - ν B := by
  rw [measure_sdiff' A hB.nullMeasurableSet (measure_ne_top _ _)]
  congr 1
  apply measure_congr
  filter_upwards [ae_mem_of_null_carrier_compl ν S hS] with x hx
  exact propext (or_iff_left_of_imp (hBA x hx))

/-- Successive prescribed cumulative masses give the actual shell
mass `1/M`. Source: arXiv:2411.04551v3, §5, property 1 of `cl: balls`.
The subtraction uses nesting only on the original full-measure carrier. -/
theorem measure_shell_eq_inv_nat (ν : Measure X) [IsFiniteMeasure ν]
    (S : Set X) (hS : ν Sᶜ = 0) (A B : Set X) (hB : MeasurableSet B)
    (hBA : ∀ x ∈ S, x ∈ B → x ∈ A) (M m : ℕ)
    (hA : ν A = ENNReal.ofReal (((m : ℝ) + 1) / M))
    (hBm : ν B = ENNReal.ofReal ((m : ℝ) / M)) :
    ν (A \ B) = ENNReal.ofReal ((M : ℝ)⁻¹) := by
  rw [measure_sdiff_of_nested_on_carrier ν S hS A B hB hBA, hA, hBm,
    ← ENNReal.ofReal_sub _ (by positivity : 0 ≤ (m : ℝ) / M)]
  congr 1
  ring

variable [MetricSpace X] [BorelSpace X]

/-- A prescribed original ball mass is attained between two actual
radius bounds. Source: arXiv:2411.04551v3, §5, `cl: balls`,
choosing the common radius while retaining its geometric bounds. -/
theorem exists_ball_mass_between (ν : Measure X) [IsFiniteMeasure ν]
    (y : X) (hboundary : ∀ r : ℝ, ν (sphere y r) = 0)
    {r₀ r₁ q : ℝ} (hr : r₀ ≤ r₁)
    (hq₀ : ν.real (ball y r₀) ≤ q) (hq₁ : q ≤ ν.real (ball y r₁)) :
    ∃ r ∈ Icc r₀ r₁, ν (ball y r) = ENNReal.ofReal q := by
  have hc : Continuous (fun r : ℝ => ν.real (ball y r)) :=
    continuous_iff_continuousAt.mpr fun r =>
      (continuousAt_ball_mass ν y r (hboundary r)).comp
        (continuous_const.prodMk continuous_id).continuousAt
  obtain ⟨r, hr, heq⟩ := intermediate_value_Icc hr hc.continuousOn ⟨hq₀, hq₁⟩
  refine ⟨r, hr, ?_⟩
  change ν.real (ball y r) = q at heq
  rw [← heq, measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]

/-- All prescribed cumulative masses of the source are actual
quantiles, and their selected parameters are strictly decreasing.
Source: arXiv:2411.04551v3, §5, `cl: balls`, choosing `sₘⁱ`.
The count is positive, including the single-ball case. -/
theorem exists_strictly_ordered_mass_quantiles (M : ℕ) (hM : 0 < M)
    {a : ℝ} (ha : 0 ≤ a) (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 a))
    (hmono : AntitoneOn f (Icc 0 a)) (h0 : f 0 = 1) (haM : f a = (M : ℝ)⁻¹) :
    ∃ t : Fin M → ℝ, (∀ m, t m ∈ Icc 0 a) ∧
      (∀ m, f (t m) = ((m : ℕ) + 1 : ℝ) / M) ∧ StrictAnti t := by
  have hMr : (0 : ℝ) < M := by exact_mod_cast hM
  have hex (m : Fin M) : ∃ t ∈ Icc (0 : ℝ) a, f t = ((m : ℕ) + 1 : ℝ) / M := by
    apply intermediate_value_Icc' ha hf
    rw [h0, haM, ← one_div]
    constructor
    · exact (div_le_div_iff_of_pos_right hMr).mpr
        (by linarith [show (0 : ℝ) ≤ m.val from Nat.cast_nonneg _])
    · apply (div_le_one hMr).mpr
      exact_mod_cast (Nat.succ_le_iff.mpr m.isLt)
  choose t ht heq using hex
  refine ⟨t, ht, heq, fun i j hij => ?_⟩
  have hm : f (t i) < f (t j) := by
    rw [heq, heq]
    apply (div_lt_div_iff_of_pos_right hMr).mpr
    exact_mod_cast (show i.val + 1 < j.val + 1 from Nat.succ_lt_succ hij)
  by_contra hnot
  exact (not_lt_of_ge (hmono (ht i) (ht j) (le_of_not_gt hnot))) hm

/-- A genuine probability and carrier realize the actual set
comparison and shell-subtraction inputs at once. Source:
arXiv:2411.04551v3, §5, `cl: balls`, full-measure carrier arithmetic. -/
example : let ν := Measure.dirac (0 : ℝ)
    ν (univ : Set ℝ)ᶜ = 0 ∧ MeasurableSet (∅ : Set ℝ) ∧
      (∀ x ∈ (univ : Set ℝ), x ∈ (∅ : Set ℝ) → x ∈ univ) ∧
      (∀ x ∈ (univ : Set ℝ), x ∉ (∅ : Set ℝ)) ∧
      ν univ = ENNReal.ofReal (((0 : ℕ) : ℝ) + 1) ∧
      ν ∅ = ENNReal.ofReal (((0 : ℕ) : ℝ) / (1 : ℕ)) := by simp

/-- Two genuinely distinct quantile parameters satisfy all the
continuity, monotonicity and endpoint inputs. Source:
arXiv:2411.04551v3, §5, the nonconstant mass sweep in `cl: balls`. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    ContinuousOn (fun t : ℝ => 1 - t / 2) (Icc 0 1) ∧
    AntitoneOn (fun t : ℝ => 1 - t / 2) (Icc 0 1) ∧
    (fun t : ℝ => 1 - t / 2) 0 = 1 ∧
    (fun t : ℝ => 1 - t / 2) 1 = (2 : ℝ)⁻¹ := by
  refine ⟨by decide, zero_le_one, by fun_prop, ?_, by norm_num, by norm_num⟩
  intro x hx y hy hxy
  linarith

end Transformer.Interpolation
