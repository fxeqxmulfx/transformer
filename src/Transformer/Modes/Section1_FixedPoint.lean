import Transformer.Modes.Section2_Gt
import Mathlib.Analysis.Analytic.Order
import Mathlib.MeasureTheory.Topology

/-
# The number of modes of a Gaussian KDE — a fixed point is almost surely not a mode

The first bullet of `thm:mammen`, arXiv:2412.09080v3, §1.1, uses the indicator
`0 ∈ [a,b]`. To test its boundary cases one must distinguish a random mode
near zero from a mode exactly at zero. Every fixed observation point has
probability zero of being a critical point, for every positive sample count.

For one sample, `x ↦ G(t,x)` is a nonconstant real analytic function. Each
level set is discrete and is null for the Gaussian measure. Conditioning
on all but one sample transfers this fact to the finite sum in `eq:Fn`.
The relation between `F_n` and the derivative of the KDE then excludes a
local maximum at a prescribed point. No joint density, continuity of a
joint density, or Kac–Rice formula is used.

In particular, the expected mode count in the singleton `[0,0]` is exactly
zero for every positive bandwidth and every positive sample count. It cannot
converge to the indicator value one appearing in the source's first bullet.
The consequences for nondegenerate intervals are considered separately.

Source: arXiv:2412.09080v3, §1.1, `thm:mammen`; §2.1, `eq:Fn`;
§2.2, `eq: Gt`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- The summand `G(t,x)` is real analytic as a function of the sample.
Source: arXiv:2412.09080v3, §2.2, `eq: Gt`. -/
theorem analyticOnNhd_bigG_sample (β t : ℝ) : AnalyticOnNhd ℝ (bigG β t) Set.univ := by
  apply ContDiff.analyticOnNhd
  unfold bigG
  fun_prop

/-- The summand is not equal to a constant on the whole real line.
This holds for all real `β`; the KDE itself will require `β > 0`.
Source: arXiv:2412.09080v3, §2.2, `eq: Gt`. -/
theorem exists_bigG_sample_ne (β t c : ℝ) : ∃ x, bigG β t x ≠ c := by
  by_cases hc : c = 0
  · refine ⟨t - 1, ?_⟩
    rw [hc]
    have hx : t - (t - 1) = 1 := by ring
    simp only [bigG, hx, one_pow, mul_one]
    exact (Real.exp_pos _).ne'
  · refine ⟨t, ?_⟩
    simpa only [bigG, sub_self, mul_zero, ne_eq, eq_comm] using hc

/-- Each level set of a single summand has Gaussian probability zero:
the complement of the zeros of a nonconstant analytic function is codiscrete.
Source: arXiv:2412.09080v3, §2.2, `eq: Gt`. -/
theorem ae_bigG_sample_ne (β t c : ℝ) :
    ∀ᵐ x ∂gaussianReal 0 1, bigG β t x ≠ c := by
  let : NullSingletonClass (gaussianReal 0 1) :=
    nullSingletonClass_gaussianReal (by norm_num)
  obtain ⟨x, hx⟩ := exists_bigG_sample_ne β t c
  have hf := (analyticOnNhd_bigG_sample β t).sub
    (analyticOnNhd_const (v := c))
  have hzero : bigG β t x - c ≠ 0 := sub_ne_zero.mpr hx
  have hcodiscrete := hf.preimage_zero_mem_codiscreteWithin hzero (Set.mem_univ x)
    isConnected_univ
  have hae : ∀ᵐ y ∂(gaussianReal 0 1).restrict Set.univ, bigG β t y - c ≠ 0 :=
    ae_restrict_le_codiscreteWithin (μ := gaussianReal 0 1) MeasurableSet.univ hcodiscrete
  simpa only [Measure.restrict_univ, sub_ne_zero] using hae

/-- Finite sums of the summands are measurable functions of the sample.
Source: arXiv:2412.09080v3, §2.2, the sum following `eq: Gt`. -/
theorem measurable_sum_bigG (n : ℕ) (β t : ℝ) :
    Measurable fun X : Fin n → ℝ => ∑ i, bigG β t (X i) := by
  unfold bigG
  fun_prop

/-- A nonempty sum of independent summands has no atom at any fixed value.
Conditioning on the remaining coordinates reduces this to `ae_bigG_sample_ne`.
Source: arXiv:2412.09080v3, §2.2, the sum following `eq: Gt`. -/
theorem ae_sum_bigG_ne {n : ℕ} (hn : 0 < n) (β t c : ℝ) :
    ∀ᵐ X ∂gaussianSample n, (∑ i, bigG β t (X i)) ≠ c := by
  cases n with
  | zero => omega
  | succ n =>
    have hmeas : MeasurableSet {p : ℝ × (Fin n → ℝ) |
        bigG β t p.1 + ∑ i, bigG β t (p.2 i) ≠ c} := by
      apply MeasurableSet.compl (measurableSet_eq_fun _ measurable_const)
      unfold bigG
      fun_prop
    have hcond : ∀ᵐ X ∂gaussianSample n, ∀ᵐ x ∂gaussianReal 0 1,
        bigG β t x + ∑ i, bigG β t (X i) ≠ c := by
      apply ae_of_all
      intro X
      filter_upwards [ae_bigG_sample_ne β t (c - ∑ i, bigG β t (X i))] with x hx
      intro h
      apply hx
      linarith
    have hprod : ∀ᵐ p ∂(gaussianReal 0 1).prod (gaussianSample n),
        bigG β t p.1 + ∑ i, bigG β t (p.2 i) ≠ c :=
      (Measure.ae_prod_iff_ae_ae hmeas).mpr ((Measure.ae_ae_comm hmeas).mpr hcond)
    have hpres := measurePreserving_piFinSuccAbove
      (fun _ : Fin (n + 1) => gaussianReal 0 1) 0
    have hsample := hpres.quasiMeasurePreserving.ae hprod
    filter_upwards [hsample] with X hX
    change bigG β t (X 0) + ∑ i, bigG β t (X ((0 : Fin (n + 1)).succAbove i)) ≠ c at hX
    rwa [Fin.sum_univ_succAbove (fun i => bigG β t (X i)) 0]

/-- The nonempty-sample hypothesis of the sum lemma has a finite witness. -/
example : 0 < (1 : ℕ) := one_pos

/-- The field `F_n(t)` almost surely does not vanish at any prescribed `t`.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`. -/
theorem ae_fieldF_ne_zero {n : ℕ} (hn : 0 < n) (β t : ℝ) :
    ∀ᵐ X ∂gaussianSample n, fieldF β X t ≠ 0 := by
  filter_upwards [ae_sum_bigG_ne hn β t 0] with X hX
  rw [fieldF_eq_sum_bigG]
  exact mul_ne_zero (one_div_ne_zero (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hn)).ne') hX

/-- One sample satisfies the field's sample-count hypothesis. -/
example : 0 < (1 : ℕ) := one_pos

/-- A fixed point is almost surely not a mode: a differentiable local
maximum would force `F_n(t) = 0` by `eq:Fn`.
Source: arXiv:2412.09080v3, §1.1, `thm:mammen`; §2.1, `eq:Fn`. -/
theorem ae_not_isLocalMax_kde {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) (t : ℝ) :
    ∀ᵐ X ∂gaussianSample n, ¬ IsLocalMax (kde β X) t := by
  filter_upwards [ae_fieldF_ne_zero hn β t] with X hX
  intro hmax
  apply hX
  rw [fieldF_eq hβ hn X t, hmax.deriv_eq_zero, mul_zero]

/-- Positive bandwidth and one sample satisfy both mode hypotheses. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) := ⟨one_pos, one_pos⟩

/-- The expected count in a singleton is zero, rather than the indicator
of whether that singleton contains the population mode.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem expectedModes_singleton_eq_zero {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (hn : 0 < n) (t : ℝ) : expectedModes β n {t} = 0 := by
  rw [expectedModes, ← lintegral_zero (μ := gaussianSample n)]
  apply lintegral_congr_ae
  filter_upwards [ae_not_isLocalMax_kde hβ hn t] with X hX
  have hempty : modeSet (kde β X) {t} = ∅ := by
    apply Set.eq_empty_iff_forall_notMem.mpr
    rintro s ⟨hs, hmax⟩
    exact hX ((Set.mem_singleton_iff.mp hs) ▸ hmax)
  simp only [modeCount, hempty, Set.encard_empty, ENat.toENNReal_zero]

/-- The singleton count's hypotheses hold simultaneously. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) := ⟨one_pos, one_pos⟩

end Transformer.Modes
