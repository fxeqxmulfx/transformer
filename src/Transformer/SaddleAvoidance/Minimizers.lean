/-
# Saddle avoidance — gradient descent finds local minimizers

Let `f` be `C²` on a finite-dimensional real inner product space with an
`L`-Lipschitz gradient, and `0 < η`, `η L < 1`.  Gradient descent started
from a law with a density almost surely converges to no strict saddle
(`ae_not_tendsto_of_isStrictSaddle`).  If every critical point is a local
minimizer or a strict saddle, every limit of almost every orbit is therefore
a local minimizer (`ae_isLocalMin_of_tendsto`).

Whether an orbit has a limit is a separate question.  If `f` satisfies the
Łojasiewicz inequality at each critical point, and almost every orbit has a
limit point up to translations leaving `f` invariant, almost every orbit
converges to a critical point that is not a strict saddle
(`ae_exists_tendsto_not_isStrictSaddle`); if the orbits are bounded and
every critical point is a local minimizer or a strict saddle, to a local
minimizer (`ae_exists_tendsto_isLocalMin`).  These are `thm:no-saddles`,
`thm:local-min` and the first part of `prop:rates` of arXiv:1602.04915, with
the changes their docstrings give.
-/

import Transformer.SaddleAvoidance.Convergence
import Transformer.SaddleAvoidance.GradientDescent

open Filter Metric MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **Gradient descent almost never converges to a strict saddle.**  For `f`
of class `C²` with an `L`-Lipschitz gradient, `0 < η` and `η L < 1`, the orbit
of a start drawn from a law `ν` absolutely continuous with respect to a Haar
measure almost surely converges to no strict saddle.  Corollary
`thm:no-saddles` of arXiv:1602.04915 reads: "Let `C` be the set of saddle
points and assume they are all strict.  If `C` has at most countably
infinite cardinality, then `Pr(lim_k x_k ∈ C) = 0`."  This theorem gives it
without the countability of `C`. -/
theorem ae_not_tendsto_of_isStrictSaddle (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) :
    ∀ᵐ x ∂ν, ∀ z, IsStrictSaddle f z →
      ¬Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z) := by
  rw [ae_iff]
  refine hν (measure_mono_null (fun x hx => ?_) (measure_tendsto_strictSaddle_eq_zero μ hf
    (fun _ => norm_fderiv_le_of_lipschitz ℝ hL) hη hηL))
  rw [Set.mem_ofPred_eq] at hx
  push Not at hx
  exact hx

/-- **Every limit of gradient descent is almost surely a local minimizer**
when every critical point of `f` is a local minimizer or a strict saddle, the
strict saddle property of §1 of arXiv:1602.04915.  `thm:local-min` there
reads: "Assume the same conditions as Theorem `thm:no-saddles` and
`lim_k x_k` exists, then `Pr(lim_k x_k = x*) = 1`, where `x*` is a local
minimizer."  Its proof passes from `Pr(lim_k x_k ∈ C) = 0` to a local
minimizer, which needs every critical point outside the set `C` of saddle
points to be a local minimizer.  The strict saddle property `hcrit` gives
this and replaces the hypotheses of `thm:no-saddles`; no limit is assumed to
exist. -/
theorem ae_isLocalMin_of_tendsto (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E}
    (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f)) (hη : 0 < η)
    (hηL : η * L < 1) (hcrit : ∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν, ∀ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) →
      IsLocalMin f w := by
  filter_upwards [ae_not_tendsto_of_isStrictSaddle μ hν hf hL hη hηL] with x hx w hw
  exact (hcrit w (gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw)).resolve_right
    fun hs => hx w hs hw

/-- **Gradient descent almost surely converges to a critical point that is
not a strict saddle**, if `f` satisfies the Łojasiewicz inequality with an
exponent `0 ≤ a < 1` at each critical point and almost every orbit has a
limit point up to translations in a set `Λ` leaving `f` invariant. -/
theorem ae_exists_tendsto_not_isStrictSaddle (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧
      gradient f w = 0 ∧ ¬IsStrictSaddle f w := by
  filter_upwards [hcl, ae_not_tendsto_of_isStrictSaddle μ hν hf hL hη hηL] with x ⟨z, hz⟩ hx
  obtain ⟨a, ha0, ha1, hza⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
    by_cases h0 : gradient f z = 0
    · exact hloj z h0
    · exact ⟨0, le_rfl, one_pos,
        isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h0 le_rfl⟩
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt (hf.differentiable (by norm_num)) hL hη
    hηL hΛ ha0 ha1 hza hz
  exact ⟨w, hw, gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw, fun hs => hx w hs hw⟩

/-- **`prop:rates` of arXiv:1602.04915, first part, with the Łojasiewicz
inequality added.**  If every critical point of `f` is a local minimizer or a
strict saddle and satisfies the Łojasiewicz inequality with an exponent
`0 ≤ a < 1`, and almost every orbit is bounded, then almost every orbit
converges to a local minimizer.  The source reads: "Assume the same
conditions as Theorem `thm:no-saddles`, and the iterates do not escape to
`∞`, meaning `{x_k}` is a bounded sequence.  Then `lim_k x_k` exists and
`lim_k x_k = x*` for a local minimum `x*`."  Its proof takes the existence of
the limit from Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim. 16,
2005), which the source introduces as the sufficient condition based on the
Łojasiewicz inequality `def:gradineq`; that inequality is added as `hloj`.
The strict saddle property `hcrit` replaces the hypotheses of
`thm:no-saddles`, as in `ae_isLocalMin_of_tendsto`. -/
theorem ae_exists_tendsto_isLocalMin (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E}
    (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f)) (hη : 0 < η)
    (hηL : η * L < 1) (hcrit : ∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hbdd : ∀ᵐ x ∂ν, Bornology.IsBounded (Set.range fun k => (gradientStep f η)^[k] x)) :
    ∀ᵐ x ∂ν, ∃ w, IsLocalMin f w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  have hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({0} : Set E),
      ‖(gradientStep f η)^[k] x + v - z‖ < ε := by
    filter_upwards [hbdd] with x hx
    obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt hx.isCompact_closure (Λ := {0})
      (u := fun k => (gradientStep f η)^[k] x) fun k => ⟨0, Set.mem_singleton 0, by
        rw [add_zero]
        exact subset_closure (Set.mem_range_self k)⟩
    exact ⟨z, hz⟩
  filter_upwards [ae_exists_tendsto_not_isStrictSaddle μ hν hf hL hη hηL (by simp) hloj hcl]
    with x ⟨w, hw, hw0, hws⟩
  exact ⟨w, (hcrit w hw0).resolve_right hws, hw⟩

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line with Lebesgue measure, `L = 1`, `η = 1/2` and `Λ = {0}`: the only
critical point `0` is a minimizer, where the Łojasiewicz inequality holds
with `a = 1/2`, and gradient descent halves every start, so every orbit is
bounded and converges to `0`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧
      (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z) ∧
      (∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) ∧
      (∀ᵐ x ∂(volume : Measure ℝ), ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({0} : Set ℝ),
        ‖(gradientStep f (1 / 2))^[k] x + v - z‖ < ε) ∧
      ∀ᵐ x ∂(volume : Measure ℝ),
        Bornology.IsBounded (Set.range fun k => (gradientStep f (1 / 2))^[k] x) := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have horb : ∀ k x, (gradientStep f (1 / 2))^[k] x = (1 / 2 : ℝ) ^ k * x := fun k x => by
    simp only [f, iterate_gradientStep_half_mul_norm_sq, smul_eq_mul]
    norm_num
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl,
    contDiff_const.mul (contDiff_norm_sq ℝ), ?_, by norm_num, by norm_num, by simp,
    fun z hz => Or.inl ?_, fun z hz => ⟨1 / 2, by norm_num, by norm_num, ?_⟩,
    Eventually.of_forall fun x => ⟨0, fun ε hε => ?_⟩, Eventually.of_forall fun x => ?_⟩
  · rw [hg]
    exact LipschitzWith.id
  · rw [hg, id_eq] at hz
    subst hz
    exact Eventually.of_forall fun y => by
      simp only [f, norm_zero]
      nlinarith [sq_nonneg ‖y‖]
  · rw [hg, id_eq] at hz
    subst hz
    exact isLojasiewiczAt_half_mul_norm_sq
  · have hlim : Tendsto (fun k => (1 / 2 : ℝ) ^ k * x) atTop (𝓝 0) := by
      simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2)
        (by norm_num)).mul_const x
    exact (Metric.tendsto_nhds.1 hlim ε hε).frequently.mono fun k hk =>
      ⟨0, Set.mem_singleton 0, by rw [horb, add_zero, ← dist_eq_norm]; exact hk⟩
  · rw [isBounded_iff_forall_norm_le]
    refine ⟨‖x‖, ?_⟩
    rintro _ ⟨k, rfl⟩
    show ‖(gradientStep f (1 / 2))^[k] x‖ ≤ ‖x‖
    rw [horb, norm_mul, norm_pow, Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    exact mul_le_of_le_one_left (norm_nonneg x) (pow_le_one₀ (by norm_num) (by norm_num))

end Transformer.SaddleAvoidance
