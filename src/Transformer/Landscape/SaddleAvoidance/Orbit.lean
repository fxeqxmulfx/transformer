module

public import Transformer.Landscape.SaddleAvoidance.Sublevel

/-!
# Saddle avoidance — bounds along the orbit

The avoidance and convergence theorems so far ask for a bound on `‖∇²f‖` on all of
`E`, or near a compact sublevel set. A run of gradient descent can check a bound
only along its own orbit `x_k = (gradientStep f η)^[k] x`; this file asks for it
there.

* **One step.**  If `‖∇²f‖ ≤ L` on the segment from `x` to its gradient step, the
  step lowers `f` by at least `η (1 - η L) ‖∇f(x)‖²`
  (`le_sub_of_gradientStep_of_segment`), by the descent lemma on that segment
  (`le_add_inner_add_mul_norm_sq_of_convex`). So along an orbit whose segments
  `[x_k, x_{k+1}]` carry the bound, step `k` lowers `f` by at least
  `η (1 - η L) ‖∇f(x_k)‖²` (`iterate_succ_le_sub_of_segment`), and `f` decreases
  when `0 ≤ η` and `η L ≤ 1` (`antitone_iterate_of_segment`).
* **Avoidance.**  For `0 < η` and `η L < 1`, the starts whose orbit has
  `‖∇²f(x_k)‖ ≤ L` at every `k` and converges to a strict saddle form a null set
  for every Haar measure (`measure_iterate_norm_le_tendsto_strictSaddle_eq_zero`):
  gradient descent maps these starts into themselves, so
  `measure_inter_tendsto_strictSaddle_eq_zero` applies. The bound is asked at the
  points `x_k` only, not on the segments; on the segments it gives the same
  (`measure_segment_tendsto_strictSaddle_eq_zero`), almost surely for a law absolutely
  continuous with respect to a Haar measure (`ae_not_tendsto_strictSaddle_of_segment`).

The gradient-descent corollary in §5.1 of arXiv:1710.07406 assumes a globally
`L`-Lipschitz gradient and a step size `α < 1/L`; here the bound is asked only
along the orbit.
-/

@[expose] public section

open Set Filter MeasureTheory InnerProductSpace
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {f : E → ℝ} {L η : ℝ}

/-- **Sufficient decrease from a bound on the segment of the step.**  If `f` is `C²`
and `‖∇²f‖ ≤ L` on the segment from `x` to `x - η ∇f(x)`, the step lowers `f` by at
least `η (1 - η L) ‖∇f(x)‖²`. -/
theorem le_sub_of_gradientStep_of_segment (hf : ContDiff ℝ 2 f) {x : E}
    (hL : ∀ y ∈ segment ℝ x (gradientStep f η x), ‖fderiv ℝ (gradient f) y‖ ≤ L) :
    f (gradientStep f η x) ≤ f x - η * (1 - η * L) * ‖gradient f x‖ ^ 2 := by
  have h := le_add_inner_add_mul_norm_sq_of_convex hf (convex_segment _ _) hL
    (left_mem_segment ℝ _ _) (right_mem_segment ℝ _ _)
  have hstep : gradientStep f η x - x = -(η • gradient f x) := by simp [gradientStep]
  rw [hstep, inner_neg_right, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_neg,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at h
  linarith

/-- **The decrease along an orbit whose segments carry the bound.**  If
`‖∇²f‖ ≤ L` on every segment `[x_k, x_{k+1}]` of the orbit of `x`, step `k` lowers
`f` by at least `η (1 - η L) ‖∇f(x_k)‖²`. -/
theorem iterate_succ_le_sub_of_segment (hf : ContDiff ℝ 2 f) {x : E}
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L) (k : ℕ) :
    f ((gradientStep f η)^[k + 1] x) ≤ f ((gradientStep f η)^[k] x) -
      η * (1 - η * L) * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2 := by
  have h := hS k
  rw [Function.iterate_succ_apply'] at h ⊢
  exact le_sub_of_gradientStep_of_segment hf h

/-- **`f` decreases along an orbit whose segments carry the bound**, when `0 ≤ η`
and `η L ≤ 1`. -/
theorem antitone_iterate_of_segment (hf : ContDiff ℝ 2 f) (hη : 0 ≤ η) (hηL : η * L ≤ 1)
    {x : E}
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L) :
    Antitone fun k => f ((gradientStep f η)^[k] x) := by
  refine antitone_nat_of_succ_le fun k => ?_
  have h := iterate_succ_le_sub_of_segment hf hS k
  have hnn := mul_nonneg (mul_nonneg hη (sub_nonneg.2 hηL))
    (sq_nonneg ‖gradient f ((gradientStep f η)^[k] x)‖)
  linarith

variable [MeasurableSpace E] [BorelSpace E]

/-- **Gradient descent almost never converges to a strict saddle along an orbit
with the bound at its points.**  If `f` is `C²`, `0 < η` and `η L < 1`, the starts
whose orbit has `‖∇²f(x_k)‖ ≤ L` at every `k` and converges to a strict saddle form a
null set for every Haar measure `μ`. These starts form a set that gradient descent
maps into itself, on which `‖∇²f‖ ≤ L`. -/
theorem measure_iterate_norm_le_tendsto_strictSaddle_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] (hf : ContDiff ℝ 2 f) (hη : 0 < η) (hηL : η * L < 1) :
    μ {x | (∀ k, ‖fderiv ℝ (gradient f) ((gradientStep f η)^[k] x)‖ ≤ L) ∧
      ∃ z, IsStrictSaddle f z ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)} =
      0 :=
  measure_inter_tendsto_strictSaddle_eq_zero μ hf
    (K := {x | ∀ k, ‖fderiv ℝ (gradient f) ((gradientStep f η)^[k] x)‖ ≤ L})
    (fun x hx k => by simpa only [Function.iterate_succ_apply] using hx (k + 1))
    (fun x hx => hx 0) hη hηL

/-- **The same under the bound on the segments**, which gives it at their left ends
`x_k`: the starts whose orbit has `‖∇²f‖ ≤ L` on every segment `[x_k, x_{k+1}]` and
converges to a strict saddle form a null set. -/
theorem measure_segment_tendsto_strictSaddle_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    (hf : ContDiff ℝ 2 f) (hη : 0 < η) (hηL : η * L < 1) :
    μ {x | (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) ∧
      ∃ z, IsStrictSaddle f z ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)} =
      0 := by
  refine measure_mono_null (fun x hx => ?_)
    (measure_iterate_norm_le_tendsto_strictSaddle_eq_zero μ hf hη hηL)
  exact ⟨fun k => hx.1 k _ (left_mem_segment ℝ _ _), hx.2⟩

/-- **Almost surely**: for a law `ν` absolutely continuous with respect to a Haar
measure, `ν`-almost every start whose orbit has `‖∇²f‖ ≤ L` on every segment
`[x_k, x_{k+1}]` converges to no strict saddle. -/
theorem ae_not_tendsto_strictSaddle_of_segment (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hη : 0 < η) (hηL : η * L < 1) :
    ∀ᵐ x ∂ν, (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) →
      ∀ z, IsStrictSaddle f z → ¬Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z) := by
  rw [ae_iff]
  refine hν (measure_mono_null (fun x hx => ?_)
    (measure_segment_tendsto_strictSaddle_eq_zero μ hf hη hηL))
  rw [Set.mem_ofPred_eq] at hx
  push Not at hx
  obtain ⟨hS, z, hz, hlim⟩ := hx
  exact ⟨hS, z, hz, hlim⟩

/-- `‖x‖² / 2` on the line has the Hessian `1`, so the bound holds on every segment
with `L = 1`; with `η = 1/2` a step from `1` lowers `f` by at least `1/4`. -/
example := le_sub_of_gradientStep_of_segment (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (L := 1) (η := 1 / 2) (contDiff_const.mul (contDiff_norm_sq ℝ)) (x := 1) fun y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le

/-- The orbit of `1` for `‖x‖² / 2`, `L = 1` and `η = 1/2` satisfies the bound on
every segment, and step `k` lowers `f` by at least `‖x_k‖² / 4`. -/
example := iterate_succ_le_sub_of_segment (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (L := 1) (η := 1 / 2) (contDiff_const.mul (contDiff_norm_sq ℝ)) (x := 1) (fun _ y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le) 0

/-- Along the same orbit `f` decreases: `0 ≤ 1/2` and `1/2 · 1 ≤ 1`. -/
example := antitone_iterate_of_segment (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1)
  (η := 1 / 2) (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) (x := 1)
  fun _ y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le

/-- The avoidance theorem applies to the double well `x⁴/4 - x²/2` on the line, whose
Hessian `3x² - 1` is unbounded, with Lebesgue measure, `η = 1/10` and `L = 23/4`:
`η L = 23/40 < 1`. -/
example := measure_iterate_norm_le_tendsto_strictSaddle_eq_zero (volume : Measure ℝ)
  (f := fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) (L := 23 / 4) (η := 1 / 10) (by fun_prop)
  (by norm_num) (by norm_num)

/-- The same under the bound on the segments, for the double well. -/
example := measure_segment_tendsto_strictSaddle_eq_zero (volume : Measure ℝ)
  (f := fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) (L := 23 / 4) (η := 1 / 10) (by fun_prop)
  (by norm_num) (by norm_num)

/-- And almost surely, for the double well and a start drawn uniformly from
`[-3/2, 3/2]`: the law is the restriction of Lebesgue measure. -/
example := ae_not_tendsto_strictSaddle_of_segment (volume : Measure ℝ)
  (ν := volume.restrict (Icc (-3 / 2) (3 / 2)))
  (Measure.absolutelyContinuous_of_le Measure.restrict_le_self)
  (f := fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) (L := 23 / 4) (η := 1 / 10) (by fun_prop)
  (by norm_num) (by norm_num)

end Transformer.Landscape.SaddleAvoidance
