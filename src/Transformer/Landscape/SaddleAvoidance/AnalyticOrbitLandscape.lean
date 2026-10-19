module

public import Transformer.Landscape.SaddleAvoidance.OrbitLandscape
public import Transformer.Landscape.SaddleAvoidance.LocalInequality
public import Transformer.Landscape.SaddleAvoidance.SecondOrderCritical

/-!
# Saddle avoidance — the scheme along the orbit for analytic functions

An analytic function on a finite-dimensional real inner product space is `C²`
(`AnalyticOnNhd.contDiff`) and satisfies the Łojasiewicz inequality at every point
(`exists_isLojasiewiczAt_of_analyticOnNhd`), so for it the scheme along the orbit of
`OrbitLandscape` needs neither that hypothesis nor a level. Almost every start whose
orbit has `‖∇²f‖ ≤ L`, `η L < 1`, on the segment of every step (S) converges to a
critical point that is not a strict saddle, or runs off to infinity
(`ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd`);
to a global minimizer if every critical point is one or a strict saddle
(`ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd`).

The double well `x⁴/4 - x²/2` on the line shows that (S) asks for no global bound.
Its Hessian `3x² - 1` is unbounded, yet for `η = 1/10` gradient descent maps
`[-3/2, 3/2]` into itself, and there the Hessian is at most `23/4`. So almost every
start in `[-3/2, 3/2]` converges to one of the global minimizers `±1`.

Source: item M8.3 of `landscape_plan.md`.
-/

@[expose] public section

open Set Filter MeasureTheory
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L η : ℝ}

/-- **The scheme along the orbit for analytic `f`**, with neither the Łojasiewicz
hypothesis nor a level: for `ν`-almost every `x` whose orbit has `‖∇²f‖ ≤ L` on the
segment of every step, the orbit converges to a critical point `w` with `f w ≤ f x`
that is not a strict saddle, or `‖x_k‖ → ∞`. -/
theorem ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ)
    (hf : AnalyticOnNhd ℝ f univ) (hη : 0 < η) (hηL : η * L < 1) :
    ∀ᵐ x ∂ν,
      (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) →
      (∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧ gradient f w = 0 ∧
          f w ≤ f x ∧ ¬IsStrictSaddle f w) ∨
        Tendsto (fun k => ‖(gradientStep f η)^[k] x‖) atTop atTop := by
  filter_upwards [ae_not_tendsto_strictSaddle_of_segment μ hν hf.contDiff hη hηL] with x hx hS
  exact (exists_tendsto_or_tendsto_norm_of_segment hf.contDiff hη hηL hS
    fun z _ _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z).imp_left
      fun ⟨w, hw, hw0, hwx⟩ => ⟨w, hw, hw0, hwx, fun hs => hx hS w hs hw⟩

/-- **The scheme along the orbit for analytic `f`, to a global minimizer.**  If every
critical point is a global minimizer or a strict saddle, then for `ν`-almost every `x`
whose orbit satisfies (S), the orbit converges to a global minimizer, or
`‖x_k‖ → ∞`. -/
theorem ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ)
    (hf : AnalyticOnNhd ℝ f univ) (hη : 0 < η) (hηL : η * L < 1)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν,
      (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) →
      (∃ w, IsMinOn f univ w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) ∨
        Tendsto (fun k => ‖(gradientStep f η)^[k] x‖) atTop atTop := by
  filter_upwards [ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd
    μ hν hf hη hηL] with x hx hS
  exact (hx hS).imp_left fun ⟨w, hw, hw0, _, hws⟩ => ⟨w, (hcrit w hw0).resolve_right hws, hw⟩

/-- The double well `x⁴/4 - x²/2` on the line. -/
noncomputable def doubleWell (x : ℝ) : ℝ := x ^ 4 / 4 - x ^ 2 / 2

/-- The double well is analytic. -/
theorem analyticOnNhd_doubleWell : AnalyticOnNhd ℝ doubleWell univ :=
  ((analyticOnNhd_id.fun_pow 4).div_const).fun_sub ((analyticOnNhd_id.fun_pow 2).div_const)

/-- The gradient of the double well is `x³ - x`. -/
theorem gradient_doubleWell : gradient doubleWell = fun x => x ^ 3 - x := funext fun x => by
  rw [gradient_eq_deriv']
  exact (((hasDerivAt_pow 4 x).div_const 4).fun_sub
    ((hasDerivAt_pow 2 x).div_const 2)).deriv.trans (by ring)

/-- The Hessian of the double well is `3x² - 1`. -/
theorem fderiv_gradient_doubleWell (y : ℝ) :
    fderiv ℝ (gradient doubleWell) y = ContinuousLinearMap.toSpanSingleton ℝ (3 * y ^ 2 - 1) := by
  rw [gradient_doubleWell, ((hasDerivAt_pow 3 y).fun_sub (hasDerivAt_id' y)).hasFDerivAt.fderiv]
  norm_num

/-- The norm of the Hessian of the double well is `|3x² - 1|`. -/
theorem norm_fderiv_gradient_doubleWell (y : ℝ) :
    ‖fderiv ℝ (gradient doubleWell) y‖ = |3 * y ^ 2 - 1| := by
  rw [fderiv_gradient_doubleWell, ContinuousLinearMap.norm_toSpanSingleton, Real.norm_eq_abs]

/-- The Hessian of the double well is unbounded. -/
theorem not_bddAbove_norm_fderiv_gradient_doubleWell :
    ¬BddAbove (range fun y => ‖fderiv ℝ (gradient doubleWell) y‖) := by
  rintro ⟨M, hM⟩
  have h := hM ⟨|M| + 1, rfl⟩
  simp only [norm_fderiv_gradient_doubleWell] at h
  nlinarith [le_abs_self M, abs_nonneg M, le_abs_self (3 * (|M| + 1) ^ 2 - 1)]

/-- For `η = 1/10` gradient descent maps `[-3/2, 3/2]` into itself:
`10 (3/2 - (x - (x³ - x)/10)) = (3/2 - x)(35/4 - 3x/2 - x²) + 15/8`, and symmetrically. -/
theorem mapsTo_gradientStep_doubleWell :
    MapsTo (gradientStep doubleWell (1 / 10)) (Icc (-3 / 2) (3 / 2)) (Icc (-3 / 2) (3 / 2)) := by
  intro x hx
  simp only [gradientStep, gradient_doubleWell, smul_eq_mul, mem_Icc] at hx ⊢
  obtain ⟨h1, h2⟩ := hx
  constructor
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ x + 3 / 2)
      (by nlinarith : (0 : ℝ) ≤ 35 / 4 + 3 * x / 2 - x ^ 2)]
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 3 / 2 - x)
      (by nlinarith : (0 : ℝ) ≤ 35 / 4 - 3 * x / 2 - x ^ 2)]

/-- **Every orbit from `[-3/2, 3/2]` satisfies (S)** for the double well, `η = 1/10`
and `L = 23/4`: the orbit stays in the interval, which is convex, and there the
Hessian `3y² - 1` lies in `[-1, 23/4]`. -/
theorem norm_fderiv_gradient_doubleWell_le_of_mem_Icc :
    ∀ x ∈ Icc (-3 / 2 : ℝ) (3 / 2), ∀ k,
      ∀ y ∈ segment ℝ ((gradientStep doubleWell (1 / 10))^[k] x)
        ((gradientStep doubleWell (1 / 10))^[k + 1] x),
        ‖fderiv ℝ (gradient doubleWell) y‖ ≤ 23 / 4 := by
  intro x hx k y hy
  have hyI := (convex_Icc (-3 / 2 : ℝ) (3 / 2)).segment_subset
    (mapsTo_gradientStep_doubleWell.iterate k hx)
    (mapsTo_gradientStep_doubleWell.iterate (k + 1) hx) hy
  rw [norm_fderiv_gradient_doubleWell, abs_le]
  constructor <;> nlinarith [hyI.1, hyI.2]

/-- The bound on the first segment of the orbit from the end `3/2` of the interval. -/
example := norm_fderiv_gradient_doubleWell_le_of_mem_Icc (3 / 2) (by norm_num) 0

/-- The critical points of the double well are the global minimizers `±1` and the
strict saddle `0`, where the Hessian is `-1`. -/
theorem isMinOn_or_isStrictSaddle_doubleWell (z : ℝ) (hz : gradient doubleWell z = 0) :
    IsMinOn doubleWell univ z ∨ IsStrictSaddle doubleWell z := by
  refine (or_isStrictSaddle_iff analyticOnNhd_doubleWell.contDiff hz).2 fun h => ?_
  have h1 := h 1
  simp only [fderiv_gradient_doubleWell, ContinuousLinearMap.toSpanSingleton_apply, one_smul,
    Real.inner_apply, mul_one] at h1
  simp only [gradient_doubleWell] at hz
  have hz2 : z ^ 2 = 1 := by
    rcases mul_eq_zero.1 (show z * (z ^ 2 - 1) = 0 by linear_combination hz) with h0 | h0
    · rw [h0] at h1
      norm_num at h1
    · linarith
  refine isMinOn_univ_iff.2 fun y => ?_
  simp only [doubleWell]
  nlinarith [sq_nonneg (y ^ 2 - 1)]

/-- `0` is a critical point of the double well; it is the strict saddle. -/
example := isMinOn_or_isStrictSaddle_doubleWell 0 (by rw [gradient_doubleWell]; norm_num)

/-- The first theorem applies to the double well with `η = 1/10`, `L = 23/4` and a
start drawn uniformly from `[-3/2, 3/2]`. -/
example := ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd
  (volume : Measure ℝ) (ν := volume.restrict (Icc (-3 / 2) (3 / 2)))
  (Measure.absolutelyContinuous_of_le Measure.restrict_le_self) analyticOnNhd_doubleWell
  (L := 23 / 4) (η := 1 / 10) (by norm_num) (by norm_num)

/-- So does the second: every critical point is a global minimizer or a strict saddle. -/
example := ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd
  (volume : Measure ℝ) (ν := volume.restrict (Icc (-3 / 2) (3 / 2)))
  (Measure.absolutelyContinuous_of_le Measure.restrict_le_self) analyticOnNhd_doubleWell
  (L := 23 / 4) (η := 1 / 10) (by norm_num) (by norm_num) isMinOn_or_isStrictSaddle_doubleWell

/-- **Almost every start in `[-3/2, 3/2]` converges to a global minimizer of the
double well**, for `η = 1/10`, although its Hessian is unbounded: every orbit from
the interval satisfies (S) with `L = 23/4` and stays in the interval, so it does not
run off to infinity. -/
example : ∀ᵐ x ∂(volume.restrict (Icc (-3 / 2 : ℝ) (3 / 2))), ∃ w, IsMinOn doubleWell univ w ∧
    Tendsto (fun k => (gradientStep doubleWell (1 / 10))^[k] x) atTop (𝓝 w) := by
  filter_upwards [ae_restrict_mem measurableSet_Icc,
    ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd (volume : Measure ℝ)
      (Measure.absolutelyContinuous_of_le Measure.restrict_le_self) analyticOnNhd_doubleWell
      (L := 23 / 4) (η := 1 / 10) (by norm_num) (by norm_num)
      isMinOn_or_isStrictSaddle_doubleWell] with x hx h
  refine (h (norm_fderiv_gradient_doubleWell_le_of_mem_Icc x hx)).resolve_right fun hinf => ?_
  obtain ⟨k, hk⟩ := (hinf.eventually_gt_atTop (3 / 2)).exists
  have hkI := mapsTo_gradientStep_doubleWell.iterate k hx
  rw [Real.norm_eq_abs] at hk
  exact (abs_le.2 ⟨by linarith [hkI.1], hkI.2⟩).not_gt hk

end Transformer.Landscape.SaddleAvoidance
