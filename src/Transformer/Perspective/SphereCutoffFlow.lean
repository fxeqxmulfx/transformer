/-
# Saddle avoidance for block-rescaled gradient flows on a shell of tuples

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Let `g` be `C²` and the scales `w_i > 0` be `C¹` on the tuples of nonzero vectors, and
consider the trajectories of `ẏ = M(y) ∇g(y)`, `M(y)` the block scaling by `w(y)`, that
stay in an open shell `R`.  This is the flow `ẏ = -M(y) ∇(-g)(y)`, and off the zero
vectors `M` is `C¹` and symmetric positive definite (`blockScale_isSymmetric_pos`).  For
a cut-off `χ` equal to `1` on `R` with compact support away from the zero vectors, the
trajectories in `R` stay in the support of `χ`, a compact set, so saddle avoidance on
an open set (`SaddleAvoidance.ae_not_tendsto_metricGradient_of_isCompact`) applies.  A
critical point `z ∈ R` of `g` with a direction of positive second derivative is a
strict saddle of `-χ g`, which is `C²` on the whole space, and so of `-g`, equal to it
near `z`.  Hence for almost every start, no trajectory in `R` converges to such a point
(`ae_not_tendsto_of_cutoff`).

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`: the set of initial conditions
whose gradient ascent "converges to a strict saddle point of `f` is of volume zero", by
the center-stable manifold theorem, applied to the time-one map as in `SaddleAvoidance`;
for `thm: boumal` of §6.
-/

import Transformer.Perspective.SphereCutoff
import Transformer.Perspective.CircleEnergyHessian
import Transformer.SaddleAvoidance.OpenAvoidance
import Transformer.SaddleAvoidance.Necessary

open Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

open SaddleAvoidance

variable {d n : ℕ}

/-- **A block scaling by positive numbers is symmetric positive definite.** -/
theorem blockScale_isSymmetric_pos {c : Idx n → ℝ} (hc : ∀ i, 0 < c i) :
    (blockScale (d := d) c : TupleSpace d n →ₗ[ℝ] TupleSpace d n).IsSymmetric ∧
      ∀ v : TupleSpace d n, v ≠ 0 → 0 < inner (𝕜 := ℝ) (blockScale c v) v := by
  refine ⟨fun v w => ?_, fun v hv => ?_⟩
  · change inner (𝕜 := ℝ) (blockScale c v) w = inner (𝕜 := ℝ) v (blockScale c w)
    rw [inner_blockScale, real_inner_comm, inner_blockScale]
    exact Finset.sum_congr rfl fun i _ => by rw [real_inner_comm]
  · obtain ⟨i, hi⟩ : ∃ i, v.ofLp i ≠ 0 := by
      by_contra h
      simp only [not_exists, not_not] at h
      exact hv (PiLp.ext fun i => by simp [h i])
    rw [inner_blockScale]
    exact Finset.sum_pos' (fun j _ => mul_nonneg (hc j).le real_inner_self_nonneg)
      ⟨i, Finset.mem_univ _, mul_pos (hc i) (real_inner_self_pos.2 hi)⟩

/-- The hypothesis of `blockScale_isSymmetric_pos` holds for the scaling by `1`. -/
example : ∀ _ : Idx 2, (0 : ℝ) < 1 := fun _ => one_pos

variable (μ : Measure (TupleSpace d n)) [μ.IsAddHaarMeasure] {R : Set (TupleSpace d n)}
  {χ g : TupleSpace d n → ℝ} {w : TupleSpace d n → Idx n → ℝ}

/-- **Almost no start has a trajectory in the shell converging to a saddle of `g`.**
For almost every `y`, no solution of `ẏ = M(y) ∇g(y)` from `y` that stays in `R`
converges to a critical point `z ∈ R` of `g` along whose line `u ↦ g(z + u h)` the
second derivative is positive. -/
theorem ae_not_tendsto_of_cutoff (hR : IsOpen R) (hχ : ContDiff ℝ 2 χ)
    (hχc : HasCompactSupport χ) (hχU : tsupport χ ⊆ {x | ∀ i, x.ofLp i ≠ 0})
    (hχR : ∀ x ∈ R, χ x = 1)
    (hg : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 2 g x)
    (hw : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 1 w x)
    (hwpos : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ∀ i, 0 < w x i) :
    ∀ᵐ y ∂μ, ∀ Y : ℝ → TupleSpace d n, Y 0 = y → (∀ t, Y t ∈ R) →
      (∀ t, HasDerivAt Y (blockScale (w (Y t)) (gradient g (Y t))) t) →
      ∀ z ∈ R, gradient g z = 0 → ∀ (h : TupleSpace d n) (f' : ℝ → ℝ) (c : ℝ), 0 < c →
        (∀ s, HasDerivAt (fun u => g (z + u • h)) (f' s) s) → HasDerivAt f' c 0 →
          ¬Tendsto Y atTop (𝓝 z) := by
  have hgrad : ∀ x, gradient (fun x => -g x) x = -gradient g x := fun x => by
    simp only [gradient, fderiv_fun_neg, map_neg]
  have hae := ae_not_tendsto_metricGradient_of_isCompact μ Measure.AbsolutelyContinuous.rfl
    isOpen_setOf_ofLp_ne_zero hχc hχU (f := fun x => -g x) (fun x hx => (hg x hx).neg)
    (G := fun x => blockScale (w x))
    (fun x hx => (blockScaleCLM (d := d)).contDiff.contDiffAt.comp x (hw x hx))
    fun x hx => blockScale_isSymmetric_pos (hwpos x hx)
  filter_upwards [hae] with y hy
  intro Y hY0 hYR hY z hzR hgz h f' c hc h1 h2 hlim
  set F : TupleSpace d n → ℝ := fun x => -(χ x * g x)
  have hF : ContDiff ℝ 2 F := (contDiff_smul_of_tsupport_subset hχ hχU hg).neg
  have hFR : F =ᶠ[𝓝 z] fun x => -g x := by
    filter_upwards [hR.mem_nhds hzR] with x hx
    simp [F, hχR x hx]
  have hFz : gradient F z = 0 := by rw [hFR.gradient_eq, hgrad, hgz, neg_zero]
  have hφd : Differentiable ℝ fun u : ℝ => F (z + u • h) :=
    (hF.differentiable (by norm_num)).comp
      ((differentiable_const _).add (differentiable_id.smul_const h))
  have hline : Tendsto (fun u : ℝ => z + u • h) (𝓝 0) (𝓝 z) := by
    have hc : Continuous fun u : ℝ => z + u • h := by fun_prop
    simpa using hc.tendsto 0
  have hφ : (fun u : ℝ => F (z + u • h)) =ᶠ[𝓝 0] fun u => -g (z + u • h) :=
    hline.eventually hFR
  have hderiv : deriv (fun u : ℝ => F (z + u • h)) =ᶠ[𝓝 0] fun s => -f' s := by
    filter_upwards [hφ.deriv] with s hs
    rw [hs]
    exact (h1 s).neg.deriv
  have hHess : inner (𝕜 := ℝ) (fderiv ℝ (gradient F) z h) h = -c :=
    inner_fderiv_gradient_eq_of_hasDerivAt hF (fun s => (hφd s).hasDerivAt)
      (h2.neg.congr_of_eventuallyEq hderiv)
  refine hy Y hY0 (fun t _ => subset_tsupport χ ?_) (fun t _ => ?_) z
    ((isStrictSaddle_of_inner_neg hF hFz (by rw [hHess]; linarith)).congr_of_eventuallyEq
      hFR.symm) hlim
  · rw [Function.mem_support, hχR _ (hYR t)]
    exact one_ne_zero
  · simp only [hgrad, map_neg, neg_neg]
    exact (hY t).hasDerivWithinAt

/-- The hypotheses of `ae_not_tendsto_of_cutoff` hold on the shell `1/2 < ‖x‖² < 2` of
`ℝ`, with its cut-off, `g = 0` and `w = 1`. -/
example := ae_not_tendsto_of_cutoff (volume : Measure (TupleSpace 1 1))
  (isOpen_blockShell (1 / 2) 2) (contDiff_blockCutoff (by norm_num) (by norm_num))
  (hasCompactSupport_blockCutoff (by norm_num) (by norm_num))
  (tsupport_blockCutoff_subset (by norm_num) (by norm_num))
  (fun _ hx => blockCutoff_eq_one (by norm_num) (by norm_num) hx)
  (g := fun _ => 0) (fun _ _ => contDiffAt_const) (w := fun _ _ => 1)
  (fun _ _ => contDiffAt_const) fun _ _ _ => one_pos

end Perspective
end Transformer
