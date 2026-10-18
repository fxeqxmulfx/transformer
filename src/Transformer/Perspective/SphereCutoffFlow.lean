/-
# Saddle avoidance for block-rescaled gradient flows on a shell of tuples

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Let `g` be `C²` and the scales `w_i > 0` be `C¹` on the tuples of nonzero vectors, and
consider the trajectories of `ẏ = M(y) ∇g(y)`, `M(y)` the block scaling by `w(y)`, that
stay in an open shell `R`.  For a cut-off `χ` equal to `1` on `R` with compact support
away from the zero vectors, `F = -χ g` is `C²` on the whole space and the block scaling
`G` by `χ (w - 1) + 1 > 0` is `C¹` and symmetric positive definite
(`blockScale_isSymmetric_pos`); `G ∇F` is `C¹` with compact support, hence bounded and
Lipschitz.  On `R` the flow `ẋ = -G(x) ∇F(x)` is the given one, so every trajectory in
`R` is a trajectory of `metricGradientFlow G F`.  A critical point `z ∈ R` of `g` with
a direction of positive second derivative is a strict saddle of `F`.  Saddle avoidance
for metric gradient flows then gives: for almost every start, no trajectory in `R`
converges to such a point (`ae_not_tendsto_of_cutoff`).

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`: the set of initial conditions
whose gradient ascent "converges to a strict saddle point of `f` is of volume zero", by
the center-stable manifold theorem, applied to the time-one map as in `SaddleAvoidance`;
for `thm: boumal` of §6.
-/

import Transformer.Perspective.SphereCutoff
import Transformer.Perspective.CircleEnergyHessian
import Transformer.SaddleAvoidance.MetricAvoidance
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
    (hχR : ∀ x ∈ R, χ x = 1) (hχ0 : ∀ x, 0 ≤ χ x) (hχ1 : ∀ x, χ x ≤ 1)
    (hg : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 2 g x)
    (hw : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 1 w x)
    (hwpos : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ∀ i, 0 < w x i) :
    ∀ᵐ y ∂μ, ∀ Y : ℝ → TupleSpace d n, Y 0 = y → (∀ t, Y t ∈ R) →
      (∀ t, HasDerivAt Y (blockScale (w (Y t)) (gradient g (Y t))) t) →
      ∀ z ∈ R, gradient g z = 0 → ∀ (h : TupleSpace d n) (f' : ℝ → ℝ) (c : ℝ), 0 < c →
        (∀ s, HasDerivAt (fun u => g (z + u • h)) (f' s) s) → HasDerivAt f' c 0 →
          ¬Tendsto Y atTop (𝓝 z) := by
  set F : TupleSpace d n → ℝ := fun x => -(χ x * g x)
  set G : TupleSpace d n → TupleSpace d n →L[ℝ] TupleSpace d n :=
    fun x => blockScale fun i => χ x * (w x i - 1) + 1
  have hF : ContDiff ℝ 2 F := (contDiff_smul_of_tsupport_subset hχ hχU hg).neg
  have hG : ContDiff ℝ 1 G := by
    have h := (contDiff_smul_of_tsupport_subset (hχ.of_le (by norm_num)) hχU
      (φ := fun x => w x - 1) fun x hx => (hw x hx).sub contDiffAt_const).add
        (contDiff_const (c := (1 : Idx n → ℝ)))
    have heq : (fun x => χ x • (w x - 1) + 1) = fun x i => χ x * (w x i - 1) + 1 := by
      funext x i
      simp
    rw [heq] at h
    exact (blockScaleCLM (d := d)).contDiff.comp h
  have hF0 : ∀ x, x ∉ tsupport χ → gradient F x = 0 := fun x hx => by
    have h : F =ᶠ[𝓝 x] fun _ => 0 := by
      filter_upwards [notMem_tsupport_iff_eventuallyEq.1 hx] with y hy
      simp [F, show χ y = 0 from hy]
    rw [h.gradient_eq, gradient_fun_const]
  have hVc : ContDiff ℝ 1 fun x => G x (gradient F x) := hG.clm_apply (contDiff_gradient hF)
  have hVs : HasCompactSupport fun x => G x (gradient F x) :=
    hχc.mono' fun x hx => by_contra fun h => hx (by simp only [hF0 x h, map_zero])
  obtain ⟨L, hL⟩ := hVc.lipschitzWith_of_hasCompactSupport hVs one_ne_zero
  obtain ⟨M, hM⟩ := hVs.exists_bound_of_continuous hVc.continuous
  have hpos : ∀ x i, 0 < χ x * (w x i - 1) + 1 := fun x i => by
    by_cases hx : ∀ i, x.ofLp i ≠ 0
    · rcases (hχ1 x).lt_or_eq with h | h
      · nlinarith [mul_nonneg (hχ0 x) (hwpos x hx i).le]
      · rw [h]
        linarith [hwpos x hx i]
    · rw [image_eq_zero_of_notMem_tsupport fun h => hx (hχU h)]
      norm_num
  have hae := ae_not_tendsto_metricGradientFlow_of_isStrictSaddle μ
    Measure.AbsolutelyContinuous.rfl hF hG hL hM fun z _ => blockScale_isSymmetric_pos (hpos z)
  filter_upwards [hae] with y hy
  intro Y hY0 hYR hY z hzR hgz h f' c hc h1 h2 hlim
  have hFR : ∀ x ∈ R, F =ᶠ[𝓝 x] fun x => -g x := fun x hx => by
    filter_upwards [hR.mem_nhds hx] with x' hx'
    simp [F, hχR x' hx']
  have hgradR : ∀ x ∈ R, gradient F x = -gradient g x := fun x hx => by
    rw [(hFR x hx).gradient_eq]
    simp only [gradient, fderiv_fun_neg, map_neg]
  have hGR : ∀ x ∈ R, G x = blockScale (w x) := fun x hx => by
    simp only [G, hχR x hx, one_mul, sub_add_cancel]
  have hflow : ∀ t, metricGradientFlow G F t y = Y t := fun t => by
    rw [← hY0]
    refine (isFlowField_neg_metricGradient hL hM).flow_eq (fun s => ?_) 0 t
    change HasDerivAt Y (-G (Y s) (gradient F (Y s))) s
    rw [hGR _ (hYR s), hgradR _ (hYR s), map_neg, neg_neg]
    exact hY s
  have hFz : gradient F z = 0 := by rw [hgradR z hzR, hgz, neg_zero]
  have hφd : Differentiable ℝ fun u : ℝ => F (z + u • h) :=
    (hF.differentiable (by norm_num)).comp
      ((differentiable_const _).add (differentiable_id.smul_const h))
  have hline : Tendsto (fun u : ℝ => z + u • h) (𝓝 0) (𝓝 z) := by
    have hc : Continuous fun u : ℝ => z + u • h := by fun_prop
    simpa using hc.tendsto 0
  have hφ : (fun u : ℝ => F (z + u • h)) =ᶠ[𝓝 0] fun u => -g (z + u • h) :=
    hline.eventually (hFR z hzR)
  have hderiv : deriv (fun u : ℝ => F (z + u • h)) =ᶠ[𝓝 0] fun s => -f' s := by
    filter_upwards [hφ.deriv] with s hs
    rw [hs]
    exact (h1 s).neg.deriv
  have hHess : inner (𝕜 := ℝ) (fderiv ℝ (gradient F) z h) h = -c :=
    inner_fderiv_gradient_eq_of_hasDerivAt hF (fun s => (hφd s).hasDerivAt)
      (h2.neg.congr_of_eventuallyEq hderiv)
  exact hy z (isStrictSaddle_of_inner_neg hF hFz (by rw [hHess]; linarith))
    (by simpa only [hflow] using hlim)

/-- The hypotheses of `ae_not_tendsto_of_cutoff` hold on the shell `1/2 < ‖x‖² < 2` of
`ℝ`, with its cut-off, `g = 0` and `w = 1`. -/
example := ae_not_tendsto_of_cutoff (volume : Measure (TupleSpace 1 1))
  (isOpen_blockShell (1 / 2) 2) (contDiff_blockCutoff (by norm_num) (by norm_num))
  (hasCompactSupport_blockCutoff (by norm_num) (by norm_num))
  (tsupport_blockCutoff_subset (by norm_num) (by norm_num))
  (fun _ hx => blockCutoff_eq_one (by norm_num) (by norm_num) hx)
  (blockCutoff_nonneg (by norm_num) (by norm_num)) (blockCutoff_le_one (by norm_num) (by norm_num))
  (g := fun _ => 0) (fun _ _ => contDiffAt_const) (w := fun _ _ => 1)
  (fun _ _ => contDiffAt_const) fun _ _ _ => one_pos

end Perspective
end Transformer
