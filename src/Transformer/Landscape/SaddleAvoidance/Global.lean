/-
# Saddle avoidance — the starts converging to unstable points are null

Let `g` be a `C¹` map of a finite-dimensional real inner product space into
itself.

* If the derivative `A` of `g` at `z` is symmetric with an eigenvalue of
  modulus greater than one, then for some `r > 0` the starts whose orbit stays
  in `B̄(z, r)` form a null set (`exists_measure_stayNear_eq_zero`).  `A` has
  an expanding splitting `P` with constant `c > 1` and a direction `e` fixed
  by `P` (`exists_expandingSplitting`); on a small ball `g` differs from `A` by
  a map of Lipschitz constant `(c - 1)/4` (`exists_closedBall_sub_fderiv_le`);
  so `stayNear g z r` meets each line of direction `e` at most once
  (`line_meets_stayNear_once`) and is null
  (`measure_eq_zero_of_line_meets_once`).
* If moreover the derivative of `g` has nonzero determinant everywhere, the
  starts whose orbit converges to a fixed point of this kind form a null set
  (`measure_tendsto_symmetricUnstable_eq_zero`), however many such points
  there are.  Countably many of the balls `B(z, r_z)` cover them (Lindelöf).
  An orbit converging to such a point ends in one of these balls, so from some
  step `m` on it lies in the corresponding `stayNear`, and its start lies in
  the preimage of that null set under `g^m`, again null
  (`measure_preimage_iterate_eq_zero`).
* On a set `S` that `g` maps into itself it suffices that the determinant be
  nonzero on `S`: the starts in `S` converging to such a point form a null set
  (`measure_inter_tendsto_symmetricUnstable_eq_zero`), since their orbits stay
  in `S` (`measure_inter_preimage_iterate_eq_zero`).

This is the proof of `thm:main-general` in §4 of arXiv:1710.07406, for
fixed points with a symmetric derivative, with the center-stable manifold
theorem replaced by `exists_measure_stayNear_eq_zero`.
-/

import Transformer.Landscape.SaddleAvoidance.Cone
import Transformer.Landscape.SaddleAvoidance.Preimage
import Transformer.Landscape.SaddleAvoidance.Spectral
import Mathlib.Analysis.Calculus.ContDiff.Operations
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Normed.Lp.MeasurableSpace

open Set Metric Filter MeasureTheory
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

section Approx

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **A `C¹` map is close to its derivative near a point.**  For every
`ε > 0` there is a closed ball about `z` on which
`‖g x - g y - Dg(z)(x - y)‖ ≤ ε ‖x - y‖`. -/
theorem exists_closedBall_sub_fderiv_le {g : E → E} (hg : ContDiff ℝ 1 g) (z : E) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ r > 0, ∀ x ∈ closedBall z r, ∀ y ∈ closedBall z r,
      ‖g x - g y - fderiv ℝ g z (x - y)‖ ≤ ε * ‖x - y‖ := by
  obtain ⟨δ, hδ, hball⟩ := Metric.continuous_iff.1 (hg.continuous_fderiv one_ne_zero) z ε hε
  refine ⟨δ / 2, half_pos hδ, fun x hx y hy => ?_⟩
  have hdiff : ∀ w, HasFDerivAt (fun w => g w - fderiv ℝ g z w)
      (fderiv ℝ g w - fderiv ℝ g z) w := fun w =>
    ((hg.differentiable one_ne_zero w).hasFDerivAt).sub (fderiv ℝ g z).hasFDerivAt
  have hbound : ∀ w ∈ closedBall z (δ / 2), ‖fderiv ℝ g w - fderiv ℝ g z‖ ≤ ε := fun w hw => by
    rw [← dist_eq_norm]
    exact (hball w (lt_of_le_of_lt (mem_closedBall.1 hw) (half_lt_self hδ))).le
  have h := Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun w _ => (hdiff w).hasFDerivWithinAt) hbound (convex_closedBall z (δ / 2)) hy hx
  calc ‖g x - g y - fderiv ℝ g z (x - y)‖
      = ‖g x - fderiv ℝ g z x - (g y - fderiv ℝ g z y)‖ := by
        rw [map_sub]
        congr 1
        abel
    _ ≤ ε * ‖x - y‖ := h

end Approx

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- **The local stable set of a point with a symmetric, somewhere stretching
derivative is null.**  If `g` is `C¹` and its derivative at `z` is symmetric
with an eigenvalue `λ`, `|λ| > 1`, then for some `r > 0` the starts whose
forward orbit stays in `B̄(z, r)` form a null set.  The fixed point property
`g z = z` plays no part here. -/
theorem exists_measure_stayNear_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) {z : E} (hsym : (fderiv ℝ g z : E →ₗ[ℝ] E).IsSymmetric) {lam : ℝ}
    (hlam : 1 < |lam|) (hev : Module.End.HasEigenvalue (fderiv ℝ g z : E →ₗ[ℝ] E) lam) :
    ∃ r > 0, μ (stayNear g z r) = 0 := by
  obtain ⟨P, c, e, hP, hc, hPe, he0⟩ := exists_expandingSplitting hsym hlam hev
  obtain ⟨r, hr, happrox⟩ := exists_closedBall_sub_fderiv_le hg z (ε := (c - 1) / 4)
    (by linarith)
  exact ⟨r, hr, measure_eq_zero_of_line_meets_once μ
    (isClosed_stayNear hg.continuous z r).measurableSet
    (isBounded_closedBall.subset (stayNear_subset g z r))
    (line_meets_stayNear_once hP hc happrox hPe he0)⟩

/-- **Gradient-like maps almost never converge to an unstable point from a
forward-invariant set.**  If `g` is `C¹`, maps `S` into itself, and its
derivative has nonzero determinant at every point of `S`, the starts in `S`
whose orbit converges to a fixed point of `g` at which the derivative is
symmetric and has an eigenvalue of modulus greater than one form a null set.
Nothing is asked of the determinant off `S`. -/
theorem measure_inter_tendsto_symmetricUnstable_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) {S : Set E} (hS : MapsTo g S S)
    (hdet : ∀ x ∈ S, (fderiv ℝ g x).det ≠ 0) :
    μ (S ∩ {x | ∃ z, IsSymmetricUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)}) =
      0 := by
  have hloc : ∀ z : {z // IsSymmetricUnstable g z}, ∃ r > 0, μ (stayNear g z r) = 0 :=
    fun z => by
      obtain ⟨-, hsym, lam, hlam, hev⟩ := z.2
      exact exists_measure_stayNear_eq_zero μ hg hsym hlam hev
  choose r hr hnull using hloc
  obtain ⟨T, hTc, hTU⟩ := TopologicalSpace.isOpen_iUnion_countable
    (fun z : {z // IsSymmetricUnstable g z} => ball (z : E) (r z)) fun _ => isOpen_ball
  refine measure_mono_null (t := ⋃ z ∈ T, ⋃ m : ℕ, S ∩ g^[m] ⁻¹' stayNear g z (r z)) ?_ ?_
  · rintro x ⟨hxS, w, hw, hlim⟩
    have hwU : w ∈ ⋃ z ∈ T, ball (z : E) (r z) := by
      rw [hTU]
      exact mem_iUnion.2 ⟨⟨w, hw⟩, mem_ball_self (hr _)⟩
    obtain ⟨z, hzT, hwz⟩ := mem_iUnion₂.1 hwU
    obtain ⟨m, hm⟩ := exists_iterate_mem_stayNear hlim hwz
    exact mem_iUnion₂.2 ⟨z, hzT, mem_iUnion.2 ⟨m, hxS, hm⟩⟩
  · rw [measure_biUnion_null_iff hTc]
    exact fun z _ => measure_iUnion_null fun m =>
      measure_inter_preimage_iterate_eq_zero μ hg hS hdet (hnull z) m

/-- **Gradient-like maps almost never converge to an unstable point.**  If `g`
is `C¹` and its derivative has nonzero determinant everywhere, the starts
whose orbit converges to a fixed point of `g` at which the derivative is
symmetric and has an eigenvalue of modulus greater than one form a null set.

`thm:main-general` of arXiv:1710.07406 asserts this for every unstable fixed
point, without the symmetry; this theorem covers the points where the
derivative is symmetric, which include all unstable fixed points of gradient
descent (`measure_tendsto_strictSaddle_eq_zero`). -/
theorem measure_tendsto_symmetricUnstable_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) :
    μ {x | ∃ z, IsSymmetricUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)} = 0 := by
  simpa using measure_inter_tendsto_symmetricUnstable_eq_zero μ hg (mapsTo_univ g univ)
    fun x _ => hdet x

/-- **The same for any initial law with a density.**  A start drawn from a
law absolutely continuous with respect to a Haar measure almost surely does
not converge to such a point. -/
theorem measure_tendsto_symmetricUnstable_eq_zero_of_absolutelyContinuous
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) :
    ν {x | ∃ z, IsSymmetricUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)} = 0 :=
  hν (measure_tendsto_symmetricUnstable_eq_zero μ hg hdet)

/-- The hypotheses of the five theorems above, with `ε = 1`, the restriction
of Lebesgue measure to the unit ball and `S = {0}`, hold for the doubling map
of the line, whose derivative `2` is symmetric with eigenvalue `2` and has
determinant `2`, and which fixes `0`. -/
example : ContDiff ℝ 1 (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) ∧ (0 : ℝ) < 1 ∧
    (fderiv ℝ (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) 0 :
      EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)).IsSymmetric ∧
    1 < |(2 : ℝ)| ∧
    Module.End.HasEigenvalue (fderiv ℝ (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) 0 :
      EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)) 2 ∧
    (∀ x, (fderiv ℝ (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) x).det ≠ 0) ∧
    (volume : Measure (EuclideanSpace ℝ (Fin 1))).restrict (ball 0 1) ≪ volume ∧
    MapsTo (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) {0} {0} := by
  have hd : ∀ x, fderiv ℝ (fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x) x =
      (2 : ℝ) • ContinuousLinearMap.id ℝ _ := fun x =>
    ((ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin 1))).hasFDerivAt.const_smul
      (2 : ℝ)).fderiv
  refine ⟨contDiff_const_smul 2, one_pos, fun u v => ?_, by norm_num, ?_, fun x => ?_,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self, fun x hx => by simp_all⟩
  · simp [hd, real_inner_smul_left, real_inner_smul_right]
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := EuclideanSpace.single 0 1) ⟨?_, ?_⟩
    · simp [hd]
    · simp
  · rw [hd, ContinuousLinearMap.det, ContinuousLinearMap.toLinearMap_smul,
      ContinuousLinearMap.coe_id, LinearMap.det_smul, LinearMap.det_id]
    simp

end Transformer.Landscape.SaddleAvoidance
