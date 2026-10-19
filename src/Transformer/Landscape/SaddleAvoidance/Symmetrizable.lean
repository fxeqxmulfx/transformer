module

public import Transformer.Landscape.SaddleAvoidance.Global

/-!
# Saddle avoidance — symmetrizable derivatives

The avoidance theorem for maps, `measure_tendsto_symmetricUnstable_eq_zero`,
asks for a symmetric derivative at the unstable fixed point. It suffices that
the derivative be conjugate to a symmetric map by a continuous linear
automorphism `T` (`IsSymmetrizableUnstable`); in finite dimension these are
the derivatives diagonalizable with real eigenvalues. The derivative
`exp (-G(z) ∇²f(z))` of the time-one map of `ẋ = -G(x) ∇f(x)` at a critical
point `z` is one when `G(z)` is symmetric positive definite.

* The conjugate `T ∘ g ∘ T⁻¹` has the derivative `T Dg(z) T⁻¹` at `T z`
  (`hasFDerivAt_conj`), symmetric with the eigenvalues of `Dg(z)`, and its
  iterates are those of `g` conjugated (`conj_iterate`). So the starts whose
  `g`-orbit stays within `r / (‖T‖ + 1)` of `z` lie in the preimage under `T`
  of a stay set of `T ∘ g ∘ T⁻¹`, null by `exists_measure_stayNear_eq_zero`,
  and `T` preserves null sets (`exists_measure_stayNear_eq_zero_of_conj`).
* The Lindelöf argument of `measure_inter_tendsto_symmetricUnstable_eq_zero`
  uses of the unstable points only that each has a null stay set
  (`measure_inter_tendsto_eq_zero_of_stayNear`), and gives
  `measure_tendsto_symmetrizableUnstable_eq_zero`.

This is the proof of `thm:main-general` in §4 of arXiv:1710.07406 for fixed
points with a symmetrizable derivative.
-/

@[expose] public section

open Set Metric Filter MeasureTheory
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `z` is a fixed point of `g` at which the derivative of `g` is conjugate,
by a continuous linear automorphism `T`, to a symmetric map, and has an
eigenvalue of modulus greater than one.

The unstable fixed points of §4 of arXiv:1710.07406 are the fixed points whose
derivative has an eigenvalue of modulus greater than one; this predicate adds
that the derivative be conjugate to a symmetric map, which holds for the
time-one map of `ẋ = -G(x) ∇f(x)` at a critical point where `G` is symmetric
positive definite. -/
def IsSymmetrizableUnstable (g : E → E) (z : E) : Prop :=
  g z = z ∧ (∃ T : E ≃L[ℝ] E,
    ((T : E →L[ℝ] E) ∘L fderiv ℝ g z ∘L (T.symm : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric) ∧
    ∃ μ : ℝ, 1 < |μ| ∧ Module.End.HasEigenvalue (fderiv ℝ g z : E →ₗ[ℝ] E) μ

/-- A symmetric unstable fixed point is a symmetrizable one, with `T = id`. -/
theorem IsSymmetricUnstable.isSymmetrizableUnstable {g : E → E} {z : E}
    (h : IsSymmetricUnstable g z) : IsSymmetrizableUnstable g z := by
  obtain ⟨hz, hsym, hev⟩ := h
  refine ⟨hz, ⟨ContinuousLinearEquiv.refl ℝ E, ?_⟩, hev⟩
  simpa only [ContinuousLinearEquiv.coe_refl, ContinuousLinearEquiv.refl_symm,
    ContinuousLinearMap.id_comp, ContinuousLinearMap.comp_id] using hsym

/-- **Conjugating a map conjugates its iterates**:
`(T ∘ g ∘ T⁻¹)^[k] (T x) = T (g^[k] x)`. -/
theorem conj_iterate (T : E ≃L[ℝ] E) (g : E → E) (k : ℕ) (x : E) :
    (T ∘ g ∘ T.symm)^[k] (T x) = T (g^[k] x) :=
  ((Function.Semiconj.iterate_right (f := T) (ga := g) (gb := T ∘ g ∘ T.symm)
    (fun y => by simp) k) x).symm

/-- **The derivative of a conjugate**: `T ∘ g ∘ T⁻¹` has the derivative
`T Dg(z) T⁻¹` at `T z`. -/
theorem hasFDerivAt_conj (T : E ≃L[ℝ] E) {g : E → E} {z : E} (hg : DifferentiableAt ℝ g z) :
    HasFDerivAt (T ∘ g ∘ T.symm)
      ((T : E →L[ℝ] E) ∘L fderiv ℝ g z ∘L (T.symm : E →L[ℝ] E)) (T z) := by
  have hg' : HasFDerivAt g (fderiv ℝ g z) (T.symm (T z)) := by
    rw [T.symm_apply_apply]
    exact hg.hasFDerivAt
  exact T.hasFDerivAt.comp (T z) (hg'.comp (T z) T.symm.hasFDerivAt)

variable [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/-- **The local stable set of a point with a symmetrizable, somewhere
stretching derivative is null.**  If `g` is `C¹`, `T Dg(z) T⁻¹` is symmetric
and `Dg(z)` has an eigenvalue `λ`, `|λ| > 1`, then for some `r > 0` the starts
whose forward orbit stays in `B̄(z, r)` form a null set. -/
theorem exists_measure_stayNear_eq_zero_of_conj (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) {z : E} (T : E ≃L[ℝ] E)
    (hsym : ((T : E →L[ℝ] E) ∘L fderiv ℝ g z ∘L (T.symm : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric)
    {lam : ℝ} (hlam : 1 < |lam|)
    (hev : Module.End.HasEigenvalue (fderiv ℝ g z : E →ₗ[ℝ] E) lam) :
    ∃ r > 0, μ (stayNear g z r) = 0 := by
  have hh : ContDiff ℝ 1 (T ∘ g ∘ T.symm) := T.contDiff.comp (hg.comp T.symm.contDiff)
  have hd := (hasFDerivAt_conj T (hg.differentiable one_ne_zero z)).fderiv
  obtain ⟨v, hv⟩ := hev.exists_hasEigenvector
  have hgv : fderiv ℝ g z v = lam • v := hv.apply_eq_smul
  have hev' : Module.End.HasEigenvalue (fderiv ℝ (T ∘ g ∘ T.symm) (T z) : E →ₗ[ℝ] E) lam := by
    refine Module.End.hasEigenvalue_of_hasEigenvector (x := T v) ⟨?_, ?_⟩
    · rw [Module.End.mem_eigenspace_iff, hd]
      simp [hgv]
    · simpa using hv.2
  obtain ⟨r, hr, hnull⟩ := exists_measure_stayNear_eq_zero μ hh (hd ▸ hsym) hlam hev'
  set c := ‖(T : E →L[ℝ] E)‖
  refine ⟨r / (c + 1), div_pos hr (by positivity),
    measure_mono_null (t := T ⁻¹' stayNear (T ∘ g ∘ T.symm) (T z) r) (fun x hx k => ?_) ?_⟩
  · have hk := mem_closedBall.1 (hx k)
    rw [conj_iterate, mem_closedBall, dist_eq_norm, ← map_sub]
    rw [dist_eq_norm] at hk
    calc ‖T (g^[k] x - z)‖ ≤ c * ‖g^[k] x - z‖ := (T : E →L[ℝ] E).le_opNorm _
      _ ≤ c * (r / (c + 1)) := by gcongr
      _ ≤ (c + 1) * (r / (c + 1)) := by gcongr; linarith
      _ = r := mul_div_cancel₀ r (by positivity)
  · rw [Measure.addHaar_preimage_continuousLinearEquiv, hnull, mul_zero]

/-- **The starts converging to points with null stay sets form a null set.**
Let `g` be `C¹`, map `S` into itself, and have a derivative of nonzero
determinant on `S`. If every point with the property `P` has a null stay set
`stayNear g z r` for some `r > 0`, the starts in `S` whose orbit converges to
such a point form a null set: countably many of the balls `B(z, r_z)` cover
these points (Lindelöf), and an orbit converging into one of them lies from
some step `m` on in the stay set. -/
theorem measure_inter_tendsto_eq_zero_of_stayNear (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) {S : Set E} (hS : MapsTo g S S)
    (hdet : ∀ x ∈ S, (fderiv ℝ g x).det ≠ 0) {P : E → Prop}
    (hP : ∀ z, P z → ∃ r > 0, μ (stayNear g z r) = 0) :
    μ (S ∩ {x | ∃ z, P z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)}) = 0 := by
  choose r hr hnull using fun z : {z // P z} => hP z z.2
  obtain ⟨T, hTc, hTU⟩ := TopologicalSpace.isOpen_iUnion_countable
    (fun z : {z // P z} => ball (z : E) (r z)) fun _ => isOpen_ball
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

/-- **Maps almost never converge to a symmetrizable unstable point from a
forward-invariant set.**  If `g` is `C¹`, maps `S` into itself, and its
derivative has nonzero determinant at every point of `S`, the starts in `S`
whose orbit converges to a symmetrizable unstable fixed point of `g` form a
null set. -/
theorem measure_inter_tendsto_symmetrizableUnstable_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] {g : E → E} (hg : ContDiff ℝ 1 g) {S : Set E} (hS : MapsTo g S S)
    (hdet : ∀ x ∈ S, (fderiv ℝ g x).det ≠ 0) :
    μ (S ∩ {x | ∃ z, IsSymmetrizableUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)}) =
      0 :=
  measure_inter_tendsto_eq_zero_of_stayNear μ hg hS hdet fun _ ⟨_, ⟨T, hsym⟩, _, hlam, hev⟩ =>
    exists_measure_stayNear_eq_zero_of_conj μ hg T hsym hlam hev

/-- **Maps almost never converge to a symmetrizable unstable point.**  If `g`
is `C¹` and its derivative has nonzero determinant everywhere, the starts
whose orbit converges to a fixed point of `g` at which the derivative is
conjugate to a symmetric map and has an eigenvalue of modulus greater than one
form a null set: `thm:main-general` of arXiv:1710.07406 for these points. -/
theorem measure_tendsto_symmetrizableUnstable_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) :
    μ {x | ∃ z, IsSymmetrizableUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)} = 0 := by
  simpa using measure_inter_tendsto_symmetrizableUnstable_eq_zero μ hg (mapsTo_univ g univ)
    fun x _ => hdet x

/-- **The same for any initial law with a density.** -/
theorem measure_tendsto_symmetrizableUnstable_eq_zero_of_absolutelyContinuous
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) :
    ν {x | ∃ z, IsSymmetrizableUnstable g z ∧ Tendsto (fun k => g^[k] x) atTop (𝓝 z)} = 0 :=
  hν (measure_tendsto_symmetrizableUnstable_eq_zero μ hg hdet)

/-- The hypotheses of the theorems above hold for the doubling map `g` of the
line, with Lebesgue measure, its restriction to the unit ball, `S = {0}`,
`P` the symmetrizable unstable points of `g`, and at `0` the map `T = id`, the
eigenvalue `2` and determinant `2` of the derivative. -/
example : let g := fun x : EuclideanSpace ℝ (Fin 1) => (2 : ℝ) • x
    ContDiff ℝ 1 g ∧ IsSymmetrizableUnstable g 0 ∧ (∀ x, (fderiv ℝ g x).det ≠ 0) ∧
      (volume : Measure (EuclideanSpace ℝ (Fin 1))).restrict (ball 0 1) ≪ volume ∧
      MapsTo g {0} {0} ∧
      ∀ z, IsSymmetrizableUnstable g z → ∃ r > 0, volume (stayNear g z r) = 0 := by
  intro g
  have hd : ∀ x, fderiv ℝ g x = (2 : ℝ) • ContinuousLinearMap.id ℝ _ := fun x =>
    ((ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin 1))).hasFDerivAt.const_smul (2 : ℝ)).fderiv
  have hg : ContDiff ℝ 1 g := contDiff_const_smul 2
  have h0 : IsSymmetrizableUnstable g 0 := by
    refine IsSymmetricUnstable.isSymmetrizableUnstable
      ⟨by simp [g], fun u v => ?_, 2, by norm_num, ?_⟩
    · simp [hd, real_inner_smul_left, real_inner_smul_right]
    · refine Module.End.hasEigenvalue_of_hasEigenvector (x := EuclideanSpace.single 0 1) ⟨?_, ?_⟩
      · simp [hd]
      · simp
  refine ⟨hg, h0, fun x => ?_, Measure.absolutelyContinuous_of_le Measure.restrict_le_self,
    fun x hx => by simp_all [g], fun z ⟨_, ⟨T, hsym⟩, _, hlam, hev⟩ =>
      exists_measure_stayNear_eq_zero_of_conj volume hg T hsym hlam hev⟩
  rw [hd, ContinuousLinearMap.det, ContinuousLinearMap.toLinearMap_smul,
    ContinuousLinearMap.coe_id, LinearMap.det_smul, LinearMap.det_id]
  simp

end Transformer.Landscape.SaddleAvoidance
