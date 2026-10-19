module

public import Transformer.Perceptron.Section6_ReflectionPaths
public import Transformer.Perceptron.Section6_RotationZeroSets
public import Transformer.Perceptron.Hyperplane
public import Transformer.Metastability.Section4_UniformLaw

/-!
# Positive-measure analytic uniqueness on the actual sphere

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (a),
uses the fact that a nonzero real-analytic spherical function has a
uniform-null zero set. We prove the required ambient-analytic case
for every dimension at least two and every original invariant law.

Pick a support point of the actual zero-set restriction. To reach any
other sphere point, compose two Householder reflections: the first
normal is a unit vector orthogonal to the starting point, and the
second normal is the difference of the two points. Their affine
normal never vanishes, because its inner product with the starting
point has a strictly positive coefficient. The resulting full real
path is analytic and preserves the original spherical measure.

The preceding Fubini theorem then makes the starting support point's
entire actual orbit zero. Its endpoint is the arbitrary target point.
Thus a positive-measure zero set forces zero on the whole sphere.
The ambient function need not vanish off the sphere; the conclusion
does not silently replace spherical uniqueness by ambient uniqueness.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Any two actual sphere points in dimension at least two are joined
by the original reflection path with a normal nonzero at every real time.
Source: arXiv:2601.21366v2, §6, the connected spherical uniqueness
argument in part (a) of `prop: unified.log`.
The constructed path starts at the identity as already proved. -/
theorem exists_reflectionPath_between_sphere_points {d : ℕ} (hd : 2 ≤ d)
    (x y : SSphere d) : ∃ u v : EucSpace d,
      (∀ t, reflectionNormal u v t ≠ 0) ∧
      Perspective.sphereMap d (reflectionPath u v 1) x = y := by
  have hx : ‖(x : EucSpace d)‖ = 1 := norm_eq_of_mem_sphere x
  have hy : ‖(y : EucSpace d)‖ = 1 := norm_eq_of_mem_sphere y
  have hx₀ : (x : EucSpace d) ≠ 0 := by
    intro h
    rw [h, norm_zero] at hx
    exact zero_ne_one hx
  by_cases hxy : x = y
  · subst y
    refine ⟨(x : EucSpace d), (x : EucSpace d), ?_, Subtype.ext ?_⟩
    · intro t
      rw [reflectionNormal, ← add_smul, sub_add_cancel, one_smul]
      exact hx₀
    · simp only [Perspective.sphereMap, reflectionPath, reflectionNormal,
        sub_self, zero_smul, one_smul, zero_add, LinearIsometryEquiv.trans_apply]
      exact Submodule.reflection_involutive ((ℝ ∙ (x : EucSpace d))ᗮ) (x : EucSpace d)
  have hvec : (x : EucSpace d) ≠ (y : EucSpace d) := fun h => hxy (Subtype.ext h)
  have hpos : 0 < inner (𝕜 := ℝ) (x : EucSpace d) ((x : EucSpace d) - (y : EucSpace d)) := by
    have hn := sq_pos_of_pos (norm_pos_iff.mpr (sub_ne_zero.mpr hvec))
    have hs := norm_sub_sq_real (x : EucSpace d) (y : EucSpace d)
    rw [hx, hy] at hs
    rw [inner_sub_right, real_inner_self_eq_norm_sq, hx]
    nlinarith
  obtain ⟨u, hu, hxu⟩ := exists_unit_inner_eq_zero hd hx₀
  have hnormal : ∀ t, reflectionNormal u ((x : EucSpace d) - (y : EucSpace d)) t ≠ 0 := by
    intro t ht
    have h := congrArg (fun z : EucSpace d => inner (𝕜 := ℝ) (x : EucSpace d) z) ht
    simp only [reflectionNormal, inner_add_right, real_inner_smul_right, hxu,
      mul_zero, zero_add, inner_zero_right] at h
    have ht₀ : t = 0 := (mul_eq_zero.mp h).resolve_right hpos.ne'
    subst t
    simp only [reflectionNormal, sub_zero, one_smul, zero_smul, add_zero] at ht
    exact one_ne_zero (by rw [← hu, ht, norm_zero])
  have hfix : (ℝ ∙ u)ᗮ.reflection (x : EucSpace d) = (x : EucSpace d) :=
    Submodule.reflection_mem_subspace_eq_self
      (Submodule.mem_orthogonal_singleton_iff_inner_left.mpr hxu)
  refine ⟨u, (x : EucSpace d) - (y : EucSpace d), hnormal, Subtype.ext ?_⟩
  change reflectionPath u ((x : EucSpace d) - (y : EucSpace d)) 1 (x : EucSpace d) = _
  rw [reflectionPath, LinearIsometryEquiv.trans_apply, hfix]
  simp only [reflectionNormal, sub_self, zero_smul, one_smul, zero_add]
  exact Submodule.reflection_sub (hx.trans hy.symm)

/-- The path's dimension input is realized with two distinct actual
unit points on the circle. Source: arXiv:2601.21366v2, §6,
the spherical uniqueness argument of `prop: unified.log`. -/
example : (2 : ℕ) ≤ 2 ∧ ∃ x y : SSphere 2, x ≠ y := by
  refine ⟨le_rfl, basePoint 1,
    Perspective.sphereMap 2 (LinearIsometryEquiv.neg ℝ) (basePoint 1), ?_⟩
  intro h
  have h₀ := congrArg (fun z : SSphere 2 => (z : EucSpace 2) 0) h
  norm_num [Perspective.sphereMap, basePoint] at h₀

/-- A positive original uniform measure of actual spherical zeros
forces the ambient-analytic function to vanish on the whole sphere.
Source: arXiv:2601.21366v2, §6, the positive-measure uniqueness
step in part (a) of `prop: unified.log` and `rem:unifiedlog-B`.
No uniform density, chart Jacobian or transitivity hypothesis is assumed. -/
theorem eq_zero_on_sphere_of_positive_zero_measure {d : ℕ} (hd : 2 ≤ d)
    (f : EucSpace d → ℝ) (hf : AnalyticOnNhd ℝ f univ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν {x : SSphere d | f (x : EucSpace d) = 0}) :
    ∀ x : SSphere d, f (x : EucSpace d) = 0 := by
  have : IsProbabilityMeasure ν := hν.1
  let S : Set (SSphere d) := {x | f (x : EucSpace d) = 0}
  have hrest : ν.restrict S ≠ 0 := by
    intro h
    have hz : ν S = 0 := by
      simpa using congrArg (fun μ : Measure (SSphere d) => μ univ) h
    exact hpos.ne' hz
  obtain ⟨x₀, hx₀⟩ := Measure.nonempty_support hrest
  intro y
  obtain ⟨u, v, hne, hendpoint⟩ := exists_reflectionPath_between_sphere_points hd x₀ y
  have hc : Continuous (fun x : SSphere d => f (x : EucSpace d)) :=
    hf.continuous.comp continuous_subtype_val
  have hzero (x : SSphere d) : Perspective.sphereMap d (reflectionPath u v 0) x = x :=
    Subtype.ext (reflectionPath_zero_apply u v (x : EucSpace d))
  have hana (x : SSphere d) : AnalyticOnNhd ℝ
      (fun t => f (Perspective.sphereMap d (reflectionPath u v t) x : EucSpace d)) univ := by
    intro t ht
    exact (hf _ (mem_univ _)).comp (analyticOnNhd_reflectionPath_orbit u v hne (x : EucSpace d) t ht)
  have horbit := support_zero_restrict_subset_zero_orbits ν _ hc
    (fun t x => Perspective.sphereMap d (reflectionPath u v t) x)
    (continuous_reflectionPath_sphere u v hne) hzero (fun t => hν.2 _) hana hx₀
  have h := horbit 1
  rwa [hendpoint] at h

/-- A nonzero ambient-analytic restriction has an actual uniform-null
zero set on the sphere. Source: arXiv:2601.21366v2, §6,
the contrapositive used in `prop: unified.log`, part (a). -/
theorem measure_sphere_zeros_eq_zero_of_analytic_nonzero {d : ℕ} (hd : 2 ≤ d)
    (f : EucSpace d → ℝ) (hf : AnalyticOnNhd ℝ f univ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hne : ∃ x : SSphere d, f (x : EucSpace d) ≠ 0) :
    ν {x : SSphere d | f (x : EucSpace d) = 0} = 0 := by
  by_contra h
  obtain ⟨x, hx⟩ := hne
  exact hx (eq_zero_on_sphere_of_positive_zero_measure hd f hf ν hν (pos_iff_ne_zero.mpr h) x)

/-- The positive-zero-measure inputs are jointly realized by the actual
uniform circle law and the zero scalar function. Source:
arXiv:2601.21366v2, §6, positive-measure uniqueness in `prop: unified.log`. -/
example : (2 : ℕ) ≤ 2 ∧ AnalyticOnNhd ℝ (fun _ : EucSpace 2 => (0 : ℝ)) univ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) {x : SSphere 2 | (fun _ : EucSpace 2 => (0 : ℝ)) x = 0} := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨le_rfl, fun _ _ => analyticAt_const, hν, by simp⟩

/-- The nonzero-function input is realized by the original first-coordinate
inner-product functional, which equals one at an actual unit point.
Source: arXiv:2601.21366v2, §6, the null-zero-set alternative. -/
example : AnalyticOnNhd ℝ
      (fun z : EucSpace 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) z) univ ∧
    ∃ x : SSphere 2, inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (x : EucSpace 2) ≠ 0 := by
  refine ⟨fun z _ => (innerSL ℝ (basePoint 1 : EucSpace 2)).analyticAt z, basePoint 1, ?_⟩
  rw [real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere]
  norm_num

end Transformer.Perceptron
