module

public import Transformer.Perceptron.Section6_SphereUniqueness
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Actual vector-field uniqueness and constancy on the sphere

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (a),
uses analytic uniqueness for the stationary vector field, followed by
constancy of the first variation on the sphere. These two steps are
proved here from the preceding scalar spherical uniqueness theorem
and the actual orthogonal paths.

Vector-field uniqueness tests the original field against fixed inner
product functionals. A positive measure of actual vector zeros is a
positive measure of scalar zeros for every such functional. Applying
the scalar theorem and then the inner-product separation property
gives zero for the original vector field on the whole sphere.

For constancy, the supplied covector is the actual Fréchet derivative
of the original function. Along the actual orthogonal path the squared
norm is constant, so its actual velocity is tangent. A zero tangent
projection therefore makes the scalar derivative along that path zero.
The mean-value theorem compares the original endpoint values.
All derivative and uniform-law hypotheses have actual witnesses below.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Positive original uniform measure of actual vector zeros forces
zero on the entire sphere. Source: arXiv:2601.21366v2, §6,
the vector-field uniqueness step of `prop: unified.log`, part (a).
Both the domain dimension and the vector codomain are arbitrary. -/
theorem vector_eq_zero_on_sphere_of_positive_zero_measure {d m : ℕ} (hd : 2 ≤ d)
    (F : EucSpace d → EucSpace m) (hF : AnalyticOnNhd ℝ F univ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν {x : SSphere d | F (x : EucSpace d) = 0}) :
    ∀ x : SSphere d, F (x : EucSpace d) = 0 := by
  intro x
  let v := F (x : EucSpace d)
  let f : EucSpace d → ℝ := fun z => inner (𝕜 := ℝ) v (F z)
  have hf : AnalyticOnNhd ℝ f univ := fun z hz =>
    ((innerSL ℝ v).analyticAt (F z)).comp (hF z hz)
  have hposf : 0 < ν {z : SSphere d | f (z : EucSpace d) = 0} :=
    hpos.trans_le (measure_mono (fun z hz => by
      change inner (𝕜 := ℝ) v (F (z : EucSpace d)) = 0
      rw [hz, inner_zero_right]))
  have hx := eq_zero_on_sphere_of_positive_zero_measure hd f hf ν hν hposf x
  exact inner_self_eq_zero.mp hx

/-- A nonzero actual analytic vector field has a uniform-null zero set.
Source: arXiv:2601.21366v2, §6, the null-support argument
of `prop: unified.log`, part (a), and `rem:unifiedlog-B`. -/
theorem measure_sphere_vector_zeros_eq_zero_of_analytic_nonzero {d m : ℕ} (hd : 2 ≤ d)
    (F : EucSpace d → EucSpace m) (hF : AnalyticOnNhd ℝ F univ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hne : ∃ x : SSphere d, F (x : EucSpace d) ≠ 0) :
    ν {x : SSphere d | F (x : EucSpace d) = 0} = 0 := by
  by_contra h
  obtain ⟨x, hx⟩ := hne
  exact hx (vector_eq_zero_on_sphere_of_positive_zero_measure hd F hF ν hν
    (pos_iff_ne_zero.mpr h) x)

/-- The vector uniqueness hypotheses hold jointly for the actual uniform
circle law and the zero vector field, whose actual zero set is the whole
sphere. Source: arXiv:2601.21366v2, §6, `prop: unified.log`. -/
example : (2 : ℕ) ≤ 2 ∧ AnalyticOnNhd ℝ (fun _ : EucSpace 2 => (0 : EucSpace 2)) univ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2)
      {x : SSphere 2 | (fun _ : EucSpace 2 => (0 : EucSpace 2)) x = 0} := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨le_rfl, fun _ _ => analyticAt_const, hν, by simp⟩

/-- The nonzero-vector input is realized by the identity field and an
actual unit vector. Source: arXiv:2601.21366v2, §6, the null-zero-set
alternative used for the original stationary branch field. -/
example : AnalyticOnNhd ℝ (fun z : EucSpace 2 => z) univ ∧
    ∃ x : SSphere 2, (fun z : EucSpace 2 => z) (x : EucSpace 2) ≠ 0 := by
  refine ⟨fun _ _ => analyticAt_id, basePoint 1, ?_⟩
  apply norm_ne_zero_iff.mp
  rw [norm_eq_of_mem_sphere]
  exact one_ne_zero

/-- A zero actual spherical gradient makes the original scalar function
constant on the sphere. Source: arXiv:2601.21366v2, §6,
the constancy step leading to `eq:log-identity-global` and
`eq:ratio-const-correct`. The derivative identity is explicit;
no gradient or constancy conclusion is assumed through a definition. -/
theorem eq_on_sphere_of_projected_derivative_zero {d : ℕ} (hd : 2 ≤ d)
    (f : EucSpace d → ℝ) (D : EucSpace d → EucSpace d)
    (hD : ∀ z, HasFDerivAt f (innerSL ℝ (D z)) z)
    (hproj : ∀ z : EucSpace d, ‖z‖ = 1 → proj d z (D z) = 0) :
    ∀ x y : SSphere d, f (x : EucSpace d) = f (y : EucSpace d) := by
  intro x y
  obtain ⟨u, v, hne, hendpoint⟩ := exists_reflectionPath_between_sphere_points hd x y
  let γ : ℝ → EucSpace d := fun t => reflectionPath u v t (x : EucSpace d)
  have hunit (t : ℝ) : ‖γ t‖ = 1 :=
    (LinearIsometryEquiv.norm_map _ _).trans (norm_eq_of_mem_sphere x)
  have hcurve (t : ℝ) : HasDerivAt γ (deriv γ t) t :=
    (analyticOnNhd_reflectionPath_orbit u v hne (x : EucSpace d) t (mem_univ t)).differentiableAt.hasDerivAt
  have hnorm : (fun t => ‖γ t‖ ^ 2) = fun _ => (1 : ℝ) := funext (fun t => by rw [hunit t, one_pow])
  have hderiv (t : ℝ) : HasDerivAt (fun s => f (γ s)) 0 t := by
    have hn := (hcurve t).norm_sq
    rw [hnorm] at hn
    have hz := hn.unique (hasDerivAt_const t (1 : ℝ))
    have htangent : inner (𝕜 := ℝ) (γ t) (deriv γ t) = 0 := by linarith
    have hp : inner (𝕜 := ℝ) (proj d (γ t) (D (γ t))) (deriv γ t) = 0 := by
      rw [hproj (γ t) (hunit t), inner_zero_left]
    rw [proj, inner_sub_left, real_inner_smul_left, htangent, mul_zero, sub_zero] at hp
    exact ((hD (γ t)).comp_hasDerivAt t (hcurve t)).congr_deriv (by
      rw [innerSL_apply_apply]
      exact hp)
  have heq := is_const_of_deriv_eq_zero
    (f := fun t => f (γ t)) (fun t => (hderiv t).differentiableAt)
    (fun t => (hderiv t).deriv) 0 1
  have hstart : γ 0 = (x : EucSpace d) := reflectionPath_zero_apply u v (x : EucSpace d)
  have hend : γ 1 = (y : EucSpace d) := congrArg Subtype.val hendpoint
  rwa [hstart, hend] at heq

/-- All constancy inputs hold for the nonzero constant scalar function
at level one and its actual zero derivative field in dimension two.
Source: arXiv:2601.21366v2, §6, the constant first-variation alternative. -/
example : (2 : ℕ) ≤ 2 ∧
    (∀ z : EucSpace 2, HasFDerivAt (fun _ : EucSpace 2 => (1 : ℝ))
      (innerSL ℝ ((fun _ : EucSpace 2 => (0 : EucSpace 2)) z)) z) ∧
    ∀ z : EucSpace 2, ‖z‖ = 1 → proj 2 z ((fun _ : EucSpace 2 => (0 : EucSpace 2)) z) = 0 := by
  refine ⟨le_rfl, ?_, fun z _ => by simp [proj]⟩
  intro z
  apply (hasFDerivAt_const (1 : ℝ) z).congr_fderiv
  ext w
  simp

/-- The same constancy inputs also hold for the actual nonconstant
ambient function `‖z‖²` and its actual radial derivative `2z`.
It has constant value on the sphere while varying off it, so the
statement really concerns the original spherical restriction.
Source: arXiv:2601.21366v2, §6, the spherical first-variation gradient. -/
example : (∀ z : EucSpace 2, HasFDerivAt (fun y : EucSpace 2 => ‖y‖ ^ 2)
      (innerSL ℝ ((2 : ℝ) • z)) z) ∧
    ∀ z : EucSpace 2, ‖z‖ = 1 → proj 2 z ((2 : ℝ) • z) = 0 := by
  refine ⟨?_, fun z hz => proj_smul_self hz 2⟩
  intro z
  apply ((hasFDerivAt_id z).norm_sq).congr_fderiv
  ext w
  simp

end Transformer.Perceptron
