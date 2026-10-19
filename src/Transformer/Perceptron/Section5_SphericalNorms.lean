module

public import Transformer.Perceptron.Section5_SphericalInteraction
public import Mathlib.MeasureTheory.Integral.Bochner.Set

/-!
# Original L¹ and L² estimates for localized spherical fields

ArXiv:2601.21366v2, §5.3, the localization estimate in
`thm: any.d` (i), extending the Cauchy–Schwarz step of §5.2.
Integrating the actual nonnegative variance proves the finite-
measure inequality `(∫ f)² ≤ μ(univ) ∫ f²`. Restricting the
original measure to the actual field's support neighborhood
gives its small-mass factor, with the original L² norm retained.

A continuous pair kernel bounded by the product of the two field
norms has integral bounded by the square of the original L¹ norm.
These two estimates combine with the proved mixed Hessian formula.
Every integral below is justified on the compact original space;
there is no assumption about a density or finite atomic support.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Finite-measure Cauchy–Schwarz from the actual nonnegative variance.
Source: arXiv:2601.21366v2, §5.3, the localized version of the
L¹–L² estimate in §5.2, the proof of `thm: circle.gelu`.
Positive total real mass is sufficient; no probability normalization is used. -/
theorem sq_integral_le_mass_mul_integral_sq {α : Type*} [TopologicalSpace α]
    [CompactSpace α] [MeasurableSpace α] [OpensMeasurableSpace α]
    (μ : Measure α) [IsFiniteMeasure μ] (f : α → ℝ) (hf : Continuous f)
    (hmass : 0 < μ.real univ) :
    (∫ x, f x ∂μ) ^ 2 ≤ μ.real univ * (∫ x, f x ^ 2 ∂μ) := by
  let m : ℝ := μ.real univ
  let I : ℝ := ∫ x, f x ∂μ
  let c : ℝ := I / m
  have hm : 0 < m := hmass
  have hmc : m * c = I := by
    dsimp only [c]
    exact mul_div_cancel₀ I hm.ne'
  have hIf : Integrable f μ := hf.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hIsq : Integrable (fun x => f x ^ 2) μ :=
    (hf.pow 2).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hIscale := hIf.const_mul (2 * c)
  have hIdiff : Integrable (fun x => f x ^ 2 - (2 * c) * f x) μ := hIsq.sub hIscale
  have hexpand : (fun x => (f x - c) ^ 2) = fun x => f x ^ 2 - (2 * c) * f x + c ^ 2 := by
    funext x
    ring
  have hvar : (∫ x, (f x - c) ^ 2 ∂μ) =
      (∫ x, f x ^ 2 ∂μ) - (2 * c) * I + m * c ^ 2 := by
    rw [hexpand, integral_add hIdiff (integrable_const _),
      integral_sub hIsq hIscale, integral_const_mul, integral_const]
    simp only [smul_eq_mul]
    rfl
  have hnonneg : 0 ≤ (∫ x, f x ^ 2 ∂μ) - (2 * c) * I + m * c ^ 2 := by
    rw [← hvar]
    exact integral_nonneg fun x => sq_nonneg (f x - c)
  have hmul : 0 ≤ m * (∫ x, f x ^ 2 ∂μ) - I ^ 2 := by
    calc
      0 ≤ m * ((∫ x, f x ^ 2 ∂μ) - (2 * c) * I + m * c ^ 2) := mul_nonneg hm.le hnonneg
      _ = m * (∫ x, f x ^ 2 ∂μ) - I ^ 2 := by rw [← hmc]; ring
  change I ^ 2 ≤ m * (∫ x, f x ^ 2 ∂μ)
  linarith

/-- A localized original field has its L¹ norm squared bounded by the
neighborhood's original mass times its original squared L² norm.
Source: arXiv:2601.21366v2, §5.3, the small-support Hessian estimate. -/
theorem sq_integral_norm_le_local_mass {d : ℕ} (μ : Perspective.ProbSphere d)
    (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) (U : Set (SSphere d))
    (hsupp : Function.support ξ ⊆ U) (hmass : 0 < (μ : Measure (SSphere d)).real U) :
    (∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d))) ^ 2 ≤
      (μ : Measure (SSphere d)).real U * (∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d))) := by
  have hz (x : SSphere d) (hx : x ∉ U) : ξ x = 0 := by
    by_contra h
    exact hx (hsupp (Function.mem_support.mpr h))
  have hnorm : (∫ x in U, ‖ξ x‖ ∂(μ : Measure (SSphere d))) =
      ∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d)) :=
    setIntegral_eq_integral_of_forall_compl_eq_zero fun x hx => by rw [hz x hx, norm_zero]
  have hsq : (∫ x in U, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d))) =
      ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) :=
    setIntegral_eq_integral_of_forall_compl_eq_zero fun x hx => by rw [hz x hx, norm_zero]; norm_num
  have hmass' : 0 < ((μ : Measure (SSphere d)).restrict U).real univ := by
    simpa only [measureReal_def, Measure.restrict_apply_univ] using hmass
  have h := sq_integral_le_mass_mul_integral_sq ((μ : Measure (SSphere d)).restrict U)
    (fun x => ‖ξ x‖) hcξ.norm hmass'
  rw [hnorm, hsq] at h
  simpa only [measureReal_def, Measure.restrict_apply_univ] using h

/-- A continuous original pair kernel bounded by a product of field norms
has integral bounded by the square of the original L¹ norm.
Source: arXiv:2601.21366v2, §5.3, the mixed Hessian estimate
extending `eq:Hess-bilinear-analytic` of §5.2. -/
theorem abs_integral_pair_le_sq_integral_norm {d : ℕ} (μ : Perspective.ProbSphere d)
    (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ)
    (K : SSphere d × SSphere d → ℝ) (hK : Continuous K) (C : ℝ)
    (hbound : ∀ p : SSphere d × SSphere d, |K p| ≤ C * ‖ξ p.1‖ * ‖ξ p.2‖) :
    |∫ p, K p ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))| ≤
      C * (∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d))) ^ 2 := by
  have hI : Integrable K ((μ : Measure (SSphere d)).prod μ) :=
    hK.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hR : Integrable (fun p : SSphere d × SSphere d => C * ‖ξ p.1‖ * ‖ξ p.2‖)
      ((μ : Measure (SSphere d)).prod μ) :=
    (((hcξ.norm.comp continuous_fst).const_mul C).mul
      (hcξ.norm.comp continuous_snd)).integrable_of_hasCompactSupport
        (HasCompactSupport.of_compactSpace _)
  calc
    _ = ‖∫ p, K p ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))‖ :=
      (Real.norm_eq_abs _).symm
    _ ≤ ∫ p, ‖K p‖ ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
      norm_integral_le_integral_norm _
    _ = ∫ p, |K p| ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := by
      simp only [Real.norm_eq_abs]
    _ ≤ ∫ p, C * ‖ξ p.1‖ * ‖ξ p.2‖
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := integral_mono hI.abs hR hbound
    _ = _ := by
      simp_rw [mul_assoc]
      rw [integral_const_mul, integral_prod_mul
        (fun x : SSphere d => ‖ξ x‖) (fun x : SSphere d => ‖ξ x‖)]
      ring

/-- Continuous nonzero functions and positive finite mass jointly witness
the variance inputs on the compact space `Unit` with its actual Dirac law. -/
example : Continuous (fun _ : Unit => (1 : ℝ)) ∧
    0 < (Measure.dirac () : Measure Unit).real univ := by
  exact ⟨continuous_const, by simp⟩

/-- The nonzero unit circle tangent, full neighborhood and constant unit
kernel jointly realize the continuity, localization and pair-bound
inputs on an original Dirac probability. -/
example : Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    Function.support (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ⊆ univ ∧
    0 < (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).real univ ∧
    Continuous (fun _ : SSphere 2 × SSphere 2 => (1 : ℝ)) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ p : SSphere 2 × SSphere 2, |(1 : ℝ)| ≤
      1 * ‖circleTangent (p.1 : EucSpace 2)‖ * ‖circleTangent (p.2 : EucSpace 2)‖) := by
  refine ⟨continuous_circleTangent.comp continuous_subtype_val, subset_univ _, by simp,
    continuous_const, zero_le_one, fun p => ?_⟩
  rw [norm_circleTangent, norm_circleTangent, norm_eq_of_mem_sphere p.1,
    norm_eq_of_mem_sphere p.2]
  norm_num

/-- The nonzero unit tangent makes the probability L¹–L² estimate an
equality, checking that the retained norm is the original field norm. -/
example (μ : Perspective.ProbSphere 2) :
    (∫ x, ‖circleTangent (x : EucSpace 2)‖ ∂(μ : Measure (SSphere 2))) ^ 2 =
      ∫ x, ‖circleTangent (x : EucSpace 2)‖ ^ 2 ∂(μ : Measure (SSphere 2)) := by
  simp only [norm_circleTangent, norm_eq_of_mem_sphere, one_pow, integral_const,
    probReal_univ, one_smul]

end Transformer.Perceptron
