/-
# Expanding the actual circle Hessian

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`. Symmetry of the
actual pair curvature and Fubini's theorem expand the squared
speed difference. The two squared-speed terms agree after
swapping the original product measure. The remaining mixed term
is the integral operator appearing in the source.

The local coefficient is the continuous, actual first-variation
curvature from `Section5_ContinuousCurvature`. Both the second
derivative identity and its strict SOPD lower bound are established
for arbitrary probability measures, before any support reduction.
-/

import Transformer.Perceptron.Section5_GeneralHessian
import Transformer.Perceptron.Section5_ContinuousCurvature

open Real MeasureTheory

namespace Transformer.Perceptron

/-- A symmetric continuous pair kernel has the source's quadratic-form
expansion. Source: arXiv:2601.21366v2, §5.2, the algebra and symmetry
passing from `eq:Hess-bilinear.std` to `eq:Hess-bilinear`.
This step also holds for any finite measure on a compact space with
a countable base. -/
theorem integral_symmetric_pair_difference_sq {α : Type*} [TopologicalSpace α]
    [CompactSpace α] [SecondCountableTopology α] [MeasurableSpace α] [OpensMeasurableSpace α]
    (μ : Measure α) [IsFiniteMeasure μ] (K : α × α → ℝ) (hK : Continuous K)
    (hsymm : ∀ x y, K (y, x) = K (x, y)) (s : α → ℝ) (hs : Continuous s) :
    (1 / 2 : ℝ) * (∫ p, K p * (s p.1 - s p.2) ^ 2 ∂μ.prod μ) =
      (∫ x, (∫ y, K (x, y) ∂μ) * s x ^ 2 ∂μ) -
        (∫ p, K p * s p.1 * s p.2 ∂μ.prod μ) := by
  let f₁ : α × α → ℝ := fun p => K p * s p.1 ^ 2
  let f₂ : α × α → ℝ := fun p => K p * s p.2 ^ 2
  let g : α × α → ℝ := fun p => K p * s p.1 * s p.2
  have hf₁ : Integrable f₁ (μ.prod μ) :=
    (hK.mul ((hs.comp continuous_fst).pow 2)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hf₂ : Integrable f₂ (μ.prod μ) :=
    (hK.mul ((hs.comp continuous_snd).pow 2)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hg : Integrable g (μ.prod μ) :=
    ((hK.mul (hs.comp continuous_fst)).mul (hs.comp continuous_snd)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hswap : (fun p => f₁ p.swap) = f₂ := by
    funext p
    rcases p with ⟨x, y⟩
    dsimp [f₁, f₂]
    rw [hsymm x y]
  have h₂ : (∫ p, f₂ p ∂μ.prod μ) = ∫ p, f₁ p ∂μ.prod μ := by
    have h := integral_prod_swap (μ := μ) (ν := μ) f₁
    rw [hswap] at h
    exact h
  have h₁ : (∫ p, f₁ p ∂μ.prod μ) = ∫ x, (∫ y, K (x, y) ∂μ) * s x ^ 2 ∂μ := by
    calc
      _ = ∫ x, ∫ y, K (x, y) * s x ^ 2 ∂μ ∂μ :=
        (integral_integral (f := fun x y => K (x, y) * s x ^ 2) hf₁).symm
      _ = _ := integral_congr_ae (ae_of_all _ fun x => integral_mul_const _ _)
  have hexpand : (fun p : α × α => K p * (s p.1 - s p.2) ^ 2) =
      fun p => f₁ p + f₂ p - 2 * g p := by
    funext p
    dsimp [f₁, f₂, g]
    ring
  rw [hexpand, integral_sub (hf₁.fun_add hf₂) (hg.const_mul 2), integral_add hf₁ hf₂,
    integral_const_mul, h₂, h₁]
  dsimp only [g]
  ring

/-- All symmetry and continuity inputs are jointly realized by a constant
negative kernel and constant unit speed on the nonempty compact space `Unit`.
The finite measure can be the actual Dirac probability at `()`. -/
example : Continuous (fun _ : Unit × Unit => (-1 : ℝ)) ∧
    (∀ x y : Unit, (fun _ : Unit × Unit => (-1 : ℝ)) (y, x) =
      (fun _ : Unit × Unit => (-1 : ℝ)) (x, y)) ∧
    Continuous (fun _ : Unit => (1 : ℝ)) :=
  ⟨continuous_const, fun _ _ => rfl, continuous_const⟩

/-- The source's expanded expression for the actual circle coefficients.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
Every term retains the original measure and actual kernel curvature. -/
theorem circle_hessian_expanded (β : ℝ) (hβ : 0 < β) (σ σ' : ℝ → ℝ)
    (hσ : Continuous σ) (hσ' : Continuous σ') (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    (1 / 2 : ℝ) * (∫ p, circlePairCurvature β p.1 p.2 * (s p.1 - s p.2) ^ 2
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
      (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2 ∂(μ : Measure (SSphere 2))) =
    (∫ x, circleFirstVariationCurvature β σ σ' ω a μ x * s x ^ 2
      ∂(μ : Measure (SSphere 2))) -
      (∫ p, circlePairCurvature β p.1 p.2 * s p.1 * s p.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) := by
  have hK := integral_symmetric_pair_difference_sq (μ : Measure (SSphere 2))
    (fun p => circlePairCurvature β p.1 p.2) (continuous_circlePairCurvature β)
    (fun x y => circlePairCurvature_swap β y x) s hs
  have hKC := (continuous_integral_circlePairCurvature β hβ μ).mul (hs.pow 2)
  have hKI : Integrable (fun x => (∫ y, circlePairCurvature β x y
      ∂(μ : Measure (SSphere 2))) * s x ^ 2) (μ : Measure (SSphere 2)) :=
    hKC.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hCC := (continuous_circlePotentialCurvature σ σ' hσ hσ' ω a).mul (hs.pow 2)
  have hCI : Integrable (fun x => circlePotentialCurvature σ σ' ω a x * s x ^ 2)
      (μ : Measure (SSphere 2)) :=
    hCC.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hA : (∫ x, circleFirstVariationCurvature β σ σ' ω a μ x * s x ^ 2
      ∂(μ : Measure (SSphere 2))) =
      (∫ x, (∫ y, circlePairCurvature β x y ∂(μ : Measure (SSphere 2))) * s x ^ 2
        ∂(μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2 ∂(μ : Measure (SSphere 2))) := by
    simp only [circleFirstVariationCurvature, add_mul]
    exact integral_add hKI hCI
  rw [hK, hA]
  ring

/-- The expanded formula is the actual full-energy second derivative.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
No atomicity or cancellation assumption is used. -/
theorem hasDerivAt_deriv_energy_circleRotation_expanded (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    HasDerivAt (deriv (fun t => energy β φ ω a (μ.map (circleRotation s t))))
      ((∫ x, circleFirstVariationCurvature β σ σ' ω a μ x * s x ^ 2
        ∂(μ : Measure (SSphere 2))) -
        (∫ p, circlePairCurvature β p.1 p.2 * s p.1 * s p.2
          ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))) 0 := by
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have h := hasDerivAt_deriv_energy_circleRotation β hβ φ σ σ' hφ hσ hσ' ω a μ s hs
  exact h.congr_deriv (circle_hessian_expanded β hβ σ σ' hσc hσ' ω a μ s hs)

/-- Strict SOPD dominates the expanded Hessian on every actual smooth
gradient direction. Source: arXiv:2601.21366v2, §2.2, `eq:strict2order`,
and §5.2, `eq:Hess-bilinear`. The derivative exists for arbitrary support. -/
theorem strict_circle_hessian_coercive_expanded (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPD β φ σ ω a μ) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ s : SSphere 2 → ℝ, Continuous s →
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) →
      κ * (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) ≤
        (∫ x, circleFirstVariationCurvature β σ σ' ω a μ x * s x ^ 2
          ∂(μ : Measure (SSphere 2))) -
          (∫ p, circlePairCurvature β p.1 p.2 * s p.1 * s p.2
            ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) := by
  obtain ⟨κ, hκ, hH⟩ := strict_circle_hessian_coercive β hβ φ σ σ' hφ hσ hσ' ω a μ hμ
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  refine ⟨κ, hκ, fun s hs hgrad => ?_⟩
  rw [← circle_hessian_expanded β hβ σ σ' hσc hσ' ω a μ s hs]
  exact hH s hs hgrad

/-- Every regularity and strict SOPD input is jointly realized by the
actual pinned Dirac with primitive `id`, activation `1/2` and derivative
zero. Zero speed supplies a continuous smooth gradient direction. -/
example : (0 : ℝ) < 1 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (fun _ : ℝ => (2 : ℝ)⁻¹) ∧ Continuous (0 : ℝ → ℝ) ∧
    IsStrictSOPD 1 (fun t => t) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    Continuous (fun _ : SSphere 2 => (0 : ℝ)) ∧
    IsGradientField (fun x => (0 : ℝ) • circleTangent (x : EucSpace 2)) := by
  refine ⟨one_pos, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    continuous_const, isStrictSOPD_pin 1 0 (basePoint 1), continuous_const, ?_⟩
  simpa only [zero_smul] using (isGradientField_zero (d := 2))

end Transformer.Perceptron
