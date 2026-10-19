module

public import Transformer.Perceptron.Section5_MapHessian

/-!
# Expanding the actual matrix circle Hessian

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending
`eq:Hess-bilinear` of §5.2 to the matrix energy `E_B` of §6.
Swapping the original product measure identifies the two diagonal
second-variation integrals, even though their coefficient need not
be symmetric. Fubini then gives the actual local force curvature.

The mixed coefficient retains its matrix formula and sign. Writing
its negative as the pair kernel puts the full Hessian in the form
used by the already proved local Hessian estimate. Strict SOPD
therefore supplies a lower bound on the same actual quadratic form.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Swapping a product measure and Fubini expand two diagonal terms
and one mixed term. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
the matrix counterpart of the expansion of `eq:Hess-bilinear` in §5.2.
No symmetry of the diagonal coefficient is assumed. -/
theorem integral_pair_diagonal_mixed_expansion {α : Type*} [TopologicalSpace α]
    [CompactSpace α] [SecondCountableTopology α] [MeasurableSpace α] [OpensMeasurableSpace α]
    (μ : Measure α) [IsFiniteMeasure μ] (D M : α × α → ℝ)
    (hD : Continuous D) (hM : Continuous M) (s : α → ℝ) (hs : Continuous s) :
    (1 / 2 : ℝ) * (∫ p, D p * s p.1 ^ 2 + D p.swap * s p.2 ^ 2 +
      2 * M p * s p.1 * s p.2 ∂μ.prod μ) =
      (∫ x, (∫ y, D (x, y) ∂μ) * s x ^ 2 ∂μ) +
        (∫ p, M p * s p.1 * s p.2 ∂μ.prod μ) := by
  let f₁ : α × α → ℝ := fun p => D p * s p.1 ^ 2
  let f₂ : α × α → ℝ := fun p => D p.swap * s p.2 ^ 2
  let g : α × α → ℝ := fun p => M p * s p.1 * s p.2
  have hf₁ : Integrable f₁ (μ.prod μ) :=
    (hD.mul ((hs.comp continuous_fst).pow 2)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hf₂ : Integrable f₂ (μ.prod μ) :=
    ((hD.comp continuous_swap).mul ((hs.comp continuous_snd).pow 2)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hg : Integrable g (μ.prod μ) :=
    ((hM.mul (hs.comp continuous_fst)).mul (hs.comp continuous_snd)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have h₂ : (∫ p, f₂ p ∂μ.prod μ) = ∫ p, f₁ p ∂μ.prod μ :=
    integral_prod_swap (μ := μ) (ν := μ) f₁
  have h₁ : (∫ p, f₁ p ∂μ.prod μ) = ∫ x, (∫ y, D (x, y) ∂μ) * s x ^ 2 ∂μ := by
    calc
      _ = ∫ x, ∫ y, D (x, y) * s x ^ 2 ∂μ ∂μ :=
        (integral_integral (f := fun x y => D (x, y) * s x ^ 2) hf₁).symm
      _ = _ := integral_congr_ae (ae_of_all _ fun x => integral_mul_const _ _)
  have hexpand : (fun p : α × α => D p * s p.1 ^ 2 + D p.swap * s p.2 ^ 2 +
      2 * M p * s p.1 * s p.2) = fun p => f₁ p + f₂ p + 2 * g p := by
    funext p
    dsimp only [f₁, f₂, g]
    ring
  rw [hexpand, integral_add (hf₁.fun_add hf₂) (hg.const_mul 2), integral_add hf₁ hf₂,
    integral_const_mul, h₂, h₁]
  dsimp only [g]
  ring

/-- All continuity inputs are jointly realized by nonzero constant
coefficients and unit speed on the nonempty compact space `Unit`.
Its actual Dirac probability supplies the finite measure input. -/
example : Continuous (fun _ : Unit × Unit => (1 : ℝ)) ∧
    Continuous (fun _ : Unit × Unit => (-1 : ℝ)) ∧
    Continuous (fun _ : Unit => (1 : ℝ)) :=
  ⟨continuous_const, continuous_const, continuous_const⟩

/-- The actual matrix coefficients give the full local-plus-mixed
Hessian expansion. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending `eq:Hess-bilinear` of §5.2. The local coefficient is
the proved actual force derivative, with its original measure. -/
theorem circle_map_hessian_expanded (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (σ σ' : ℝ → ℝ) (hσ : Continuous σ) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    (1 / 2 : ℝ) * (∫ p, rotatedMapPairCurvature B s 0 p
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
      (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2 ∂(μ : Measure (SSphere 2))) =
    (∫ x, circleCurvatureMap B σ σ' ω a μ x * s x ^ 2 ∂(μ : Measure (SSphere 2))) +
      (∫ p, circleMapPairMixedCurvature B p.1 p.2 * s p.1 * s p.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) := by
  have hpair := integral_pair_diagonal_mixed_expansion (μ : Measure (SSphere 2))
    (fun p => circleMapPairCurvature B p.1 p.2) (fun p => circleMapPairMixedCurvature B p.1 p.2)
    (continuous_circleMapPairCurvature B) (continuous_circleMapPairMixedCurvature B) s hs
  dsimp only [Prod.swap] at hpair
  have hDI : Integrable (fun x => (∫ y, circleMapPairCurvature B x y
      ∂(μ : Measure (SSphere 2))) * s x ^ 2) (μ : Measure (SSphere 2)) :=
    ((continuous_integral_circle_pair μ _ (continuous_circleMapPairCurvature B)).mul
      (hs.pow 2)).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hPI : Integrable (fun x => circlePotentialCurvature σ σ' ω a x * s x ^ 2)
      (μ : Measure (SSphere 2)) :=
    ((continuous_circlePotentialCurvature σ σ' hσ hσ' ω a).mul
      (hs.pow 2)).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hA : (∫ x, circleCurvatureMap B σ σ' ω a μ x * s x ^ 2 ∂(μ : Measure (SSphere 2))) =
      (∫ x, (∫ y, circleMapPairCurvature B x y ∂(μ : Measure (SSphere 2))) * s x ^ 2
        ∂(μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2 ∂(μ : Measure (SSphere 2))) := by
    simp only [circleCurvatureMap, add_mul]
    exact integral_add hDI hPI
  simp only [rotatedMapPairCurvature_zero]
  rw [hpair, hA]
  ring

/-- The expanded expression is the actual full-energy derivative.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending
`eq:Hess-bilinear` of §5.2. Its existence holds for arbitrary support. -/
theorem hasDerivAt_deriv_energyMap_circleRotation_expanded (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) :
    HasDerivAt (deriv (fun t => energyMap B φ ω a (μ.map (circleRotation s t))))
      ((∫ x, circleCurvatureMap B σ σ' ω a μ x * s x ^ 2 ∂(μ : Measure (SSphere 2))) +
        (∫ p, circleMapPairMixedCurvature B p.1 p.2 * s p.1 * s p.2
          ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))) 0 := by
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have h := hasDerivAt_deriv_energyMap_circleRotation B hsymm φ σ σ' hφ hσ hσ' ω a μ s hs
  exact h.congr_deriv (circle_map_hessian_expanded B σ σ' hσc hσ' ω a μ s hs)

/-- Strict SOPD dominates the expanded actual Hessian in the signed-kernel
form used by the local estimate. Source: arXiv:2601.21366v2, §2.2,
`eq:strict2order`, and §3.1, `rem: ext` (ii), extending §5.2. -/
theorem strict_circle_map_hessian_coercive_expanded (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStrictSOPDMap B φ σ ω a μ) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ s : SSphere 2 → ℝ, Continuous s →
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) →
      κ * (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) ≤
        (∫ x, circleCurvatureMap B σ σ' ω a μ x * s x ^ 2 ∂(μ : Measure (SSphere 2))) -
          (∫ p, (-circleMapPairMixedCurvature B p.1 p.2) * s p.1 * s p.2
            ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) := by
  obtain ⟨κ, hκ, hH⟩ := strict_circle_map_hessian_coercive B hsymm φ σ σ' hφ hσ hσ' ω a μ hμ
  refine ⟨κ, hκ, fun s hs hgrad => ?_⟩
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have h := hH s hs hgrad
  rw [circle_map_hessian_expanded B σ σ' hσc hσ' ω a μ s hs] at h
  simpa only [neg_mul, integral_neg, sub_neg_eq_add] using h

/-- All activation, primitive, continuity, matrix symmetry and strictness
inputs are jointly realized by the actual pinned Dirac at `B=id`.
The coefficient identities also allow continuous unit angular speed. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (fun _ : ℝ => (2 : ℝ)⁻¹) ∧ Continuous (0 : ℝ → ℝ) ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun t => t)
      (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => rfl, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const, continuous_const,
    continuous_const, (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 _)⟩

end Transformer.Perceptron
