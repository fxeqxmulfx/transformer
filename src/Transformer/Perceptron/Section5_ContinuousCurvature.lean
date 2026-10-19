module

public import Transformer.Perceptron.Section5_RotatedKernel
public import Transformer.Perceptron.Section5_CirclePotential
public import Transformer.Perceptron.Section5_FirstVariationCurvature

/-!
# Continuous first-variation curvature on the actual circle

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`. The coefficient
of the local squared speed in the expanded Hessian is the
actual first variation's angular second derivative. It is the
interaction curvature integrated against the original measure
plus the normalized potential curvature.

At positive temperature the interaction is uniformly dominated
by `exp β`. Continuous activation derivatives therefore make
this coefficient continuous on the sphere, without any finite
support or density hypothesis. Its actual derivative identity
is proved using the preceding angular first-variation calculus.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The coefficient of the squared speed in the expanded Hessian.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
Its equality to the actual first-variation derivative is proved
below at positive temperature under the primitive convention. -/
noncomputable def circleFirstVariationCurvature (β : ℝ) (σ σ' : ℝ → ℝ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (x : SSphere 2) : ℝ :=
  (∫ y, circlePairCurvature β x y ∂(μ : Measure (SSphere 2))) +
    circlePotentialCurvature σ σ' ω a x

/-- The continuous coordinate pair curvature agrees with the
previous actual angular curvature. Source: arXiv:2601.21366v2,
§5.2, the kernel `K_β''/β` in `eq:Hess-bilinear`. -/
theorem circlePairCurvature_circlePoint_left (β θ : ℝ) (y : SSphere 2) :
    circlePairCurvature β (circlePoint θ) y = circleInteractionCurvature β θ y := by
  rw [circlePairCurvature, circleInteractionCurvature, circleTangent_circlePoint,
    real_inner_comm (y : EucSpace 2) (circlePoint θ : EucSpace 2)]

/-- The pair curvature is integrable at every actual circle point.
Source: arXiv:2601.21366v2, §5.2, the convolution in `eq:Hess-bilinear`.
Continuity on the compact sphere supplies integrability directly. -/
theorem integrable_circlePairCurvature (β : ℝ) (μ : Perspective.ProbSphere 2)
    (x : SSphere 2) :
    Integrable (circlePairCurvature β x) (μ : Measure (SSphere 2)) := by
  have hc : Continuous (circlePairCurvature β x) :=
    (continuous_circlePairCurvature β).comp (continuous_const.prodMk continuous_id)
  exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

/-- The integrated pair curvature is continuous for an arbitrary measure.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
The sharp bound `exp β` supplies uniform domination in the first point. -/
theorem continuous_integral_circlePairCurvature (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) :
    Continuous (fun x => ∫ y, circlePairCurvature β x y ∂(μ : Measure (SSphere 2))) := by
  apply continuous_of_dominated (bound := fun _ => Real.exp β)
  · intro x
    exact ((continuous_circlePairCurvature β).comp
      (continuous_const.prodMk continuous_id)).aestronglyMeasurable
  · intro x
    exact ae_of_all _ fun y => by
      simpa only [Real.norm_eq_abs] using abs_circlePairCurvature_le β hβ x y
  · exact integrable_const _
  · exact ae_of_all _ fun y => (continuous_circlePairCurvature β).comp
      (continuous_id.prodMk continuous_const)

/-- The actual first-variation curvature is continuous when the activation
and its actual derivative are continuous. Source: arXiv:2601.21366v2,
§5.2, `eq:Hess-bilinear`. Analytic activations satisfy these inputs. -/
theorem continuous_circleFirstVariationCurvature (β : ℝ) (hβ : 0 < β)
    (σ σ' : ℝ → ℝ) (hσ : Continuous σ) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) :
    Continuous (circleFirstVariationCurvature β σ σ' ω a μ) :=
  (continuous_integral_circlePairCurvature β hβ μ).add
    (continuous_circlePotentialCurvature σ σ' hσ hσ' ω a)

/-- The coordinate coefficient is the actual angular second derivative.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
The interaction and potential identities retain their original factors. -/
theorem circleFirstVariationCurvature_circlePoint (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    circleFirstVariationCurvature β σ σ' ω a μ (circlePoint θ) =
      deriv^[2] (angularFirstVariation β φ ω a μ) θ := by
  rw [circleFirstVariationCurvature, secondDeriv_angularFirstVariation β hβ φ σ σ' hφ hσ,
    circlePotentialCurvature_circlePoint φ σ σ' hφ hσ]
  congr 1
  exact integral_congr_ae (ae_of_all _ fun y => circlePairCurvature_circlePoint_left β θ y)

/-- The actual derivative of the stationary angular force has this
coordinate curvature. Source: arXiv:2601.21366v2, §5.2,
the differentiation behind `eq:global-cancel-analytic` and `eq:Hess-bilinear`.
This identity holds before any analyticity or support conclusion. -/
theorem hasDerivAt_angular_force_circleCurvature (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (deriv (angularFirstVariation β φ ω a μ))
      (circleFirstVariationCurvature β σ σ' ω a μ (circlePoint θ)) θ := by
  rw [circleFirstVariationCurvature_circlePoint β hβ φ σ σ' hφ hσ]
  exact (hasDerivAt_deriv_angularFirstVariation β hβ φ σ σ' hφ hσ ω a μ θ).differentiableAt.hasDerivAt

/-- The coordinate formula at a Dirac includes its actual self-curvature.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
At positive temperature this is the actual first-variation curvature;
the explicit coordinate formula also admits the displayed zero-temperature
extension, without asserting a derivative identity at zero temperature. -/
theorem circleFirstVariationCurvature_dirac_self (β : ℝ) (σ σ' : ℝ → ℝ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    circleFirstVariationCurvature β σ σ' ω a (Perspective.diracProb 2 (circlePoint θ))
      (circlePoint θ) = -Real.exp β + circlePotentialCurvature σ σ' ω a (circlePoint θ) := by
  unfold circleFirstVariationCurvature
  change (∫ y, circlePairCurvature β (circlePoint θ) y ∂Measure.dirac (circlePoint θ)) + _ = _
  rw [integral_dirac, circlePairCurvature_circlePoint_left, circleInteractionCurvature_self]

/-- With zero output weights the coefficient is exactly the actual
self-interaction curvature. Source: arXiv:2601.21366v2, §5.2,
the interaction convolution in `eq:Hess-bilinear`. This checks
the coordinate normalization on a nonempty probability measure. -/
theorem circleFirstVariationCurvature_dirac_zero_weights (β : ℝ) (σ σ' : ℝ → ℝ)
    (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    circleFirstVariationCurvature β σ σ' 0 a (Perspective.diracProb 2 (circlePoint θ))
      (circlePoint θ) = -Real.exp β := by
  rw [circleFirstVariationCurvature_dirac_self]
  simp [circlePotentialCurvature]

/-- Every regularity and positive-temperature input is jointly realized
by the pinned Dirac primitive `φ=id`, activation `σ=1/2`, derivative zero.
This actual strict SOPD witness has coefficient `1/2-exp 1`: the negative
self-curvature cancels in its full Hessian, whose value is `1/2`.
Thus strict SOPD alone is not being interpreted as positivity of this
coefficient at every individual atom. -/
example : (0 : ℝ) < 1 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (fun _ : ℝ => (2 : ℝ)⁻¹) ∧ Continuous (0 : ℝ → ℝ) ∧
    IsStrictSOPD 1 (fun t => t) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(circlePoint 0 : EucSpace 2)))
      (Perspective.diracProb 2 (circlePoint 0)) ∧
    circleFirstVariationCurvature 1 (fun _ => (2 : ℝ)⁻¹) 0 (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(circlePoint 0 : EucSpace 2)))
      (Perspective.diracProb 2 (circlePoint 0)) (circlePoint 0) = 1 / 2 - Real.exp 1 := by
  refine ⟨one_pos, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    continuous_const, isStrictSOPD_pin 1 0 (circlePoint 0), ?_⟩
  rw [circleFirstVariationCurvature_dirac_self]
  norm_num [circlePotentialCurvature, circleTangent_circlePoint, Fin.sum_univ_two,
    circlePoint, greatCircle, circleVel, basePoint, secondAxis]
  ring

end Transformer.Perceptron
