module

public import Transformer.Perceptron.Section5_SphericalKernelBounds
public import Transformer.Perceptron.Section5_DominatedVariation

/-!
# Regularity of transported spherical interaction integrands

ArXiv:2601.21366v2, §5.3, the localization argument for
`thm: any.d` (i), extending the actual second variations of §5.2.
Two continuous tangent fields specify the actual exponential
curves of both original integration points. The kernel and its
two genuine time derivatives are continuous in that original pair.

Compactness bounds both initial speeds. The pointwise estimates
from `Section5_SphericalKernelBounds` consequently give constants
dominating the actual derivative integrands at every real time.
The next step can therefore differentiate under the original
product probability measure twice, including for arbitrary support.
The two fields may differ; taking one to be zero supplies the
one-point transport terms needed for the exact energy cancellation.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The actual pair kernel along two original Monge fields.
Source: arXiv:2601.21366v2, §2.2–§2.3 and §5.3. -/
noncomputable def transportedPairKernel {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  spherePairKernel β (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- Its actual first derivative, with the original two fields.
Source: arXiv:2601.21366v2, §5.3, the first interaction differentiation. -/
noncomputable def transportedPairForce {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  spherePairForce β (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- Its actual second derivative, retaining the radial accelerations.
Source: arXiv:2601.21366v2, §5.3, the interaction Hessian. -/
noncomputable def transportedPairCurvature {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  spherePairCurvature β (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- Actual positions vary continuously with the original point.
Source: arXiv:2601.21366v2, §5.3, the transported kernel integrability. -/
theorem continuous_spherePosition_field {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hc : Continuous ξ) (t : ℝ) :
    Continuous (fun x : SSphere d => spherePosition (x : EucSpace d) (ξ x) t) := by
  unfold spherePosition
  exact ((Real.continuous_cos.comp (hc.norm.const_mul t)).smul continuous_subtype_val).add
    ((continuous_const.mul (Real.continuous_sinc.comp (hc.norm.const_mul t))).smul hc)

/-- The actual transported kernel is continuous in the original pair.
Source: arXiv:2601.21366v2, §5.3, before differentiation under the integral. -/
theorem continuous_transportedPairKernel {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedPairKernel β ξ η t) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := (continuous_spherePosition_field ξ hξ t).comp hfst
  have hY := (continuous_spherePosition_field η hη t).comp hsnd
  exact Real.continuous_exp.comp ((hX.inner hY).const_mul β)

/-- The actual pair inner-product velocity is continuous in the pair.
Source: arXiv:2601.21366v2, §5.3, the first-derivative domination inputs. -/
theorem continuous_transportedPairSlope {d : ℕ}
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (fun p : SSphere d × SSphere d =>
      spherePairSlope (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := (continuous_spherePosition_field ξ hξ t).comp hfst
  have hY := (continuous_spherePosition_field η hη t).comp hsnd
  have hV := (continuous_sphereVelocity_field ξ hξ t).comp hfst
  have hW := (continuous_sphereVelocity_field η hη t).comp hsnd
  exact (hV.inner hY).add (hX.inner hW)

/-- The original radial-acceleration expression is continuous.
Source: arXiv:2601.21366v2, §5.3, the second-derivative domination inputs. -/
theorem continuous_transportedPairAcceleration {d : ℕ}
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (fun p : SSphere d × SSphere d =>
      spherePairAcceleration (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := (continuous_spherePosition_field ξ hξ t).comp hfst
  have hY := (continuous_spherePosition_field η hη t).comp hsnd
  have hV := (continuous_sphereVelocity_field ξ hξ t).comp hfst
  have hW := (continuous_sphereVelocity_field η hη t).comp hsnd
  have hI : Continuous (fun p : SSphere d × SSphere d => inner (𝕜 := ℝ)
      (spherePosition (p.1 : EucSpace d) (ξ p.1) t)
      (spherePosition (p.2 : EucSpace d) (η p.2) t)) := hX.inner hY
  exact ((hV.inner hW).const_mul 2).sub
    ((((hξ.norm.comp continuous_fst).pow 2).add
      ((hη.norm.comp continuous_snd).pow 2)).mul hI)

/-- The actual first-derivative integrand is continuous in the pair.
Source: arXiv:2601.21366v2, §5.3, first differentiation under the integral. -/
theorem continuous_transportedPairForce {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedPairForce β ξ η t) :=
  (continuous_transportedPairKernel β ξ η hξ hη t).mul
    ((continuous_transportedPairSlope ξ η hξ hη t).const_mul β)

/-- The actual second-derivative integrand is continuous in the pair.
Source: arXiv:2601.21366v2, §5.3, second differentiation under the integral. -/
theorem continuous_transportedPairCurvature {d : ℕ} (β : ℝ)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedPairCurvature β ξ η t) :=
  (continuous_transportedPairKernel β ξ η hξ hη t).mul
    ((((continuous_transportedPairSlope ξ η hξ hη t).pow 2).const_mul (β ^ 2)).add
      ((continuous_transportedPairAcceleration ξ η hξ hη t).const_mul β))

/-- Bounded initial speeds dominate both actual kernel derivatives at
every transport time. Source: arXiv:2601.21366v2, §5.3, the domination
justifying the actual Hessian in the localized-gradient argument. -/
theorem transportedPair_derivative_bounds {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (ξ η : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0)
    (B C : ℝ) (hB : ∀ x, ‖ξ x‖ ≤ B) (hC : ∀ x, ‖η x‖ ≤ C) :
    (∀ t p, |transportedPairForce β ξ η t p| ≤ Real.exp β * (β * (B + C))) ∧
    (∀ t p, |transportedPairCurvature β ξ η t p| ≤
      Real.exp β * ((β ^ 2 + β) * (B + C) ^ 2)) := by
  constructor
  · intro t p
    apply (abs_spherePairForce_le β hβ (norm_eq_of_mem_sphere p.1) (hξ p.1)
      (norm_eq_of_mem_sphere p.2) (hη p.2) t).trans
    gcongr
    · exact hB p.1
    · exact hC p.2
  · intro t p
    apply (abs_spherePairCurvature_le β hβ (norm_eq_of_mem_sphere p.1) (hξ p.1)
      (norm_eq_of_mem_sphere p.2) (hη p.2) t).trans
    gcongr
    · exact hB p.1
    · exact hC p.2

/-- Every regularity, tangency, temperature and speed-bound input is
jointly realized by the nonzero unit tangent field on the actual circle.
Source: arXiv:2601.21366v2, §2.2 and §5.2, the circle transport. -/
example : Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ x : SSphere 2, ‖circleTangent (x : EucSpace 2)‖ ≤ 1) :=
  ⟨continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2), zero_le_one,
    fun x => by rw [norm_circleTangent, norm_eq_of_mem_sphere x]⟩

end Transformer.Perceptron
