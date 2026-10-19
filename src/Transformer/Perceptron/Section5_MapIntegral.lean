module

public import Transformer.Perceptron.Section5_MapTransport
public import Transformer.Perceptron.Section5_DominatedVariation

/-!
# Dominated matrix pair variations for arbitrary circle measures

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2
to the actual matrix interaction `E_B` of §6, `rem:unifiedlog-B`.
The actual transported kernel and its two proved derivatives are
continuous in the original sphere pair when its speeds are continuous.

Domination follows from compactness of two copies of the pair space:
one pair supplies the initial speeds and one pair supplies the current
kernel coefficients. This bounds the actual derivative integrands at
every real time, without replacing `B` by a scalar matrix or imposing
an atomic presentation of the probability measure.

Two applications of differentiation under the integral sign prove
existence of the interaction's actual second derivative. The factor
one half in `E_B` is applied subsequently to this raw pair integral.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Continuity of the transported original matrix kernel. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §6, `E_B`. -/
theorem continuous_rotatedMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedMapPairKernel B s t) := by
  have hR := continuous_circleRotation s hs t
  exact (continuous_circleMapPairKernel B).comp
    ((hR.comp continuous_fst).prodMk (hR.comp continuous_snd))

/-- Continuity of the actual first derivative integrand. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem continuous_rotatedMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedMapPairForce B s t) := by
  have hR := continuous_circleRotation s hs t
  have hF := continuous_circleMapPairForce B
  unfold rotatedMapPairForce
  fun_prop

/-- Continuity of both diagonal and mixed second derivative terms.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem continuous_rotatedMapPairCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedMapPairCurvature B s t) := by
  have hR := continuous_circleRotation s hs t
  have hC := continuous_circleMapPairCurvature B
  have hM := continuous_circleMapPairMixedCurvature B
  unfold rotatedMapPairCurvature
  fun_prop

/-- Compact initial and current pairs bound the first integrand for
all transport times. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
the dominated first differentiation of the interaction in §5.2. -/
theorem exists_bound_rotatedMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    ∃ C : ℝ, 0 < C ∧ ∀ t p, |rotatedMapPairForce B s t p| ≤ C := by
  let G : (SSphere 2 × SSphere 2) × (SSphere 2 × SSphere 2) → ℝ := fun q =>
    s q.1.1 * circleMapPairForce B q.2.1 q.2.2 +
      s q.1.2 * circleMapPairForce B q.2.2 q.2.1
  have hF := continuous_circleMapPairForce B
  have hG : Continuous G := by
    dsimp only [G]
    fun_prop
  obtain ⟨C, hC, hbound⟩ := exists_pos_bound_continuous_compact G hG
  exact ⟨C, hC, fun t p => hbound (p, (circleRotation s t p.1, circleRotation s t p.2))⟩

/-- The same compact pairs bound the actual second derivative for
all times. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
the dominated second differentiation of the interaction in §5.2. -/
theorem exists_bound_rotatedMapPairCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    ∃ C : ℝ, 0 < C ∧ ∀ t p, |rotatedMapPairCurvature B s t p| ≤ C := by
  let G : (SSphere 2 × SSphere 2) × (SSphere 2 × SSphere 2) → ℝ := fun q =>
    s q.1.1 ^ 2 * circleMapPairCurvature B q.2.1 q.2.2 +
      s q.1.2 ^ 2 * circleMapPairCurvature B q.2.2 q.2.1 +
        2 * s q.1.1 * s q.1.2 * circleMapPairMixedCurvature B q.2.1 q.2.2
  have hC := continuous_circleMapPairCurvature B
  have hM := continuous_circleMapPairMixedCurvature B
  have hG : Continuous G := by
    dsimp only [G]
    fun_prop
  obtain ⟨C, hC, hbound⟩ := exists_pos_bound_continuous_compact G hG
  exact ⟨C, hC, fun t p => hbound (p, (circleRotation s t p.1, circleRotation s t p.2))⟩

/-- The justified actual first derivative under the original product
measure. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.2.
The supplied symmetry is the actual matrix's symmetry. -/
theorem hasDerivAt_integral_rotatedMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (fun r => ∫ p, rotatedMapPairKernel B s r p
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
      (∫ p, rotatedMapPairForce B s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) t := by
  obtain ⟨C, -, hC⟩ := exists_bound_rotatedMapPairForce B s hs
  exact hasDerivAt_integral_on_compact
    ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
    (rotatedMapPairKernel B s) (rotatedMapPairForce B s)
    (continuous_rotatedMapPairKernel B s hs) (continuous_rotatedMapPairForce B s hs)
    C hC (fun r p => hasDerivAt_rotatedMapPairKernel B hsymm s p.1 p.2 r) t

/-- Two dominated differentiations give an actual second derivative,
before any support conclusion. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (ii), extending `eq:Hess-bilinear.std` of §5.2. -/
theorem hasDerivAt_deriv_integral_rotatedMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ p, rotatedMapPairKernel B s r p
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))))
      (∫ p, rotatedMapPairCurvature B s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) t := by
  obtain ⟨C₁, -, hC₁⟩ := exists_bound_rotatedMapPairForce B s hs
  obtain ⟨C₂, -, hC₂⟩ := exists_bound_rotatedMapPairCurvature B s hs
  exact hasDerivAt_deriv_integral_on_compact
    ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
    (rotatedMapPairKernel B s) (rotatedMapPairForce B s) (rotatedMapPairCurvature B s)
    (continuous_rotatedMapPairKernel B s hs) (continuous_rotatedMapPairForce B s hs)
    (continuous_rotatedMapPairCurvature B s hs) C₁ C₂ hC₁ hC₂
    (fun r p => hasDerivAt_rotatedMapPairKernel B hsymm s p.1 p.2 r)
    (fun r p => hasDerivAt_rotatedMapPairForce B hsymm s p.1 p.2 r) t

/-- At the initial time the actual transported pair is the original
pair. Source: arXiv:2601.21366v2, §2.2, the Monge geodesic. -/
theorem rotatedMapPairKernel_zero (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (p : SSphere 2 × SSphere 2) :
    rotatedMapPairKernel B s 0 p = circleMapPairKernel B p.1 p.2 := by
  simp only [rotatedMapPairKernel, circleRotation_zero]

/-- The initial second variation retains both diagonal terms and
the actual mixed coefficient. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (ii), extending the Hessian at time zero from §5.2. -/
theorem rotatedMapPairCurvature_zero (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (p : SSphere 2 × SSphere 2) :
    rotatedMapPairCurvature B s 0 p =
      circleMapPairCurvature B p.1 p.2 * s p.1 ^ 2 +
        circleMapPairCurvature B p.2 p.1 * s p.2 ^ 2 +
          2 * circleMapPairMixedCurvature B p.1 p.2 * s p.1 * s p.2 := by
  simp only [rotatedMapPairCurvature, circleRotation_zero]
  ring

/-- All continuity and symmetry inputs are jointly realized by the
identity matrix and constant unit speed. The actual Dirac measure
is an available probability input to both derivative theorems. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) := ⟨fun _ _ => rfl, continuous_const⟩

end Transformer.Perceptron
