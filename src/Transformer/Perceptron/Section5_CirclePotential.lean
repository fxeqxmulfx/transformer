module

public import Transformer.Perceptron.Section5_CircleRotation
public import Transformer.Perceptron.CircleDeriv

/-!
# Perceptron force and curvature in continuous circle coordinates

ArXiv:2601.21366v2, §2.3, `eq: primitive.field`, and §5.2,
`eq:Hess-bilinear.std`. Under `φ'=2σ`, the potential contributes
one half of its actual angular derivative to the first variation.
The two explicit coordinate formulas below retain that normalization.

Their equality to the actual derivatives is proved in the angular
chart, and their derivatives along the actual rotation map are
computed before integrating. A continuous activation derivative
makes the curvature continuous on the compact sphere. Analytic
activations satisfy that regularity; no density or atomicity of
the measure is used in this calculation.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Perceptron

/-- One half of the perceptron's angular derivative in actual coordinates.
Source: arXiv:2601.21366v2, §2.3, `eq: primitive.field`, and §5.2. -/
noncomputable def circlePotentialForce (σ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (x : SSphere 2) : ℝ :=
  ∑ j : Idx 2, ω j * (σ (inner (𝕜 := ℝ) (a j) (x : EucSpace 2)) *
    inner (𝕜 := ℝ) (a j) (circleTangent (x : EucSpace 2)))

/-- One half of the perceptron's angular second derivative. Source:
arXiv:2601.21366v2, §5.2, the potential term of `eq:Hess-bilinear.std`.
The negative radial term is the unit circle's acceleration. -/
noncomputable def circlePotentialCurvature (σ σ' : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (x : SSphere 2) : ℝ :=
  ∑ j : Idx 2, ω j *
    (σ' (inner (𝕜 := ℝ) (a j) (x : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleTangent (x : EucSpace 2)) ^ 2 +
      σ (inner (𝕜 := ℝ) (a j) (x : EucSpace 2)) *
        -inner (𝕜 := ℝ) (a j) (x : EucSpace 2))

/-- The coordinate force is the actual normalized angular derivative.
Source: arXiv:2601.21366v2, §2.3, `φ'=2σ`, and §5.2. -/
theorem circlePotentialForce_circlePoint (φ σ : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    circlePotentialForce σ ω a (circlePoint θ) =
      (1 / 2 : ℝ) * deriv (fun r => potential φ ω a (circlePoint r : EucSpace 2)) θ := by
  rw [deriv_potential_circlePoint φ σ hφ]
  unfold circlePotentialForce
  rw [circleTangent_circlePoint]
  ring

/-- The coordinate curvature is the actual normalized second derivative.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
Only the actual pointwise derivative of the activation is required here. -/
theorem circlePotentialCurvature_circlePoint (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    circlePotentialCurvature σ σ' ω a (circlePoint θ) =
      (1 / 2 : ℝ) * deriv^[2] (fun r => potential φ ω a (circlePoint r : EucSpace 2)) θ := by
  rw [secondDeriv_potential_circlePoint φ σ σ' hφ hσ]
  unfold circlePotentialCurvature
  rw [circleTangent_circlePoint]
  ring

/-- The actual force derivative in the source's angular chart.
Source: arXiv:2601.21366v2, §5.2, the potential differentiation
in `eq:Hess-bilinear.std`. The formula is not assumed as a Hessian. -/
theorem hasDerivAt_circlePotentialForce (σ σ' : ℝ → ℝ)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    HasDerivAt (fun r => circlePotentialForce σ ω a (circlePoint r))
      (circlePotentialCurvature σ σ' ω a (circlePoint θ)) θ := by
  have hterm (j : Idx 2) : HasDerivAt
      (fun r => ω j * (σ (inner (𝕜 := ℝ) (a j) (circlePoint r : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel r)))
      (ω j * (σ' (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel θ) ^ 2 +
        σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          -inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2))) θ := by
    have h := ((hσ _).comp θ (hasDerivAt_inner_circlePoint (a j) θ)).mul
      (hasDerivAt_inner_circleVel (a j) θ)
    exact (h.congr_deriv (by dsimp; ring)).const_mul (ω j)
  simpa only [circlePotentialForce, circlePotentialCurvature,
    circleTangent_circlePoint, Finset.sum_apply] using
    HasDerivAt.fun_sum (fun j (_ : j ∈ Finset.univ) => hterm j)

/-- The actual potential along the actual transport differentiates once.
Source: arXiv:2601.21366v2, §2.2 and §5.2, `eq:Hess-bilinear.std`.
The speed is fixed at the original point. -/
theorem hasDerivAt_rotated_potential (φ σ : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (s : SSphere 2 → ℝ) (x : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => potential φ ω a (circleRotation s r x : EucSpace 2))
      (2 * circlePotentialForce σ ω a (circleRotation s t x) * s x) t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have ha : HasDerivAt (fun r : ℝ => θ + r * s (circlePoint θ)) (s (circlePoint θ)) t := by
    simpa only [id_eq, one_mul] using ((hasDerivAt_id t).mul_const (s (circlePoint θ))).const_add θ
  convert (hasDerivAt_potential_circlePoint φ σ hφ ω a _).comp t ha using 1
  · funext r
    rw [circleRotation_circlePoint]
    rfl
  · rw [circleRotation_circlePoint, circlePotentialForce, circleTangent_circlePoint]

/-- The actual first-derivative integrand differentiates again.
Source: arXiv:2601.21366v2, §5.2, the squared speed in
the potential term of `eq:Hess-bilinear.std`. -/
theorem hasDerivAt_rotated_potential_force (σ σ' : ℝ → ℝ)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (s : SSphere 2 → ℝ) (x : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => 2 * circlePotentialForce σ ω a (circleRotation s r x) * s x)
      (2 * circlePotentialCurvature σ σ' ω a (circleRotation s t x) * s x ^ 2) t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have ha : HasDerivAt (fun r : ℝ => θ + r * s (circlePoint θ)) (s (circlePoint θ)) t := by
    simpa only [id_eq, one_mul] using ((hasDerivAt_id t).mul_const (s (circlePoint θ))).const_add θ
  convert (((hasDerivAt_circlePotentialForce σ σ' hσ ω a _).comp t ha).const_mul
    (2 : ℝ)).mul_const (s (circlePoint θ)) using 1
  · funext r
    rw [circleRotation_circlePoint]
    rfl
  · rw [circleRotation_circlePoint]
    ring

/-- Continuous activations give a continuous coordinate force. Source:
arXiv:2601.21366v2, §5.2, the dominated potential differentiation. -/
theorem continuous_circlePotentialForce (σ : ℝ → ℝ) (hσ : Continuous σ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) :
    Continuous (circlePotentialForce σ ω a) := by
  have hJ := continuous_circleTangent
  unfold circlePotentialForce
  fun_prop

/-- Continuous activation derivatives give continuous coordinate curvature.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem continuous_circlePotentialCurvature (σ σ' : ℝ → ℝ)
    (hσ : Continuous σ) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) :
    Continuous (circlePotentialCurvature σ σ' ω a) := by
  have hJ := continuous_circleTangent
  unfold circlePotentialCurvature
  fun_prop

/-- All primitive, actual-derivative and continuity hypotheses are
jointly realized by the quadratic primitive, linear activation and
constant derivative one. Arbitrary neurons and actual rotations
are allowed by these pointwise and continuity results. -/
example : (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) r) r) ∧
    Continuous (fun t : ℝ => t) ∧ Continuous (fun _ : ℝ => (1 : ℝ)) := by
  exact ⟨fun r => by simpa using hasDerivAt_pow 2 r,
    fun r => hasDerivAt_id' (𝕜 := ℝ) (x := r), continuous_id, continuous_const⟩

end Transformer.Perceptron
