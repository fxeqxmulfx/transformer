/-
# Twice differentiating the actual transported perceptron integral

ArXiv:2601.21366v2, §2.3, `eq: primitive.field`, and §5.2,
`eq:Hess-bilinear.std`. The potential along a Monge transport
is integrated against the original probability measure. The
coordinate force and curvature proved in `Section5_CirclePotential`
are continuous and bounded on the compact circle when `σ` and
its actual derivative are continuous.

Two dominated differentiations prove existence and the value
of the actual second derivative for arbitrary measure support.
The raw potential integrand retains the factor two from `φ'=2σ`;
the energy will subsequently multiply its integral by one half.
-/

import Transformer.Perceptron.Section5_CirclePotential
import Transformer.Perceptron.Section5_DominatedVariation
import Transformer.Perceptron.MinMax

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The original potential at the transported point. Source:
arXiv:2601.21366v2, §2.3 and §5.2, `eq:Hess-bilinear.std`. -/
noncomputable def rotatedPotential (φ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (t : ℝ) (x : SSphere 2) : ℝ :=
  potential φ ω a (circleRotation s t x : EucSpace 2)

/-- The actual first potential derivative along the transport.
Source: arXiv:2601.21366v2, §5.2, the first differentiation. -/
noncomputable def rotatedPotentialForce (σ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (t : ℝ) (x : SSphere 2) : ℝ :=
  2 * circlePotentialForce σ ω a (circleRotation s t x) * s x

/-- The actual second potential derivative, with squared initial speed.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
noncomputable def rotatedPotentialCurvature (σ σ' : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (t : ℝ) (x : SSphere 2) : ℝ :=
  2 * circlePotentialCurvature σ σ' ω a (circleRotation s t x) * s x ^ 2

/-- Continuity of the original potential along the transport. Source:
arXiv:2601.21366v2, §5.2, dominated differentiation of the energy. -/
theorem continuous_rotatedPotential (φ : ℝ → ℝ) (hφ : Continuous φ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedPotential φ ω a s t) :=
  (continuous_potential hφ ω a).comp (continuous_circleRotation s hs t).subtype_val

/-- Continuity of its first-derivative integrand. Source:
arXiv:2601.21366v2, §5.2, the first potential differentiation. -/
theorem continuous_rotatedPotentialForce (σ : ℝ → ℝ) (hσ : Continuous σ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedPotentialForce σ ω a s t) :=
  (((continuous_circlePotentialForce σ hσ ω a).comp
    (continuous_circleRotation s hs t)).const_mul 2).mul hs

/-- Continuity of its second-derivative integrand. Source:
arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem continuous_rotatedPotentialCurvature (σ σ' : ℝ → ℝ)
    (hσ : Continuous σ) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (rotatedPotentialCurvature σ σ' ω a s t) :=
  (((continuous_circlePotentialCurvature σ σ' hσ hσ' ω a).comp
    (continuous_circleRotation s hs t)).const_mul 2).mul (hs.pow 2)

/-- Compact force and speed bounds dominate the first derivative globally
in time. Source: arXiv:2601.21366v2, §5.2, the potential integral. -/
theorem abs_rotatedPotentialForce_le (σ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (BF BS : ℝ) (hBF₀ : 0 ≤ BF)
    (hBF : ∀ x, |circlePotentialForce σ ω a x| ≤ BF) (hBS : ∀ x, |s x| ≤ BS)
    (t : ℝ) (x : SSphere 2) :
    |rotatedPotentialForce σ ω a s t x| ≤ 2 * BF * BS := by
  rw [rotatedPotentialForce, abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  exact mul_le_mul (mul_le_mul_of_nonneg_left (hBF _) (by norm_num)) (hBS x)
    (abs_nonneg _) (by positivity)

/-- The same compact bounds dominate the second derivative globally.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem abs_rotatedPotentialCurvature_le (σ σ' : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (BC BS : ℝ)
    (hBC₀ : 0 ≤ BC) (hBS₀ : 0 ≤ BS)
    (hBC : ∀ x, |circlePotentialCurvature σ σ' ω a x| ≤ BC) (hBS : ∀ x, |s x| ≤ BS)
    (t : ℝ) (x : SSphere 2) :
    |rotatedPotentialCurvature σ σ' ω a s t x| ≤ 2 * BC * BS ^ 2 := by
  rw [rotatedPotentialCurvature, abs_mul, abs_mul,
    abs_of_pos (by norm_num : (0 : ℝ) < 2), abs_of_nonneg (sq_nonneg (s x))]
  have hsq : (s x) ^ 2 ≤ BS ^ 2 := by
    have h := hBS x
    nlinarith [abs_nonneg (s x), sq_abs (s x)]
  exact mul_le_mul (mul_le_mul_of_nonneg_left (hBC _) (by norm_num)) hsq
    (sq_nonneg _) (by positivity)

/-- The actual first derivative under the original probability measure.
Source: arXiv:2601.21366v2, §5.2, before `eq:Hess-bilinear.std`. -/
theorem hasDerivAt_integral_rotatedPotential (φ σ : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : Continuous σ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (fun r => ∫ x, rotatedPotential φ ω a s r x ∂(μ : Measure (SSphere 2)))
      (∫ x, rotatedPotentialForce σ ω a s t x ∂(μ : Measure (SSphere 2))) t := by
  obtain ⟨BF, hBF₀, hBF⟩ := exists_pos_bound_continuous_compact _
    (continuous_circlePotentialForce σ hσ ω a)
  obtain ⟨BS, -, hBS⟩ := exists_pos_bound_continuous_compact s hs
  exact hasDerivAt_integral_on_compact (μ : Measure (SSphere 2))
    (rotatedPotential φ ω a s) (rotatedPotentialForce σ ω a s)
    (continuous_rotatedPotential φ (continuous_of_hasDerivAt hφ) ω a s hs)
    (continuous_rotatedPotentialForce σ hσ ω a s hs) (2 * BF * BS)
    (abs_rotatedPotentialForce_le σ ω a s BF BS hBF₀.le hBF hBS)
    (fun r x => hasDerivAt_rotated_potential φ σ hφ ω a s x r) t

/-- Two dominated differentiations prove the actual second potential
variation for arbitrary support. Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear.std`. Continuous `σ'` supplies the compact bound. -/
theorem hasDerivAt_deriv_integral_rotatedPotential (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ x, rotatedPotential φ ω a s r x
      ∂(μ : Measure (SSphere 2))))
      (∫ x, rotatedPotentialCurvature σ σ' ω a s t x ∂(μ : Measure (SSphere 2))) t := by
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  obtain ⟨BF, hBF₀, hBF⟩ := exists_pos_bound_continuous_compact _
    (continuous_circlePotentialForce σ hσc ω a)
  obtain ⟨BC, hBC₀, hBC⟩ := exists_pos_bound_continuous_compact _
    (continuous_circlePotentialCurvature σ σ' hσc hσ' ω a)
  obtain ⟨BS, hBS₀, hBS⟩ := exists_pos_bound_continuous_compact s hs
  exact hasDerivAt_deriv_integral_on_compact (μ : Measure (SSphere 2))
    (rotatedPotential φ ω a s) (rotatedPotentialForce σ ω a s)
    (rotatedPotentialCurvature σ σ' ω a s)
    (continuous_rotatedPotential φ (continuous_of_hasDerivAt hφ) ω a s hs)
    (continuous_rotatedPotentialForce σ hσc ω a s hs)
    (continuous_rotatedPotentialCurvature σ σ' hσc hσ' ω a s hs)
    (2 * BF * BS) (2 * BC * BS ^ 2)
    (abs_rotatedPotentialForce_le σ ω a s BF BS hBF₀.le hBF hBS)
    (abs_rotatedPotentialCurvature_le σ σ' ω a s BC BS hBC₀.le hBS₀.le hBC hBS)
    (fun r x => hasDerivAt_rotated_potential φ σ hφ ω a s x r)
    (fun r x => hasDerivAt_rotated_potential_force σ σ' hσ ω a s x r) t

/-- At time zero the curvature uses the actual original point and
its fixed speed. Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem rotatedPotentialCurvature_zero (σ σ' : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (s : SSphere 2 → ℝ) (x : SSphere 2) :
    rotatedPotentialCurvature σ σ' ω a s 0 x =
      2 * circlePotentialCurvature σ σ' ω a x * s x ^ 2 := by
  simp only [rotatedPotentialCurvature, circleRotation_zero]

/-- All regularity and domination inputs are jointly realized by the
quadratic primitive, linear activation with derivative one, unit speed,
and zero weights; the measure can be the actual Dirac probability. -/
example : (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) r) r) ∧
    Continuous (fun t : ℝ => t) ∧ Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ x : SSphere 2, |circlePotentialForce (fun t => t) 0 0 x| ≤ 1) ∧
    (∀ x : SSphere 2, |circlePotentialCurvature (fun t => t) (fun _ => 1) 0 0 x| ≤ 1) ∧
    (∀ x : SSphere 2, |(fun _ : SSphere 2 => (1 : ℝ)) x| ≤ 1) := by
  refine ⟨fun r => by simpa using hasDerivAt_pow 2 r,
    fun r => hasDerivAt_id' (𝕜 := ℝ) (x := r), continuous_id, continuous_const,
    continuous_const, zero_le_one, ?_, ?_, fun _ => by norm_num⟩
  · intro x
    simp [circlePotentialForce]
  · intro x
    simp [circlePotentialCurvature]

end Transformer.Perceptron
