/-
# Twice differentiating the actual transported pair integral

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. For an
arbitrary circle probability measure, the interaction along a
Monge transport is integrated against the original product
measure. The two actual pointwise kernel derivatives established
in `Section5_RotatedKernel` have global bounds for continuous
angular speeds, since the circle is compact.

Two dominated differentiations therefore give a genuine second
derivative of the interaction integral without a density or finite
atomic presentation. The temperature and pair-speed factors are
retained before multiplying by the energy's prefactor `1/(2β)`.
-/

import Transformer.Perceptron.Section5_RotatedKernel
import Transformer.Perceptron.Section5_DominatedVariation
import Mathlib.MeasureTheory.Integral.Prod

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The original pair kernel evaluated at the actual transported points.
Source: arXiv:2601.21366v2, §2.3 and §5.2, `eq:Hess-bilinear.std`. -/
noncomputable def rotatedPairKernel (β : ℝ) (s : SSphere 2 → ℝ) (t : ℝ)
    (p : SSphere 2 × SSphere 2) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ)
    (circleRotation s t p.1 : EucSpace 2) (circleRotation s t p.2 : EucSpace 2))

/-- The actual first-derivative integrand, with the initial speed difference.
Source: arXiv:2601.21366v2, §5.2, before `eq:Hess-bilinear.std`. -/
noncomputable def rotatedPairForce (β : ℝ) (s : SSphere 2 → ℝ) (t : ℝ)
    (p : SSphere 2 × SSphere 2) : ℝ :=
  β * circlePairForce β (circleRotation s t p.1) (circleRotation s t p.2) * (s p.1 - s p.2)

/-- The actual second-derivative integrand. Source: arXiv:2601.21366v2,
§5.2, the interaction term of `eq:Hess-bilinear.std`. -/
noncomputable def rotatedPairCurvature (β : ℝ) (s : SSphere 2 → ℝ) (t : ℝ)
    (p : SSphere 2 × SSphere 2) : ℝ :=
  β * circlePairCurvature β (circleRotation s t p.1) (circleRotation s t p.2) *
    (s p.1 - s p.2) ^ 2

/-- The transported kernel is continuous in the original pair.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem continuous_rotatedPairKernel (β : ℝ) (s : SSphere 2 → ℝ)
    (hs : Continuous s) (t : ℝ) : Continuous (rotatedPairKernel β s t) := by
  have hR := (continuous_circleRotation s hs t).subtype_val
  exact Real.continuous_exp.comp
    (((hR.comp continuous_fst).inner (hR.comp continuous_snd)).const_mul β)

/-- The first-derivative integrand is continuous in that pair.
Source: arXiv:2601.21366v2, §5.2, the first differentiation. -/
theorem continuous_rotatedPairForce (β : ℝ) (s : SSphere 2 → ℝ)
    (hs : Continuous s) (t : ℝ) : Continuous (rotatedPairForce β s t) := by
  have hR := continuous_circleRotation s hs t
  exact ((continuous_circlePairForce β).comp
    ((hR.comp continuous_fst).prodMk (hR.comp continuous_snd))).const_mul β |>.mul
      ((hs.comp continuous_fst).sub (hs.comp continuous_snd))

/-- The second-derivative integrand is continuous as well.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem continuous_rotatedPairCurvature (β : ℝ) (s : SSphere 2 → ℝ)
    (hs : Continuous s) (t : ℝ) : Continuous (rotatedPairCurvature β s t) := by
  have hR := continuous_circleRotation s hs t
  exact ((continuous_circlePairCurvature β).comp
    ((hR.comp continuous_fst).prodMk (hR.comp continuous_snd))).const_mul β |>.mul
      (((hs.comp continuous_fst).sub (hs.comp continuous_snd)).pow 2)

/-- A global bound for the first actual derivative integrand.
Source: arXiv:2601.21366v2, §5.2, differentiation of the pair energy. -/
theorem abs_rotatedPairForce_le (β : ℝ) (hβ : 0 < β)
    (s : SSphere 2 → ℝ) (B : ℝ) (hB : ∀ x, |s x| ≤ B)
    (t : ℝ) (p : SSphere 2 × SSphere 2) :
    |rotatedPairForce β s t p| ≤ β * Real.exp β * (2 * B) := by
  rw [rotatedPairForce, abs_mul, abs_mul, abs_of_pos hβ]
  have hforce := abs_circlePairForce_le β hβ.le (circleRotation s t p.1) (circleRotation s t p.2)
  have hspeed := abs_speed_sub_le s B hB p.1 p.2
  gcongr

/-- A global bound for the second actual derivative integrand.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem abs_rotatedPairCurvature_le (β : ℝ) (hβ : 0 < β)
    (s : SSphere 2 → ℝ) (B : ℝ) (hB₀ : 0 ≤ B) (hB : ∀ x, |s x| ≤ B)
    (t : ℝ) (p : SSphere 2 × SSphere 2) :
    |rotatedPairCurvature β s t p| ≤ β * Real.exp β * (2 * B) ^ 2 := by
  rw [rotatedPairCurvature, abs_mul, abs_mul, abs_of_pos hβ,
    abs_of_nonneg (sq_nonneg (s p.1 - s p.2))]
  have hcurv := abs_circlePairCurvature_le β hβ (circleRotation s t p.1) (circleRotation s t p.2)
  have hspeed := speed_sub_sq_le s B hB₀ hB p.1 p.2
  gcongr

/-- The first derivative under the actual product probability measure.
Source: arXiv:2601.21366v2, §5.2, before `eq:Hess-bilinear.std`.
The bound is independent of transport time and of the measure's support. -/
theorem hasDerivAt_integral_rotatedPairKernel (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (fun r => ∫ p, rotatedPairKernel β s r p
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
      (∫ p, rotatedPairForce β s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) t := by
  obtain ⟨B, hB₀, hB⟩ := exists_pos_bound_continuous_compact s hs
  exact hasDerivAt_integral_on_compact
    ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
    (rotatedPairKernel β s) (rotatedPairForce β s)
    (continuous_rotatedPairKernel β s hs) (continuous_rotatedPairForce β s hs)
    (β * Real.exp β * (2 * B))
    (abs_rotatedPairForce_le β hβ s B hB)
    (fun r p => hasDerivAt_rotated_pair_kernel β s p.1 p.2 r) t

/-- Two dominated differentiations give the source's actual interaction
second variation for arbitrary support. Source: arXiv:2601.21366v2,
§5.2, `eq:Hess-bilinear.std`. This proves existence of the derivative,
as required when applying the strict SOPD inequality. -/
theorem hasDerivAt_deriv_integral_rotatedPairKernel (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ p, rotatedPairKernel β s r p
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))))
      (∫ p, rotatedPairCurvature β s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) t := by
  obtain ⟨B, hB₀, hB⟩ := exists_pos_bound_continuous_compact s hs
  exact hasDerivAt_deriv_integral_on_compact
    ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)))
    (rotatedPairKernel β s) (rotatedPairForce β s) (rotatedPairCurvature β s)
    (continuous_rotatedPairKernel β s hs) (continuous_rotatedPairForce β s hs)
    (continuous_rotatedPairCurvature β s hs)
    (β * Real.exp β * (2 * B)) (β * Real.exp β * (2 * B) ^ 2)
    (abs_rotatedPairForce_le β hβ s B hB)
    (abs_rotatedPairCurvature_le β hβ s B hB₀.le hB)
    (fun r p => hasDerivAt_rotated_pair_kernel β s p.1 p.2 r)
    (fun r p => hasDerivAt_rotated_pair_force β s p.1 p.2 r) t

/-- At the initial measure the transported curvature is exactly the
source's pair curvature times the squared initial speed difference.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem rotatedPairCurvature_zero (β : ℝ) (s : SSphere 2 → ℝ)
    (p : SSphere 2 × SSphere 2) :
    rotatedPairCurvature β s 0 p =
      β * circlePairCurvature β p.1 p.2 * (s p.1 - s p.2) ^ 2 := by
  simp only [rotatedPairCurvature, circleRotation_zero]

/-- Every nontrivial input is jointly realized at positive temperature
one and constant unit speed on the actual circle. A Dirac probability
may be used for the integration; the same proofs allow arbitrary measures. -/
example : (0 : ℝ) < 1 ∧ Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ x : SSphere 2, |(fun _ : SSphere 2 => (1 : ℝ)) x| ≤ 1) ∧
    HasDerivAt (deriv (fun r => ∫ p, rotatedPairKernel 1 (fun _ => 1) r p
      ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).prod
        (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)))) 0 0 := by
  refine ⟨one_pos, continuous_const, zero_le_one, fun _ => by norm_num, ?_⟩
  have h := hasDerivAt_deriv_integral_rotatedPairKernel 1 one_pos
    (Perspective.diracProb 2 (basePoint 1)) (fun _ => 1) continuous_const 0
  simpa [rotatedPairCurvature] using h

end Transformer.Perceptron
