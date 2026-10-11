/-
# Continuity of moments of actual weak solutions

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`.

The integrated weak equation makes every `C¹` test integral continuous
in time: its flux is interval integrable, and its primitive is continuous.
Linear and quadratic tests give continuity of the first and second moments.
In finite dimension the first vector moment is continuous as well.

At zero temperature the actual attention velocity is the tangent projection
of this first moment. These facts will give a full uniqueness witness for
the one-token cluster hypothesis, which quantifies over every weak solution.
No continuity or moment equation is assumed as part of a solution definition.
-/

import Transformer.Interpolation.Section5_PureAttention
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Integral.BoundedContinuousFunction
import Mathlib.MeasureTheory.Function.L2Space

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The coordinate inclusion of the compact sphere is integrable for every
spherical probability law. Source: arXiv:2411.04551v3, §1,
the barycenter in `eq: average.vf`. -/
theorem integrable_spherical_coe (μ : Perspective.ProbSphere d) :
    Integrable (fun x : SSphere d => (x : EucSpace d)) (μ : Measure (SSphere d)) := by
  let : MeasurableSpace (EucSpace d) := borel _
  let : BorelSpace (EucSpace d) := ⟨rfl⟩
  let f : BoundedContinuousFunction (SSphere d) (EucSpace d) :=
    BoundedContinuousFunction.mkOfCompact ⟨Subtype.val, continuous_subtype_val⟩
  exact f.integrable (μ : Measure (SSphere d))

/-- Every continuous scalar test on the compact sphere is integrable for
a spherical probability law. Source: arXiv:2411.04551v3, §1,
the weak formulation of `eq: cauchy.pb`. -/
theorem integrable_spherical_continuous (μ : Perspective.ProbSphere d)
    (f : SSphere d → ℝ) (hf : Continuous f) :
    Integrable f (μ : Measure (SSphere d)) := by
  let g : BoundedContinuousFunction (SSphere d) ℝ :=
    BoundedContinuousFunction.mkOfCompact ⟨f, hf⟩
  exact g.integrable (μ : Measure (SSphere d))

example : Continuous (fun _ : SSphere 1 => (1 : ℝ)) := continuous_const

/-- A spherical probability law has a first vector moment of norm at most
one. Source: arXiv:2411.04551v3, §1, `eq: average.vf`. -/
theorem norm_spherical_mean_le_one (μ : Perspective.ProbSphere d) :
    ‖∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))‖ ≤ 1 := by
  have hb : ∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)), ‖(x : EucSpace d)‖ ≤ 1 :=
    ae_of_all _ fun x => (mem_sphere_zero_iff_norm.mp x.2).le
  simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const hb

/-- Linear tests have their constant ambient vector as gradient.
Source: arXiv:2411.04551v3, §1, the gradient pairing in `eq: cauchy.pb`. -/
theorem gradient_inner_linear (a x : EucSpace d) :
    gradient (fun y : EucSpace d => inner (𝕜 := ℝ) a y) x = a := by
  have hg : HasGradientAt (fun y : EucSpace d => inner (𝕜 := ℝ) a y) a x := by
    rw [hasGradientAt_iff_hasFDerivAt]
    change HasFDerivAt (innerSL ℝ a) (innerSL ℝ a) x
    exact (innerSL ℝ a).hasFDerivAt
  exact hg.gradient

/-- **Every `C¹` test integral of an actual weak solution is continuous.**
The integral form of the source equation supplies a continuous primitive.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_test (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun t => ∫ x, φ (x : EucSpace d) ∂(μ t : Measure (SSphere d))) := by
  let g (s : ℝ) := ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
    (fullVF d θ (μ s) s (x : EucSpace d)) ∂(μ s : Measure (SSphere d))
  have hi (a b : ℝ) : IntervalIntegrable g volume a b :=
    ((hμ φ hφ a).1.symm).trans (hμ φ hφ b).1
  have heq : (fun t => ∫ x, φ (x : EucSpace d) ∂(μ t : Measure (SSphere d))) =
      (fun t => (∫ x, φ (x : EucSpace d) ∂(μ 0 : Measure (SSphere d))) +
        ∫ s in (0 : ℝ)..t, g s) := funext fun t => (hμ φ hφ t).2
  rw [heq]
  exact continuous_const.add (intervalIntegral.continuous_primitive hi 0)

/-- The complete weak-equation and test-regularity hypotheses hold for
a stationary nonzero Dirac law and the nonzero constant test. -/
example : cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) ∧
    ContDiff ℝ 1 (fun _ : EucSpace 1 => (1 : ℝ)) :=
  ⟨cauchyPB_pureAttention_dirac 0 (basePoint 0), contDiff_const⟩

/-- If a test flux is continuous, the actual integrated weak equation gives
its pointwise derivative. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`.
Continuity of the flux is an explicit hypothesis, since time switches in
the source can prevent a derivative at a switching time. -/
theorem hasDerivAt_cauchyPB_test (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ)
    (hflux : Continuous (fun s => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d θ (μ s) s (x : EucSpace d)) ∂(μ s : Measure (SSphere d)))) (t : ℝ) :
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
        (fullVF d θ (μ t) t (x : EucSpace d)) ∂(μ t : Measure (SSphere d))) t := by
  have heq : (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d))) =
      (fun s => (∫ x, φ (x : EucSpace d) ∂(μ 0 : Measure (SSphere d))) +
        ∫ r in (0 : ℝ)..s, ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
          (fullVF d θ (μ r) r (x : EucSpace d)) ∂(μ r : Measure (SSphere d))) :=
    funext fun s => (hμ φ hφ s).2
  rw [heq]
  exact (hflux.integral_hasStrictDerivAt 0 t).hasDerivAt.const_add _

example : cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) ∧
    ContDiff ℝ 1 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    Continuous (fun s => ∫ x, inner (𝕜 := ℝ) (gradient (fun _ : EucSpace 1 => (1 : ℝ))
      (x : EucSpace 1)) (fullVF 1 (pureAttentionParams 1 0)
        (Perspective.diracProb 1 (basePoint 0)) s (x : EucSpace 1))
      ∂(Perspective.diracProb 1 (basePoint 0) : Measure (SSphere 1))) := by
  refine ⟨cauchyPB_pureAttention_dirac 0 (basePoint 0), contDiff_const, ?_⟩
  simp only [gradient_fun_const, inner_zero_left, integral_zero]
  exact continuous_const

/-- Every linear moment of an actual weak solution is continuous.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_inner (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (a : EucSpace d) :
    Continuous (fun t => ∫ x, inner (𝕜 := ℝ) a (x : EucSpace d)
      ∂(μ t : Measure (SSphere d))) :=
  continuous_cauchyPB_test θ μ hμ _ (contDiff_const.inner ℝ contDiff_id)

example : cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) :=
  cauchyPB_pureAttention_dirac 0 (basePoint 0)

/-- Every quadratic product of linear moments is a continuous test integral
of the actual weak solution. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_inner_mul (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (a b : EucSpace d) :
    Continuous (fun t => ∫ x, inner (𝕜 := ℝ) a (x : EucSpace d) *
      inner (𝕜 := ℝ) b (x : EucSpace d) ∂(μ t : Measure (SSphere d))) :=
  continuous_cauchyPB_test θ μ hμ _
    ((contDiff_const.inner ℝ contDiff_id).mul (contDiff_const.inner ℝ contDiff_id))

example : cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) :=
  cauchyPB_pureAttention_dirac 0 (basePoint 0)

/-- **The actual first vector moment is continuous in time.** Its finitely
many coordinates are the continuous linear test integrals just proved.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: average.vf`. -/
theorem continuous_cauchyPB_mean (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) :
    Continuous (fun t => ∫ x : SSphere d, (x : EucSpace d)
      ∂(μ t : Measure (SSphere d))) := by
  let m (t : ℝ) := ∫ x : SSphere d, (x : EucSpace d) ∂(μ t : Measure (SSphere d))
  have hc (i : Fin d) : Continuous (fun t => (m t).ofLp i) := by
    have heq : (fun t => (m t).ofLp i) =
        (fun t => ∫ x, inner (𝕜 := ℝ) (EuclideanSpace.single i 1) (x : EucSpace d)
          ∂(μ t : Measure (SSphere d))) := by
      funext t
      rw [integral_inner (integrable_spherical_coe (μ t))]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
    rw [heq]
    exact continuous_cauchyPB_inner θ μ hμ _
  exact (PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).comp (continuous_pi_iff.mpr hc)

example : cauchyPB 3 (pureAttentionParams 3 0) (fun _ => Perspective.diracProb 3 (basePoint 2)) :=
  cauchyPB_pureAttention_dirac 0 (basePoint 2)

/-- At zero temperature the actual attention velocity is the projection
of the first vector moment. Source: arXiv:2411.04551v3, §1,
`eq: average.vf`, and §5, `prop:generic-discrete` at `β = 0`. -/
theorem fullVF_pureAttention_zero (μ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (pureAttentionParams d 0) μ t x =
      proj d x (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))) := by
  rw [fullVF_pureAttentionParams]
  simp only [Perspective.vectorField, Perspective.partitionMu, zero_mul, Real.exp_zero,
    integral_const, probReal_univ, one_smul, inv_one]

end Transformer.Interpolation
