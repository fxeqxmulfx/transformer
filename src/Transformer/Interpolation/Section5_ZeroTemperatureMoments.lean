module

public import Transformer.Interpolation.Section1_WeakContinuity
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Moment equations at zero temperature

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`, and §5,
`prop:generic-discrete` with `β = 0`.

The source's pure-attention velocity becomes the projection of the mean.
Its linear test flux is a polynomial in the first and second moments.
Their continuity, derived from the actual integrated weak equation, therefore
gives a pointwise differential equation for every first scalar moment.

The second expression for this flux isolates the concentration defect
`1 - ⟪a, x⟫`. It will control all weak solutions starting at a Dirac law,
rather than merely the particular solution induced by a particle trajectory.
All integrals here are the actual probability-measure integrals on the sphere.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Integrating the tangent projection separates its constant mean term
from a quadratic scalar moment. Source: arXiv:2411.04551v3, §1, `eq: vf`.
This identity uses arbitrary ambient test and velocity vectors. -/
theorem integral_projection_inner (μ : Perspective.ProbSphere d) (a b : EucSpace d) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) a (proj d (x : EucSpace d) b)
      ∂(μ : Measure (SSphere d))) = inner (𝕜 := ℝ) a b -
        ∫ x : SSphere d, inner (𝕜 := ℝ) a (x : EucSpace d) *
          inner (𝕜 := ℝ) (x : EucSpace d) b ∂(μ : Measure (SSphere d)) := by
  have hi : Integrable (fun x : SSphere d => inner (𝕜 := ℝ) a (x : EucSpace d) *
      inner (𝕜 := ℝ) (x : EucSpace d) b) (μ : Measure (SSphere d)) :=
    integrable_spherical_continuous μ _ (by fun_prop)
  simp only [proj, inner_sub_right, real_inner_smul_right]
  simp_rw [mul_comm (inner (𝕜 := ℝ) _ b)]
  rw [integral_sub (integrable_const _) hi, integral_const, probReal_univ, one_smul]

/-- The linear zero-temperature flux is a polynomial in finitely many
first and second moments. Source: arXiv:2411.04551v3, §1, `eq: vf`,
specialized to the parameters of §5, `prop:generic-discrete`. -/
theorem zero_temperature_flux_moments (μ : Perspective.ProbSphere d) (a : EucSpace d) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) a (proj d (x : EucSpace d)
      (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))))
      ∂(μ : Measure (SSphere d))) =
    inner (𝕜 := ℝ) a (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))) -
      ∑ k : Fin d, (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))).ofLp k *
        ∫ x : SSphere d, inner (𝕜 := ℝ) a (x : EucSpace d) *
          inner (𝕜 := ℝ) (EuclideanSpace.single k 1) (x : EucSpace d)
          ∂(μ : Measure (SSphere d)) := by
  let m := ∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))
  have heq (x : SSphere d) : inner (𝕜 := ℝ) a (x : EucSpace d) *
      inner (𝕜 := ℝ) (x : EucSpace d) m =
      ∑ k : Fin d, m.ofLp k * (inner (𝕜 := ℝ) a (x : EucSpace d) *
        inner (𝕜 := ℝ) (EuclideanSpace.single k 1) (x : EucSpace d)) := by
    rw [PiLp.inner_apply (x : EucSpace d) m, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    simp only [EuclideanSpace.inner_single_left, map_one, one_mul, Real.inner_apply]
    ring
  have hp : (∫ x : SSphere d, inner (𝕜 := ℝ) a (x : EucSpace d) *
      inner (𝕜 := ℝ) (x : EucSpace d) m ∂(μ : Measure (SSphere d))) =
      ∑ k : Fin d, m.ofLp k * ∫ x : SSphere d, inner (𝕜 := ℝ) a (x : EucSpace d) *
        inner (𝕜 := ℝ) (EuclideanSpace.single k 1) (x : EucSpace d)
        ∂(μ : Measure (SSphere d)) := by
    calc
      _ = ∫ x : SSphere d, ∑ k : Fin d, m.ofLp k *
          (inner (𝕜 := ℝ) a (x : EucSpace d) *
            inner (𝕜 := ℝ) (EuclideanSpace.single k 1) (x : EucSpace d))
          ∂(μ : Measure (SSphere d)) := integral_congr_ae (ae_of_all _ heq)
      _ = _ := by
        rw [integral_finsetSum]
        · apply Finset.sum_congr rfl
          intro k _
          exact integral_const_mul _ _
        · intro k _
          exact integrable_spherical_continuous μ _ (by fun_prop)
  rw [integral_projection_inner, hp]

/-- **The actual linear flux of every zero-temperature weak solution is
continuous.** No continuity of the law itself is inserted as a hypothesis.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`. -/
theorem continuous_cauchyPB_zero_flux (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ) (a : EucSpace d) :
    Continuous (fun t => ∫ x : SSphere d, inner (𝕜 := ℝ) a
      (fullVF d (pureAttentionParams d 0) (μ t) t (x : EucSpace d))
      ∂(μ t : Measure (SSphere d))) := by
  simp_rw [fullVF_pureAttention_zero, zero_temperature_flux_moments]
  have hm := continuous_cauchyPB_mean (pureAttentionParams d 0) μ hμ
  refine (continuous_const.inner hm).sub (continuous_finsetSum _ ?_)
  intro k _
  exact ((PiLp.continuous_apply (p := 2) (β := fun _ : Fin d => ℝ) k).comp hm).mul
    (continuous_cauchyPB_inner_mul (pureAttentionParams d 0) μ hμ a
      (EuclideanSpace.single k 1))

/-- The entire weak-solution hypothesis holds for an actual nonzero
stationary Dirac law in the dimension used by `prop:generic-discrete`. -/
example : cauchyPB 3 (pureAttentionParams 3 0)
    (fun _ => Perspective.diracProb 3 (basePoint 2)) :=
  cauchyPB_pureAttention_dirac 0 (basePoint 2)

/-- **The first scalar moment satisfies its actual attention equation.**
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` at `β = 0`.
The differentiability is proved from the weak equation and the moment
formula, without assuming a classical solution. -/
theorem hasDerivAt_cauchyPB_zero_inner (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ) (a : EucSpace d) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, inner (𝕜 := ℝ) a (x : EucSpace d)
      ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner (𝕜 := ℝ) a
        (proj d (x : EucSpace d) (∫ y : SSphere d, (y : EucSpace d)
          ∂(μ t : Measure (SSphere d)))) ∂(μ t : Measure (SSphere d))) t := by
  have hf : ContDiff ℝ 1 (fun x : EucSpace d => inner (𝕜 := ℝ) a x) :=
    contDiff_const.inner ℝ contDiff_id
  have hc : Continuous (fun s => ∫ x : SSphere d,
      inner (𝕜 := ℝ) (gradient (fun y : EucSpace d => inner (𝕜 := ℝ) a y)
        (x : EucSpace d)) (fullVF d (pureAttentionParams d 0) (μ s) s
          (x : EucSpace d)) ∂(μ s : Measure (SSphere d))) := by
    simp only [gradient_inner_linear]
    exact continuous_cauchyPB_zero_flux μ hμ a
  simpa only [gradient_inner_linear, fullVF_pureAttention_zero] using
    hasDerivAt_cauchyPB_test (pureAttentionParams d 0) μ hμ _ hf hc t

example : cauchyPB 1 (pureAttentionParams 1 0)
    (fun _ => Perspective.diracProb 1 (basePoint 0)) :=
  cauchyPB_pureAttention_dirac 0 (basePoint 0)

/-- A second expression for the actual linear flux isolates the defect
`1 - ⟪a, x⟫`. Source: arXiv:2411.04551v3, §1, `eq: vf` at `β = 0`.
This is an algebraic consequence of the spherical first moment. -/
theorem zero_temperature_flux_defect (μ : Perspective.ProbSphere d) (a : EucSpace d) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) a (proj d (x : EucSpace d)
      (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))))
      ∂(μ : Measure (SSphere d))) =
    inner (𝕜 := ℝ) a (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))) -
      ‖∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))‖ ^ 2 +
      ∫ x : SSphere d, (1 - inner (𝕜 := ℝ) a (x : EucSpace d)) *
        inner (𝕜 := ℝ) (x : EucSpace d)
          (∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d)))
          ∂(μ : Measure (SSphere d)) := by
  let m := ∫ y : SSphere d, (y : EucSpace d) ∂(μ : Measure (SSphere d))
  have hi₁ := integrable_spherical_continuous μ
    (fun x : SSphere d => inner (𝕜 := ℝ) (x : EucSpace d) m) (by fun_prop)
  have hi₂ := integrable_spherical_continuous μ
    (fun x : SSphere d => inner (𝕜 := ℝ) a (x : EucSpace d) *
      inner (𝕜 := ℝ) (x : EucSpace d) m) (by fun_prop)
  have hm : (∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) m
      ∂(μ : Measure (SSphere d))) = ‖m‖ ^ 2 := by
    calc
      _ = ∫ x : SSphere d, inner (𝕜 := ℝ) m (x : EucSpace d)
          ∂(μ : Measure (SSphere d)) :=
        integral_congr_ae (ae_of_all _ fun x => real_inner_comm _ _)
      _ = ‖m‖ ^ 2 := by
        rw [integral_inner (integrable_spherical_coe μ), real_inner_self_eq_norm_sq]
  rw [integral_projection_inner]
  simp_rw [sub_mul, one_mul]
  rw [integral_sub hi₁ hi₂, hm]
  ring

end Transformer.Interpolation
