/-
# Squared-distance tests of the original spherical weak equation

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: vf`, and `eq: cauchy.pb`.

The squared chordal distance to a unit particle is `2-2⟨y,x⟩`.
The moving linear-test rule therefore gives its actual weak derivative.
Both the particle and law use the original tangent field, so the derivative
is twice the integral of `⟨x-y,F(x)-F(y)⟩`.

A Lipschitz bound on the sphere controls this expression by the squared
distance itself. A probability with zero squared distance to one unit
point is exactly its Dirac mass. These estimates will force an original
weak solution with atomic initial data to remain on its particle path.
-/

import Transformer.Interpolation.Section1_AutonomousWeakTests

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Spherical squared distance is the corresponding original linear
moment. Source: arXiv:2411.04551v3, §1, the unit-sphere setting. -/
theorem integral_spherical_squared_distance (ρ : Perspective.ProbSphere d) (y : SSphere d) :
    (∫ x : SSphere d, ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ρ : Measure (SSphere d))) =
      2 - 2 * ∫ x : SSphere d, inner ℝ (y : EucSpace d) (x : EucSpace d) ∂(ρ : Measure (SSphere d)) := by
  have he : (fun x : SSphere d => ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2) =
      (fun x : SSphere d => 2 - 2 * inner ℝ (y : EucSpace d) (x : EucSpace d)) := by
    funext x
    rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2,
      real_inner_comm (x : EucSpace d) (y : EucSpace d)]
    ring
  have hi := integrable_spherical_continuous ρ
    (fun x : SSphere d => inner ℝ (y : EucSpace d) (x : EucSpace d)) (continuous_const.inner continuous_subtype_val)
  rw [he, integral_sub (integrable_const 2) (hi.const_mul 2), integral_const_mul, integral_const,
    probReal_univ, one_smul]

/-- The actual squared-distance test has the original relative-velocity
pairing as derivative. Tangency follows from `fullVF` itself.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_squared_distance (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, ‖(x : EucSpace d) - (Y s : EucSpace d)‖ ^ 2
      ∂(μ s : Measure (SSphere d)))
      (2 * ∫ x : SSphere d, inner ℝ ((x : EucSpace d) - (Y t : EucSpace d))
        (F x - F (Y t)) ∂(μ t : Measure (SSphere d))) t := by
  have he : (fun s => ∫ x : SSphere d, ‖(x : EucSpace d) - (Y s : EucSpace d)‖ ^ 2
      ∂(μ s : Measure (SSphere d))) =
      (fun s => 2 - 2 * ∫ x : SSphere d, inner ℝ (Y s : EucSpace d) (x : EucSpace d)
        ∂(μ s : Measure (SSphere d))) := funext fun s => integral_spherical_squared_distance (μ s) (Y s)
  rw [he]
  have htan (x : SSphere d) : inner ℝ (x : EucSpace d) (F x) = 0 := by
    rw [← hfield (μ t) t x]
    exact inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp x.2) _
  have hp (x : SSphere d) :
      inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t)) =
        -inner ℝ (Y t : EucSpace d) (F x) - inner ℝ (F (Y t)) (x : EucSpace d) := by
    simp only [inner_sub_left, inner_sub_right, htan x, htan (Y t),
      real_inner_comm (x : EucSpace d) (F (Y t))]
    ring
  have hi₁ : Integrable (fun x : SSphere d => inner ℝ (Y t : EucSpace d) (F x))
      (μ t : Measure (SSphere d)) := integrable_spherical_continuous (μ t) _
        (continuous_const.inner (hF.comp continuous_subtype_val))
  have hi₂ : Integrable (fun x : SSphere d => inner ℝ (F (Y t)) (x : EucSpace d))
      (μ t : Measure (SSphere d)) :=
    integrable_spherical_continuous (μ t) _ (continuous_const.inner continuous_subtype_val)
  have hflux : (∫ x : SSphere d, inner ℝ ((x : EucSpace d) - (Y t : EucSpace d))
      (F x - F (Y t)) ∂(μ t : Measure (SSphere d))) =
      -(∫ x : SSphere d, inner ℝ (Y t : EucSpace d) (F x) ∂(μ t : Measure (SSphere d))) -
        ∫ x : SSphere d, inner ℝ (F (Y t)) (x : EucSpace d) ∂(μ t : Measure (SSphere d)) := by
    rw [show (fun x : SSphere d => inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t))) =
      (fun x : SSphere d => -inner ℝ (Y t : EucSpace d) (F x) - inner ℝ (F (Y t)) (x : EucSpace d)) from funext hp]
    have hin : Integrable (fun x : SSphere d => -inner ℝ (Y t : EucSpace d) (F x))
        (μ t : Measure (SSphere d)) := hi₁.neg
    rw [integral_sub hin hi₂, integral_neg]
  convert ((hasDerivAt_cauchyPB_moving_inner θ F hF hfield μ hμ Y hY t).const_mul 2).const_sub 2 using 1
  rw [hflux]
  ring

/-- The original Lipschitz field bounds the relative-velocity pairing
by its Lipschitz constant times squared chordal distance.
Source: arXiv:2411.04551v3, §1, the well-posedness argument. -/
theorem integral_inner_difference_le_squared_distance (ρ : Perspective.ProbSphere d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F) (K : ℝ≥0)
    (hK : LipschitzOnWith K F (Sphere d)) (y : SSphere d) :
    (∫ x : SSphere d, inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y)
      ∂(ρ : Measure (SSphere d))) ≤
      (K : ℝ) * ∫ x : SSphere d, ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ρ : Measure (SSphere d)) := by
  have hi₁ : Integrable (fun x : SSphere d => inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y))
      (ρ : Measure (SSphere d)) := integrable_spherical_continuous ρ _
        ((continuous_subtype_val.sub continuous_const).inner ((hF.comp continuous_subtype_val).sub continuous_const))
  have hi₂ := integrable_spherical_continuous ρ
    (fun x : SSphere d => ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2) (by fun_prop)
  rw [← integral_const_mul]
  apply integral_mono_ae hi₁ (hi₂.const_mul _)
  exact ae_of_all _ fun x => by
    calc
      _ ≤ |inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y)| := le_abs_self _
      _ ≤ ‖(x : EucSpace d) - (y : EucSpace d)‖ * ‖F x - F y‖ := abs_real_inner_le_norm _ _
      _ ≤ ‖(x : EucSpace d) - (y : EucSpace d)‖ * ((K : ℝ) * ‖(x : EucSpace d) - (y : EucSpace d)‖) :=
        mul_le_mul_of_nonneg_left (lipschitzOnWith_iff_norm_sub_le.mp hK x.2 y.2) (norm_nonneg _)
      _ = _ := by ring

/-- Zero average squared distance forces the whole original probability
to equal the point mass. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, the atomic-data case of well-posedness. -/
theorem eq_diracProb_of_integral_squared_distance_eq_zero (ρ : Perspective.ProbSphere d) (y : SSphere d)
    (hzero : (∫ x : SSphere d, ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ρ : Measure (SSphere d))) = 0) :
    ρ = Perspective.diracProb d y := by
  have hi := integrable_spherical_continuous ρ
    (fun x : SSphere d => ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2) (by fun_prop)
  have hae := (integral_eq_zero_iff_of_nonneg_ae (ae_of_all _ fun x : SSphere d => sq_nonneg ‖(x : EucSpace d) - (y : EucSpace d)‖) hi).mp hzero
  have he : (id : SSphere d → SSphere d) =ᵐ[(ρ : Measure (SSphere d))] (fun _ => y) :=
    hae.mono fun x hx => by
      have hn : ‖(x : EucSpace d) - (y : EucSpace d)‖ = 0 := (sq_eq_zero_iff).mp hx
      exact Subtype.ext (sub_eq_zero.mp (norm_eq_zero.mp hn))
  apply Subtype.ext
  change (ρ : Measure (SSphere d)) = Measure.dirac y
  simpa only [Measure.map_id, Measure.map_const, measure_univ, one_smul] using Measure.map_congr he

/-- A genuine projection neuron, actual weak law and unit particle path,
and a compact-sphere Lipschitz bound jointly satisfy all dynamical premises.
Their actual common Dirac initial law has zero squared distance. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2)
    (μ : ℝ → Perspective.ProbSphere 2) (Y : ℝ → SSphere 2) (K : ℝ≥0),
    Continuous F ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    LipschitzOnWith K F (Sphere 2) ∧ cauchyPB 2 θ μ ∧
    (∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t) ∧
    μ 0 = Perspective.diracProb 2 (Y 0) ∧ (0 : ℝ) ≤ 1 ∧
    (∫ x : SSphere 2, ‖(x : EucSpace 2) - (Y 0 : EucSpace 2)‖ ^ 2 ∂(μ 0 : Measure (SSphere 2))) = 0 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 (basePoint 1))
  let Y : ℝ → SSphere 2 := fun _ => basePoint 1
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  have hfield (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2) : fullVF 2 θ ρ s x = F x := by
    rw [fullVF_reluRidgeParams]
    simp [F]
  have hinit : μ 0 = Perspective.diracProb 2 (Y 0) := ridgeLaw_zero a 0 1 _
  refine ⟨θ, F, μ, Y, K, hF.continuous, hfield, hK.mono hsphere,
    cauchyPB_ridgeLaw _ _ _ _ _, ?_, hinit, zero_le_one, ?_⟩
  · intro t
    have ha : ‖a‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
    have he : F (Y t) = 0 := by
      change proj 2 a a = 0
      simpa only [one_smul] using proj_smul_self ha 1
    rw [he]
    exact hasDerivAt_const t _
  · rw [hinit]
    change (∫ x : SSphere 2, ‖(x : EucSpace 2) - (Y 0 : EucSpace 2)‖ ^ 2 ∂Measure.dirac (Y 0)) = 0
    simp

end Transformer.Interpolation
