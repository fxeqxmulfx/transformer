/-
# Actual ReLU activation caps and scalar particle dynamics

ArXiv:2411.04551v3, Appendix, `lem: two.balls`, `eq: activation.ball`
and `eq: dissipation.sphere`, together with §1, `eq: ips` and `eq: cauchy.pb`.

The source sets `U=-1 zᵀ`, `b=cos(R)1`, whose scalar activation is
`cos(R)-⟨x,z⟩`; this activates the complement of the stated geodesic ball.
For a cap `⟨x,z⟩>c`, the inward parameters are instead `u=-z`, `b=-c`.
Here `c=1-R²/2` expresses the manuscript's ball using chordal radius `R`.

The actual complete ridge flow preserves its activation region, since its
inverse also fixes every inactive point. Thus every original weak solution
preserves the mass of that region. Its scalar output moment is monotone,
and the cap margin has its exact original particle derivative. These are
the inputs for a quantitative concentration estimate inside the cap.
-/

import Transformer.Interpolation.Section3_RidgeWellPosed

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual chordal ball is the corresponding spherical cap.
Source: arXiv:2411.04551v3, §1.5 and Appendix, `eq: activation.ball`. -/
theorem mem_spherical_ball_iff_inner (z x : SSphere d) (R : ℝ) (hR : 0 < R) :
    x ∈ Metric.ball z R ↔ 1 - R ^ 2 / 2 < inner ℝ (x : EucSpace d) (z : EucSpace d) := by
  have he : ‖(x : EucSpace d) - (z : EucSpace d)‖ ^ 2 =
      2 - 2 * inner ℝ (x : EucSpace d) (z : EucSpace d) := by
    rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp z.2]
    ring
  rw [Metric.mem_ball, Subtype.dist_eq, dist_eq_norm, ← sq_lt_sq₀ (norm_nonneg _) hR.le, he]
  constructor <;> intro h <;> linarith


/-- Corrected activation sign in the source's `eq: activation.ball`:
`u=-z`, `b=-(1-R²/2)` activates the ball, whereas the source's opposite
sign activates its complement. Source: arXiv:2411.04551v3, Appendix. -/
theorem inward_ridge_activation_ball (z x : SSphere d) (R : ℝ) (hR : 0 < R) :
    0 < -(1 - R ^ 2 / 2) - inner ℝ (x : EucSpace d) (-(z : EucSpace d)) ↔ x ∈ Metric.ball z R := by
  rw [mem_spherical_ball_iff_inner z x R hR, inner_neg_right]
  constructor <;> intro h <;> linarith

/-- The actual original neuron matrices realize the inward cap field
with the corrected sign. Source: arXiv:2411.04551v3, Appendix,
`eq: activation.ball` and §1, `eq: vf`. -/
theorem fullVF_inward_ball (i : Fin d) (p : EucSpace d) (z : SSphere d) (R : ℝ)
    (ρ : Perspective.ProbSphere d) (t : ℝ) (x : SSphere d) :
    fullVF d (fun _ => reluRidgeParams i p (-(z : EucSpace d)) (-(1 - R ^ 2 / 2))) ρ t x =
      max (inner ℝ (x : EucSpace d) (z : EucSpace d) - (1 - R ^ 2 / 2)) 0 • proj d x p := by
  rw [fullVF_reluRidgeParams, inner_neg_right]
  congr 2
  ring

/-- The actual scalar moment derivative along the original ridge particle.
Source: arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`. -/
theorem hasDerivAt_inner_ridgeFlow (p u v : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ (ridgeFlow p u b x₀ s : EucSpace d) v)
      (max (b - inner ℝ (ridgeFlow p u b x₀ t : EucSpace d) u) 0 *
        (inner ℝ p v - inner ℝ (ridgeFlow p u b x₀ t : EucSpace d) p *
          inner ℝ (ridgeFlow p u b x₀ t : EucSpace d) v)) t := by
  convert (hasDerivAt_ridgeFlow p u b x₀ t).inner ℝ (hasDerivAt_const t v) using 1
  simp only [inner_zero_right, zero_add, real_inner_smul_left, proj, inner_sub_left]

/-- Actual spherical particles have output moment in the unit interval.
Source: arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`. -/
theorem abs_inner_ridgeFlow_output_le_one (p : SSphere d) (u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    |inner ℝ (ridgeFlow (p : EucSpace d) u b x₀ t : EucSpace d) (p : EucSpace d)| ≤ 1 := by
  have h := abs_real_inner_le_norm (ridgeFlow (p : EucSpace d) u b x₀ t : EucSpace d) (p : EucSpace d)
  simpa only [mem_sphere_zero_iff_norm.mp (ridgeFlow (p : EucSpace d) u b x₀ t).2,
    mem_sphere_zero_iff_norm.mp p.2, mul_one] using h

/-- The output moment increases along every actual ridge particle.
Source: arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`. -/
theorem monotone_inner_ridgeFlow_output (p : SSphere d) (u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    Monotone (fun t => inner ℝ (ridgeFlow (p : EucSpace d) u b x₀ t : EucSpace d) (p : EucSpace d)) := by
  have hd (t : ℝ) := hasDerivAt_inner_ridgeFlow (p : EucSpace d) u (p : EucSpace d) b x₀ t
  apply monotone_of_deriv_nonneg (fun t => (hd t).differentiableAt)
  intro t
  rw [(hd t).deriv, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
  have hb := abs_le.mp (abs_inner_ridgeFlow_output_le_one p u b x₀ t)
  apply mul_nonneg (le_max_right _ _)
  nlinarith

/-- The actual inward cap margin has the stated scalar derivative.
The corrected activation sign is retained. Source: arXiv:2411.04551v3,
Appendix, `eq: activation.ball` and `eq: dissipation.sphere`. -/
theorem hasDerivAt_ridge_cap_activation (p z : EucSpace d) (c : ℝ) (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ (ridgeFlow p (-z) (-c) x₀ s : EucSpace d) z - c)
      (max (inner ℝ (ridgeFlow p (-z) (-c) x₀ t : EucSpace d) z - c) 0 *
        (inner ℝ p z - inner ℝ (ridgeFlow p (-z) (-c) x₀ t : EucSpace d) p *
          inner ℝ (ridgeFlow p (-z) (-c) x₀ t : EucSpace d) z)) t := by
  convert (hasDerivAt_inner_ridgeFlow p (-z) z (-c) x₀ t).sub_const c using 1
  simp only [inner_neg_right, sub_eq_add_neg, neg_neg, add_comm]

/-- The actual ridge flow preserves its active cap at every real time.
Source: arXiv:2411.04551v3, Appendix, the flow in `lem: two.balls`. -/
theorem ridgeFlow_active_iff (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    0 < b - inner ℝ (ridgeFlow p u b x₀ t : EucSpace d) u ↔
      0 < b - inner ℝ (x₀ : EucSpace d) u := by
  constructor
  · intro ht
    by_contra hx
    rw [ridgeFlow_inactive p u b x₀ (not_lt.mp hx) t] at ht
    exact hx ht
  · intro hx
    by_contra ht
    have hh := ridgeFlow_inactive p u b (ridgeFlow p u b x₀ t) (not_lt.mp ht) (-t)
    rw [ridgeFlow_inverse] at hh
    rw [← hh] at ht
    exact ht hx

/-- The original transported probability preserves its active-cap mass.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem ridgeLaw_active_mass (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    (ridgeLaw p u b ν t : Measure (SSphere d)) {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} =
      (ν : Measure (SSphere d)) {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} := by
  have hs : MeasurableSet {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} :=
    (isOpen_lt continuous_const (continuous_const.sub (continuous_subtype_val.inner continuous_const))).measurableSet
  rw [coe_ridgeLaw, Measure.map_apply (measurable_ridgeFlow_initial p u b t) hs]
  congr 1
  ext x
  exact ridgeFlow_active_iff p u b x t

/-- Every actual original weak law preserves the active-cap mass.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem cauchyPB_ridge_active_mass (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ) (t : ℝ) :
    (μ t : Measure (SSphere d)) {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} =
      (μ 0 : Measure (SSphere d)) {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} := by
  rw [cauchyPB_ridge_transport i p u b μ hμ t]
  exact ridgeLaw_active_mass p u b (μ 0) t

/-- A nonzero neuron and an actually activated transverse atom jointly
realize the positive-radius and original weak-solution hypotheses. -/
example : ∃ (p : SSphere 2) (u : EucSpace 2) (x₀ : SSphere 2) (μ : ℝ → Perspective.ProbSphere 2),
    (0 : ℝ) < 1 ∧ 0 < 0 - inner ℝ (x₀ : EucSpace 2) u ∧
    cauchyPB 2 (fun _ => reluRidgeParams (0 : Fin 2) p u 0) μ := by
  let p : SSphere 2 := basePoint 1
  let u : EucSpace 2 := EuclideanSpace.single 1 1
  let x₀ : SSphere 2 := ⟨-u, by simp [u]⟩
  refine ⟨p, u, x₀, ridgeLaw p u 0 (Perspective.diracProb 2 x₀), zero_lt_one, ?_,
    cauchyPB_ridgeLaw _ _ _ _ _⟩
  have hu : ‖u‖ = 1 := by simp [u]
  change 0 < 0 - inner ℝ (-u) u
  rw [inner_neg_left, real_inner_self_eq_norm_sq, hu]
  norm_num

end Transformer.Interpolation
