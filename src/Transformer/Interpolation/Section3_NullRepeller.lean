module

public import Transformer.Interpolation.Section3_ProjectionContraction
public import Transformer.Interpolation.Section3_QuadrantGeometry
public import Transformer.Interpolation.Section3_TwoRidgeLaw
public import Transformer.Interpolation.Section3_CollapseLaws
public import Transformer.Causal.UniformAtoms
public import Mathlib.MeasureTheory.Measure.Typeclasses.SFinite

/-!
# A nearby attracting direction with a null repelling pole

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
for the case where the source ball covers the entire sphere.

An arbitrary initial probability has at most countably many atoms.
The uniform spherical law has no atoms in dimension at least two and
charges every nonempty open set, so such a set contains a point carrying
zero initial mass. Applying the antipode gives an attracting direction
whose repelling pole is null, arbitrarily close to any desired target.

The actual two-neuron flow with one constant activation and a zero second
output is exactly the original projection flow. Hence almost every initial
particle converges to that direction, without assuming the initial law is
diffuse. Chordal balls of radius greater than `2` are the whole sphere.
-/

@[expose] public section

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every nonempty spherical open set contains a point with zero mass
under the given initial probability. Auxiliary for arXiv:2411.04551v3,
Appendix, the whole-sphere case of `lem: two.balls`. -/
theorem exists_null_singleton_in_open (hd : 2 ≤ d) (ν : Perspective.ProbSphere d)
    (U : Set (SSphere d)) (hU : IsOpen U) (hne : U.Nonempty) :
    ∃ x ∈ U, (ν : Measure (SSphere d)) {x} = 0 := by
  let A := {x : SSphere d | 0 < (ν : Measure (SSphere d)) {x}}
  have hcount : A.Countable := Measure.countable_meas_pos_of_disjoint_iUnion
    (As := fun x : SSphere d => ({x} : Set (SSphere d))) (fun x => measurableSet_singleton x)
    (fun _ _ hne => disjoint_singleton.mpr hne)
  let : NullSingletonClass (Causal.uniformSphere d) := ⟨Causal.uniformSphere_singleton d hd⟩
  have hnull := hcount.measure_zero (Causal.uniformSphere d)
  have hpos := Causal.uniformSphere_pos_of_isOpen d hU hne
  by_contra hnot
  have hsub : U ⊆ A := by
    intro x hx
    change 0 < (ν : Measure (SSphere d)) {x}
    exact bot_lt_iff_ne_bot.mpr (fun hzero => hnot ⟨x, hx, hzero⟩)
  exact hpos.ne' (measure_mono_null hsub hnull)

/-- A target direction can be chosen in any nonempty open region with
zero initial mass at its antipode. Auxiliary for arXiv:2411.04551v3,
Appendix, the whole-sphere case of `lem: two.balls`. -/
theorem exists_target_null_antipode (hd : 2 ≤ d) (ν : Perspective.ProbSphere d)
    (U : Set (SSphere d)) (hU : IsOpen U) (hne : U.Nonempty) :
    ∃ p ∈ U, (ν : Measure (SSphere d)) {antipode d p} = 0 := by
  have hc : Continuous (antipode d) := Continuous.subtype_mk continuous_subtype_val.neg _
  have hnon : (antipode d ⁻¹' U).Nonempty := by
    obtain ⟨x, hx⟩ := hne
    exact ⟨antipode d x, by simpa only [mem_preimage, antipode_antipode] using hx⟩
  obtain ⟨q, hq, hnull⟩ := exists_null_singleton_in_open hd ν (antipode d ⁻¹' U) (hU.preimage hc) hnon
  exact ⟨antipode d q, hq, by simpa only [antipode_antipode] using hnull⟩

/-- Such an attracting direction can be chosen arbitrarily close to a
prescribed target while staying in its requested open neighborhood.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_target_null_antipode_near (hd : 2 ≤ d) (ν : Perspective.ProbSphere d)
    (U : Set (SSphere d)) (hU : IsOpen U) (ω : SSphere d) (hω : ω ∈ U) (ε : ℝ) (hε : 0 < ε) :
    ∃ p ∈ U, dist p ω < ε ∧ (ν : Measure (SSphere d)) {antipode d p} = 0 := by
  obtain ⟨p, hp, hnull⟩ := exists_target_null_antipode hd ν (U ∩ Metric.ball ω ε)
    (hU.inter Metric.isOpen_ball) ⟨ω, hω, by simpa only [Metric.mem_ball, dist_self] using hε⟩
  exact ⟨p, hp.1, hp.2, hnull⟩

/-- A null repelling pole is excluded for almost every initial particle.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem ae_ne_antipode (ν : Perspective.ProbSphere d) (p : SSphere d)
    (hnull : (ν : Measure (SSphere d)) {antipode d p} = 0) :
    ∀ᵐ x ∂(ν : Measure (SSphere d)), x ≠ antipode d p := by
  apply ae_iff.mpr
  simpa only [not_not, ofPred_eq_eq_singleton] using hnull

/-- The actual constant-activation two-neuron flow equals the original
projection flow. Source: arXiv:2411.04551v3, §1, `eq: vf`, and §3.1,
the projection dynamics used for `lem: first.quadrant`. -/
theorem twoRidgeFlow_projection (p x : SSphere d) (t : ℝ) :
    twoRidgeFlow (p : EucSpace d) 0 1 0 0 0 x t = sphereProjectionFlow p t x := by
  have hd (s : ℝ) : HasDerivAt (fun r => (sphereProjectionFlow p r x : EucSpace d))
      (twoRidgeField (p : EucSpace d) 0 1 0 0 0 (sphereProjectionFlow p s x)) s := by
    simpa [twoRidgeField, proj] using hasDerivAt_sphereProjectionFlow p x s
  have he := twoRidgeTrajectory_unique (p : EucSpace d) 0 1 0 0 0
    (twoRidgeFlow (p : EucSpace d) 0 1 0 0 0 x) (fun s => sphereProjectionFlow p s x)
    (hasDerivAt_twoRidgeFlow _ _ _ _ _ _ x) hd (by simp only [twoRidgeFlow_zero, sphereProjectionFlow_zero])
  exact congrFun he t

/-- The actual original flow converges almost everywhere when its
repelling pole has zero initial mass. Auxiliary for arXiv:2411.04551v3,
Appendix, the whole-sphere case of `lem: two.balls`. -/
theorem ae_tendsto_twoRidge_projection (ν : Perspective.ProbSphere d) (p : SSphere d)
    (hnull : (ν : Measure (SSphere d)) {antipode d p} = 0) :
    ∀ᵐ x ∂(ν : Measure (SSphere d)),
      Tendsto (twoRidgeFlow (p : EucSpace d) 0 1 0 0 0 x) atTop (nhds p) := by
  filter_upwards [ae_ne_antipode ν p hnull] with x hx
  change Tendsto (fun t => twoRidgeFlow (p : EucSpace d) 0 1 0 0 0 x t) atTop (nhds p)
  simpa only [twoRidgeFlow_projection] using tendsto_sphereProjectionFlow p x hx

/-- A chordal ball of radius greater than `2` covers the unit sphere.
Auxiliary for arXiv:2411.04551v3, Appendix, the radius cases of
`lem: two.balls`. -/
theorem spherical_ball_eq_univ_of_two_lt (z : SSphere d) (R : ℝ) (hR : 2 < R) :
    Metric.ball z R = Set.univ := by
  apply eq_univ_of_forall
  intro x
  have hn := norm_sub_le (x : EucSpace d) (z : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp z.2] at hn
  change ‖(x : EucSpace d) - (z : EucSpace d)‖ < R
  linarith

/-- Almost-everywhere attraction to a nearby point suffices for the
Wasserstein bound to the originally prescribed target. Auxiliary for
arXiv:2411.04551v3, Appendix, the whole-sphere case of `lem: two.balls`. -/
theorem eventually_W2_map_dirac_lt_of_ae_tendsto (ν : Perspective.ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d) (hΦ : ∀ t : ℝ, Measurable (Φ t)) (p ω : SSphere d)
    (hlim : ∀ᵐ x ∂(ν : Measure (SSphere d)), Tendsto (fun t => Φ t x) atTop (nhds p))
    (ε : ℝ) (hpω : dist p ω < ε) :
    ∀ᶠ t : ℝ in atTop, W2 d (Measure.map (Φ t) (ν : Measure (SSphere d))) (Measure.dirac ω) < ε := by
  have hc := tendsto_integral_spherical_distance_sq ν Φ hΦ (fun _ => ω) (fun _ => p) measurable_const hlim
  have hc' : Tendsto (fun t : ℝ => ∫ x : SSphere d,
      ‖(Φ t x : EucSpace d) - (ω : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d)))
      atTop (nhds (‖(p : EucSpace d) - (ω : EucSpace d)‖ ^ 2)) := by simpa using hc
  have hs := (continuous_sqrt.tendsto _).comp hc'
  rw [sqrt_sq (norm_nonneg _)] at hs
  have hdist : ‖(p : EucSpace d) - (ω : EucSpace d)‖ < ε := hpω
  filter_upwards [hs.eventually (gt_mem_nhds hdist)] with t ht
  have hm := monge d ν (Φ t) (fun _ => ω) (hΦ t) measurable_const
  simpa only [Measure.map_const, measure_univ, one_smul] using hm.trans_lt ht

/-- The actual projection flow supplies all the almost-everywhere cost
hypotheses even for a probability concentrated at a single atom. -/
example (p : SSphere 2) :
    (∀ t : ℝ, Measurable (fun x : SSphere 2 => twoRidgeFlow (p : EucSpace 2) 0 1 0 0 0 x t)) ∧
    (∀ᵐ x ∂(Perspective.diracProb 2 p : Measure (SSphere 2)),
      Tendsto (fun t => twoRidgeFlow (p : EucSpace 2) 0 1 0 0 0 x t) atTop (nhds p)) ∧
    dist p p < (1 : ℝ) := by
  refine ⟨fun t => measurable_twoRidgeFlow_initial _ _ _ _ _ _ t, ?_, by simp⟩
  apply ae_tendsto_twoRidge_projection
  simp [Perspective.diracProb, (antipode_ne 2 p).symm]

/-- A probability with a genuine atom, a nonempty open region, a target
with a null repelling pole and positive accuracy jointly satisfy all
selection, almost-everywhere convergence and whole-ball hypotheses. -/
example : ∃ (ν : Perspective.ProbSphere 2) (p ω : SSphere 2)
    (U : Set (SSphere 2)) (ε R : ℝ),
    2 ≤ 2 ∧ IsOpen U ∧ U.Nonempty ∧ ω ∈ U ∧ 0 < ε ∧
    (ν : Measure (SSphere 2)) {antipode 2 p} = 0 ∧ 2 < R := by
  let p : SSphere 2 := basePoint 1
  refine ⟨Perspective.diracProb 2 p, p, p, univ, 1, 3, le_rfl,
    isOpen_univ, ⟨p, mem_univ p⟩, mem_univ p, zero_lt_one, ?_, by norm_num⟩
  simp [Perspective.diracProb, (antipode_ne 2 p).symm]

end Transformer.Interpolation
