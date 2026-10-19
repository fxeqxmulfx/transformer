module

public import Transformer.Interpolation.Section1_AutonomousTransport
public import Transformer.Interpolation.Section3_RidgeTransport

/-!
# Original ReLU weak well-posedness for every spherical probability

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: cauchy.pb` and the
well-posedness argument; §3.2, `lem: perturbation`; Appendix,
the fixed-outside clauses of `lem: two.balls`.

Every actual weak solution of the explicit single-neuron controls is
exactly the original ridge pushforward of its initial probability.
This gives global existence and uniqueness for arbitrary laws, beyond
the Dirac initial-data case. The actual solution maps compose in time,
have inverse given by time reversal, and are bijections on probabilities.

An initially inactive law stays fixed for every real time. This applies
to every original weak solution, so later selective transport arguments
can use the fixed region without assuming uniqueness or a flow formula.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every actual weak law is its original ridge-flow pushforward, for
all real times and arbitrary initial probabilities.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem cauchyPB_ridge_transport (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ) (t : ℝ) :
    μ t = ridgeLaw p u b (μ 0) t := by
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_ridge_field p u b
  have hF : Continuous (fun x : EucSpace d => max (b - inner ℝ x u) 0 • proj d x p) := by
    unfold proj
    fun_prop
  apply Subtype.ext
  change (μ t : Measure (SSphere d)) = (ridgeLaw p u b (μ 0) t : Measure (SSphere d))
  rw [coe_ridgeLaw]
  exact cauchyPB_autonomous_transport _ _ hF (fun ρ s x => fullVF_reluRidgeParams i p u b ρ s x)
    K hK (fun x t => ridgeFlow p u b x t) (hasDerivAt_ridgeFlow p u b) (ridgeFlow_zero p u b)
    (exists_lipschitzWith_ridgeFlow p u b) μ hμ t

/-- The whole original weak curve is the constructed flow law from its
prescribed arbitrary initial probability. Source: arXiv:2411.04551v3,
§1, well-posedness of `eq: cauchy.pb`. -/
theorem cauchyPB_ridge_eq (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (ρ : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ) (h0 : μ 0 = ρ) :
    μ = ridgeLaw p u b ρ := by
  funext t
  rw [cauchyPB_ridge_transport i p u b μ hμ t, h0]

/-- The explicit original ridge controls have global weak uniqueness.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem cauchyPB_ridge_unique (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ)
    (hν : cauchyPB d (fun _ => reluRidgeParams i p u b) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  funext t
  rw [cauchyPB_ridge_transport i p u b μ hμ t, cauchyPB_ridge_transport i p u b ν hν t, h0]

/-- Every spherical probability supplies exactly one actual global weak
solution for the ridge controls. Source: arXiv:2411.04551v3,
§1, well-posedness of `eq: cauchy.pb`. -/
theorem exists_unique_cauchyPB_ridge (i : Fin d) (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (fun _ => reluRidgeParams i p u b) μ := by
  refine ⟨ridgeLaw p u b ν, ⟨ridgeLaw_zero _ _ _ _, cauchyPB_ridgeLaw i p u b ν⟩, fun μ hμ => ?_⟩
  funext t
  rw [cauchyPB_ridge_transport i p u b μ hμ.2 t, hμ.1]

/-- The actual original probability solution maps compose in time.
Source: arXiv:2411.04551v3, §1, the solution map of `eq: cauchy.pb`. -/
theorem ridgeLaw_add (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) (s t : ℝ) :
    ridgeLaw p u b ν (s + t) = ridgeLaw p u b (ridgeLaw p u b ν s) t := by
  apply Subtype.ext
  change (ridgeLaw p u b ν (s + t) : Measure (SSphere d)) =
    (ridgeLaw p u b (ridgeLaw p u b ν s) t : Measure (SSphere d))
  simp only [coe_ridgeLaw]
  rw [Measure.map_map (measurable_ridgeFlow_initial p u b t) (measurable_ridgeFlow_initial p u b s)]
  congr 1
  funext x
  exact ridgeFlow_add p u b x s t

/-- Time reversal inverts the actual original probability solution map.
Source: arXiv:2411.04551v3, §1, the invertibility of the solution map. -/
theorem ridgeLaw_inverse (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    ridgeLaw p u b (ridgeLaw p u b ν t) (-t) = ν := by
  rw [← ridgeLaw_add, add_neg_cancel, ridgeLaw_zero]

/-- Each actual original ridge solution map is a bijection of probabilities.
Source: arXiv:2411.04551v3, §1, the invertible solution map. -/
theorem bijective_ridgeLaw (p u : EucSpace d) (b t : ℝ) :
    Function.Bijective (fun ν : Perspective.ProbSphere d => ridgeLaw p u b ν t) := by
  have hl : Function.LeftInverse (fun ν => ridgeLaw p u b ν (-t)) (fun ν => ridgeLaw p u b ν t) :=
    fun ν => ridgeLaw_inverse p u b ν t
  refine ⟨hl.injective, fun ν => ⟨ridgeLaw p u b ν (-t), ?_⟩⟩
  simpa only [neg_neg] using ridgeLaw_inverse p u b ν (-t)

/-- A law initially concentrated in the actual inactive set is fixed.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`,
the identity outside the activation region. -/
theorem ridgeLaw_eq_of_inactive_ae (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d)
    (hν : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)), b - inner ℝ (x : EucSpace d) u ≤ 0) (t : ℝ) :
    ridgeLaw p u b ν t = ν := by
  have he : (fun x : SSphere d => ridgeFlow p u b x t) =ᵐ[(ν : Measure (SSphere d))] id :=
    hν.mono fun x hx => ridgeFlow_inactive p u b x hx t
  apply Subtype.ext
  change (ridgeLaw p u b ν t : Measure (SSphere d)) = (ν : Measure (SSphere d))
  rw [coe_ridgeLaw]
  simpa only [Measure.map_id] using Measure.map_congr he

/-- Every original weak solution from an inactive law stays fixed.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`,
the identity outside the activation region. -/
theorem cauchyPB_ridge_stationary_of_inactive (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ)
    (h0 : ∀ᵐ x : SSphere d ∂(μ 0 : Measure (SSphere d)), b - inner ℝ (x : EucSpace d) u ≤ 0) (t : ℝ) :
    μ t = μ 0 := by
  rw [cauchyPB_ridge_transport i p u b μ hμ t]
  exact ridgeLaw_eq_of_inactive_ae p u b (μ 0) h0 t

/-- A nonzero actual neuron, an inactive atom and the actual weak law
from it jointly realize the weak, equal-initial-data and inactive-set
hypotheses for exactly the same controls. -/
example : ∃ (p u : EucSpace 2) (ρ : Perspective.ProbSphere 2) (μ ν : ℝ → Perspective.ProbSphere 2),
    p ≠ 0 ∧ cauchyPB 2 (fun _ => reluRidgeParams (0 : Fin 2) p u 0) μ ∧
    cauchyPB 2 (fun _ => reluRidgeParams (0 : Fin 2) p u 0) ν ∧ μ 0 = ν 0 ∧ μ 0 = ρ ∧
    (∀ᵐ x : SSphere 2 ∂(ρ : Measure (SSphere 2)), 0 - inner ℝ (x : EucSpace 2) u ≤ 0) ∧
    ∀ᵐ x : SSphere 2 ∂(μ 0 : Measure (SSphere 2)), 0 - inner ℝ (x : EucSpace 2) u ≤ 0 := by
  let a : SSphere 2 := basePoint 1
  let p : EucSpace 2 := EuclideanSpace.single 1 1
  let u : EucSpace 2 := a
  let ρ := Perspective.diracProb 2 a
  let μ := ridgeLaw p u 0 ρ
  have hρ : ∀ᵐ x : SSphere 2 ∂(ρ : Measure (SSphere 2)), 0 - inner ℝ (x : EucSpace 2) u ≤ 0 := by
    change ∀ᵐ x : SSphere 2 ∂Measure.dirac a, 0 - inner ℝ (x : EucSpace 2) u ≤ 0
    rw [MeasureTheory.ae_dirac_eq, Filter.eventually_pure]
    have ha : ‖(a : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp a.2
    change 0 - inner ℝ (a : EucSpace 2) (a : EucSpace 2) ≤ 0
    rw [real_inner_self_eq_norm_sq, ha]
    norm_num
  refine ⟨p, u, ρ, μ, μ, ?_, cauchyPB_ridgeLaw _ _ _ _ _, cauchyPB_ridgeLaw _ _ _ _ _, rfl, ridgeLaw_zero _ _ _ _, hρ, ?_⟩
  · intro hp
    have hh := congrArg (fun v : EucSpace 2 => v 1) hp
    norm_num [p] at hh
  · dsimp only [μ]
    rw [ridgeLaw_zero]
    exact hρ

end Transformer.Interpolation
