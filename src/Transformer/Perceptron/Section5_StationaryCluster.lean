module

public import Transformer.Perceptron.Section5_AtomicForce
public import Transformer.Perceptron.Section5_PotentialForce
public import Transformer.Perceptron.Section5_ClusterLifts
public import Transformer.Perceptron.Section5_AngularFirstVariation
public import Transformer.Perceptron.Section5_QuadraticClusterWitness
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Bounding cluster mass by two actual stationary force equations

ArXiv:2601.21366v2, §5.4, the proof of `thm: bound` and
`eq: conclusion`. The source assumes SOPD and uses a Hessian
direction with zero weighted mean. The force comparison here
proves the same printed mass bound for every stationary measure
of the stated atomic form, under the same activation assumptions.

The actual interaction force has derivative at most
`exp β (3/5-(11/10)M)` throughout the short cluster interval.
At its two distinct stationary support endpoints, its increment
is the negative of the potential-force increment. The latter is
bounded by the source's `C_ϑ` times the endpoint separation.
The positive separation can therefore be cancelled.

This gives `M ≤ 6/11+(10/11)C_ϑ exp(-β)`, which implies the
printed `0.5742+(C_ϑ+1)exp(-β)` from `β=4` on. All derivatives
are of the actual first variation and actual interaction force.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **The actual stationary endpoint equations bound the cluster mass.**
Source: arXiv:2601.21366v2, §5.4, `eq:dist.cond` and
`eq: conclusion`. This interval form takes the two genuine support
endpoints explicitly; no second-order condition is needed. -/
theorem cluster_mass_bound_on_interval {N : ℕ} {m θ : Idx N → ℝ}
    {μ : Perspective.ProbSphere 2} (hatom : IsAtomicOnCircle N m θ μ)
    (β : ℝ) (hβ : 4 ≤ β) (φ σ : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : LipschitzWith 1 σ) (hzero : σ 0 = 0)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hμ : IsStationary β σ ω a μ)
    (S : Finset (Idx N)) (α : Idx N → ℝ)
    (hrep : ∀ i, circlePoint (α i) = circlePoint (θ i)) (L U : ℝ) (hLU : L < U)
    (hwidth : U - L ≤ 1 / (2 * Real.sqrt β))
    (hcluster : ∀ i ∈ S, α i ∈ Set.Icc L U)
    (hL : circlePoint L ∈ (μ : Measure (SSphere 2)).support)
    (hU : circlePoint U ∈ (μ : Measure (SSphere 2)).support) :
    (∑ i ∈ S, m i) ≤ 0.5742 + (2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1) * Real.exp (-β) := by
  let I := fun t => ∫ z, circleInteractionForce β t z ∂(μ : Measure (SSphere 2))
  let C := 2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2
  let M := ∑ i ∈ S, m i
  have hβ0 : 0 < β := by linarith
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hIdiff : Differentiable ℝ I := fun r =>
    (hasDerivAt_integral_circleInteractionForce β hβ0 μ r).differentiableAt
  have hupper (r : ℝ) (hr : r ∈ Set.Icc L U) :
      deriv I r ≤ Real.exp β * ((3 / 5 : ℝ) - (11 / 10 : ℝ) * M) :=
    deriv_atomic_interaction_force_le hatom β hβ S α hrep L U hwidth hcluster r hr
  have hincrement := (convex_Icc L U).image_sub_le_mul_sub_of_deriv_le
    hIdiff.continuous.continuousOn (fun r hr => (hIdiff r).differentiableWithinAt)
    (fun r hr => hupper r (interior_subset hr)) L ⟨le_rfl, hLU.le⟩ U ⟨hLU.le, le_rfl⟩ hLU.le
  have hcritical (r : ℝ) (hr : circlePoint r ∈ (μ : Measure (SSphere 2)).support) :
      I r + angularPotentialForce φ ω a r = 0 := by
    have h := deriv_angularFirstVariation_zero_on_support β hβ0.ne' φ σ hφ ω a μ hμ r hr
    rw [deriv_angularFirstVariation_eq_force β hβ0.ne' φ σ hφ] at h
    exact h
  have hP := abs_angularPotentialForce_sub_le φ σ hφ hσ hzero ω a U L
  rw [abs_of_pos (sub_pos.mpr hLU)] at hP
  have hbalance : -C ≤ Real.exp β * ((3 / 5 : ℝ) - (11 / 10 : ℝ) * M) := by
    apply le_of_mul_le_mul_right ?_ (sub_pos.mpr hLU)
    calc -C * (U - L) ≤ I U - I L := by
          linarith [(abs_le.mp hP).2, hcritical L hL, hcritical U hU]
      _ ≤ _ := hincrement
  have hproduct : M * Real.exp β ≤ (6 / 11 : ℝ) * Real.exp β + (10 / 11 : ℝ) * C := by
    nlinarith [hbalance]
  have hmass : M ≤ (6 / 11 : ℝ) + (10 / 11 : ℝ) * C * Real.exp (-β) := by
    rw [Real.exp_neg]
    apply le_of_mul_le_mul_right ?_ (Real.exp_pos β)
    simpa only [add_mul, mul_assoc, inv_mul_cancel₀ (Real.exp_ne_zero β), mul_one] using hproduct
  change M ≤ 0.5742 + (C + 1) * Real.exp (-β)
  nlinarith [hmass, Real.exp_pos (-β), mul_nonneg hC (Real.exp_pos (-β)).le]

/-- **The printed cluster-mass bound follows already from stationarity.**
Source: arXiv:2601.21366v2, §3.2, `thm: bound`, and §5.4,
`eq: conclusion`. The source assumes SOPD; this strengthens its
first conclusion by using only stationarity. The exact circular
cluster condition, printed constant, and activation assumptions are retained. -/
theorem cluster_mass_bound_of_stationary {N : ℕ} {m θ : Idx N → ℝ}
    {μ : Perspective.ProbSphere 2} (hatom : IsAtomicOnCircle N m θ μ)
    (β : ℝ) (hβ : 4 ≤ β) (φ σ : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : LipschitzWith 1 σ) (hzero : σ 0 = 0)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hμ : IsStationary β σ ω a μ)
    (S : Finset (Idx N)) (hcard : 2 ≤ S.card)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ,
      |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β)) :
    (∑ i ∈ S, m i) ≤ 0.5742 + (2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1) * Real.exp (-β) := by
  have hβ0 : 0 < β := by linarith
  have hs : 1 ≤ Real.sqrt β := by
    simpa using Real.sqrt_le_sqrt (show (1 : ℝ) ≤ β by linarith)
  have hscale : 1 / (2 * Real.sqrt β) ≤ (1 / 2 : ℝ) := by
    apply (div_le_iff₀ (show 0 < 2 * Real.sqrt β by positivity)).mpr
    linarith
  obtain ⟨k, lo, hlo, hi, hhi, hLU, hwidth, hcluster⟩ :=
    exists_short_cluster_lifts S θ hatom.angle_mem hatom.angle_inj
      (1 / (2 * Real.sqrt β)) hscale hcard hpair
  let α := fun i => θ i + 2 * π * (k i : ℝ)
  have hrep (i : Idx N) : circlePoint (α i) = circlePoint (θ i) :=
    circlePoint_int_mul_period (θ i) (k i)
  apply cluster_mass_bound_on_interval hatom β hβ φ σ hφ hσ hzero ω a hμ S α hrep
    (α lo) (α hi) hLU hwidth hcluster
  · rw [hrep lo]
    exact circlePoint_mem_support_of_isAtomicOnCircle hatom lo
  · rw [hrep hi]
    exact circlePoint_mem_support_of_isAtomicOnCircle hatom hi

/-- Every input of both mass-bound forms is jointly realized by the
actual quadratic two-atom cluster at `β=4`. The full circular-distance
condition and two distinct supported interval endpoints are included;
the witness does not merely omit the stationarity assumption.
Source: arXiv:2601.21366v2, §3.2, `thm: bound`, and §5.4, `eq:dist.cond`. -/
example : (4 : ℝ) ≤ 4 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id r) r) ∧
    LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
    IsAtomicOnCircle 2 (fun _ => 1 / 2) quadraticClusterAngles quadraticClusterProb ∧
    IsStationary 4 id quadraticClusterWeights quadraticClusterNeuron quadraticClusterProb ∧
    2 ≤ (Finset.univ : Finset (Idx 2)).card ∧
    (∀ i ∈ (Finset.univ : Finset (Idx 2)), ∀ j ∈ Finset.univ, ∃ k : ℤ,
      |quadraticClusterAngles i - quadraticClusterAngles j + 2 * π * (k : ℝ)| ≤
        1 / (2 * Real.sqrt 4)) ∧
    ∃ (α : Idx 2 → ℝ) (L U : ℝ),
      (∀ i, circlePoint (α i) = circlePoint (quadraticClusterAngles i)) ∧
      L < U ∧ U - L ≤ 1 / (2 * Real.sqrt 4) ∧
      (∀ i ∈ (Finset.univ : Finset (Idx 2)), α i ∈ Set.Icc L U) ∧
      circlePoint L ∈ (quadraticClusterProb : Measure (SSphere 2)).support ∧
      circlePoint U ∈ (quadraticClusterProb : Measure (SSphere 2)).support := by
  obtain ⟨hβ, hφ, hσ, hzero, hatom, hμ, hcard, hpair⟩ := quadraticCluster_stationary_witness
  refine ⟨hβ, hφ, hσ, hzero, hatom, hμ, hcard, hpair, ?_⟩
  let α := fun i : Idx 2 => if i = 0 then (1 / 8 : ℝ) else -(1 / 8 : ℝ)
  obtain ⟨hrep, hLU, hwidth, hbounds⟩ := quadraticCluster_interval
  refine ⟨α, -(1 / 8 : ℝ), (1 / 8 : ℝ), hrep, hLU, hwidth, fun i hi => hbounds i, ?_, ?_⟩
  · change circlePoint (α 1) ∈ (quadraticClusterProb : Measure (SSphere 2)).support
    rw [hrep 1]
    exact circlePoint_mem_support_of_isAtomicOnCircle hatom 1
  · change circlePoint (α 0) ∈ (quadraticClusterProb : Measure (SSphere 2)).support
    rw [hrep 0]
    exact circlePoint_mem_support_of_isAtomicOnCircle hatom 0

end Transformer.Perceptron
