module

public import Transformer.Interpolation.Section3_CollinearRidgePhase
public import Transformer.Interpolation.Section3_FiniteRefinement
public import Transformer.Interpolation.Section3_GatedCenterLaw

/-!
# One actual ridge phase strictly refines barycenter classes

ArXiv:2411.04551v3, §3.2, the perturbation stage of `prop: separation`.
An injective finite family of original quadrant probabilities may have
repeated normalized barycenters. Choose two laws in one such class.
The actual collinear ridge separates their directions at small time.
A smaller common time keeps every original support in the quadrant
and preserves every previously distinct direction.

The ridge law is bijective, so the family of actual probabilities remains
injective. Consequently one actual phase strictly increases the number
of barycenter classes. Exact output scaling puts the same progress at
any prescribed positive horizon, with global original weak uniqueness.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Scaling the actual single-neuron output scales its exact law clock.
Source: arXiv:2411.04551v3, §3.2, time rescaling of perceptron phases. -/
theorem ridgeLaw_smul_output (κ : ℝ) (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    ridgeLaw (κ • p) u b ν t = ridgeLaw p u b ν (κ * t) := by
  rw [← twoRidgeLaw_zero_second (κ • p) u b 0 0 ν t, twoRidgeLaw_scaled_ridge]

/-- One original ridge phase preserves the finite geometric data and
strictly increases the number of barycenter directions. Source:
arXiv:2411.04551v3, §3.2, the perturbation stage of `prop: separation`. -/
theorem exists_ridge_class_refinement {ι : Type*} [Fintype ι]
    (i : Fin d) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective ν)
    (hclasses : ¬Function.Injective (fun k => barycenterDirection (ν k))) :
    ∃ (p u : EucSpace d) (b t : ℝ), 0 < t ∧
      (∀ k, (ridgeLaw p u b (ν k) t : Measure (SSphere d)).support ⊆
        positiveQuadrant d) ∧
      Function.Injective (fun k => ridgeLaw p u b (ν k) t) ∧
      (∀ k l, barycenterDirection (ν k) ≠ barycenterDirection (ν l) →
        barycenterDirection (ridgeLaw p u b (ν k) t) ≠
          barycenterDirection (ridgeLaw p u b (ν l) t)) ∧
      (∃ k l, barycenterDirection (ν k) = barycenterDirection (ν l) ∧
        barycenterDirection (ridgeLaw p u b (ν k) t) ≠
          barycenterDirection (ridgeLaw p u b (ν l) t)) ∧
      Nat.card (Set.range (fun k => barycenterDirection (ν k))) <
        Nat.card (Set.range
          (fun k => barycenterDirection (ridgeLaw p u b (ν k) t))) := by
  classical
  -- An equal-direction pair consists of genuinely distinct original laws.
  obtain ⟨k, l, hmean, hkl⟩ := Function.not_injective_iff.mp hclasses
  have hne : ν k ≠ ν l := fun h => hkl (hinj h)
  obtain ⟨p, u, b, δ, hδ, hsep⟩ :=
    exists_collinear_ridge_direction_separation (ν k) (ν l) hne
    (hν k) (hν l) (barycenter_colinear_of_direction_eq _ _ hmean)
  have hn (k : ι) : barycenter d (ν k) ≠ 0 := by
    intro he
    have hh := barycenter_coordinate_pos_of_positiveQuadrant (ν k) (hν k) i
    rw [he] at hh
    exact (lt_irrefl 0) hh
  -- One common neighborhood preserves all supports and all existing classes.
  have he := (eventually_ridge_family_positiveQuadrant p u b ν hν).and
    (eventually_ridge_family_direction_ne i p u b ν hn)
  obtain ⟨ε, hε, hsmall⟩ := Metric.eventually_nhds_iff.mp he
  let t := min δ ε / 2
  have ht : 0 < t := half_pos (lt_min hδ hε)
  have htδ : t < δ :=
    (half_lt_self (lt_min hδ hε)).trans_le (min_le_left _ _)
  have htε : t < ε :=
    (half_lt_self (lt_min hδ hε)).trans_le (min_le_right _ _)
  have hs := hsmall (y := t)
    (by simpa only [Real.dist_eq, sub_zero, abs_of_pos ht] using htε)
  have hsplit : ∃ k l, barycenterDirection (ν k) = barycenterDirection (ν l) ∧
      barycenterDirection (ridgeLaw p u b (ν k) t) ≠
        barycenterDirection (ridgeLaw p u b (ν l) t) :=
    ⟨k, l, hmean, hsep t ht htδ⟩
  -- The exact ridge bijection retains distinctness of the original probabilities.
  refine ⟨p, u, b, t, ht, hs.1,
    (bijective_ridgeLaw p u b t).1.comp hinj, hs.2, hsplit, ?_⟩
  exact card_range_lt_of_strict_refinement _ _ hs.2 hsplit

/-- One constant actual neuron gives the same strict finite progress
at any positive prescribed time, with global original weak existence
and uniqueness. Source: arXiv:2411.04551v3, §3.2, `prop: separation`. -/
theorem exists_prescribed_time_ridge_class_refinement {ι : Type*} [Fintype ι]
    (i : Fin d) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective ν)
    (hclasses : ¬Function.Injective (fun k => barycenterDirection (ν k)))
    (T : ℝ) (hT : 0 < T) :
    ∃ (p u : EucSpace d) (b : ℝ),
      (∀ ρ : Perspective.ProbSphere d, ∃! μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d (fun _ => reluRidgeParams i p u b) μ) ∧
      (∀ k, (ridgeLaw p u b (ν k) T : Measure (SSphere d)).support ⊆
        positiveQuadrant d) ∧
      Function.Injective (fun k => ridgeLaw p u b (ν k) T) ∧
      (∀ k l, barycenterDirection (ν k) ≠ barycenterDirection (ν l) →
        barycenterDirection (ridgeLaw p u b (ν k) T) ≠
          barycenterDirection (ridgeLaw p u b (ν l) T)) ∧
      (∃ k l, barycenterDirection (ν k) = barycenterDirection (ν l) ∧
        barycenterDirection (ridgeLaw p u b (ν k) T) ≠
          barycenterDirection (ridgeLaw p u b (ν l) T)) ∧
      Nat.card (Set.range (fun k => barycenterDirection (ν k))) <
        Nat.card (Set.range
          (fun k => barycenterDirection (ridgeLaw p u b (ν k) T))) := by
  obtain ⟨p, u, b, t, -, hrest⟩ := exists_ridge_class_refinement i ν hν hinj hclasses
  -- The prescribed physical clock reproduces the already chosen intrinsic clock.
  have he (ρ : Perspective.ProbSphere d) :
      ridgeLaw ((t / T) • p) u b ρ T = ridgeLaw p u b ρ t := by
    rw [ridgeLaw_smul_output, div_mul_cancel₀ _ hT.ne']
  refine ⟨(t / T) • p, u, b, exists_unique_cauchyPB_ridge i _ u b, ?_⟩
  simpa only [he] using hrest

/-- The equal-mean atomic counterexample gives a genuinely injective
family with one repeated barycenter direction. Its original supports
are positive and the prescribed horizon is positive. -/
example : ∃ ν : Fin 2 → Perspective.ProbSphere 3,
    (∀ k, (ν k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
    Function.Injective ν ∧
    ¬Function.Injective (fun k => barycenterDirection (ν k)) ∧ (0 : ℝ) < 1 := by
  obtain ⟨μ, σ, hne, hμ, hσ, hbary, -⟩ := perturbation_equal_barycenter_counterexample
  let ν : Fin 2 → Perspective.ProbSphere 3 := ![μ, σ]
  have hQ (k : Fin 2) : (ν k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 := by
    fin_cases k
    · exact hμ
    · exact hσ
  have hinj : Function.Injective ν := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact (hne he).elim
    · exact (hne he.symm).elim
    · rfl
  have hdir : barycenterDirection (ν 0) = barycenterDirection (ν 1) := by
    change barycenterDirection μ = barycenterDirection σ
    unfold barycenterDirection
    rw [hbary]
  refine ⟨ν, hQ, hinj, ?_, zero_lt_one⟩
  intro he
  have hh : (0 : Fin 2) = 1 := he hdir
  exact (by decide : (0 : Fin 2) ≠ 1) hh

end Transformer.Interpolation
