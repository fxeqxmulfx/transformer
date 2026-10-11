/-
# Attention's forward pass and Frank-Wolfe — the first exit needs bad steps

Before the first exit from a group's hull neighborhood, every particle is
near its assigned vertex. Every bad step preceding that exit is therefore
among the stopped events whose conditional probability is controlled.
The pathwise drift estimate forces at least `ceil(ε/(γ D))` such steps.
Source: arXiv:2508.09628v1, §5, proof of `lem: metastab.1`, the paragraph
before `eq: union.bound` and the binomial estimate `eq: binomial.2`.
-/

import Transformer.FrankWolfe.Section5_BadEvents
import Transformer.FrankWolfe.Section5_RareEvents

open scoped ENNReal BigOperators Pointwise
open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n κ : ℕ}

/-- No particle leaves its group's initial hull neighborhood through time `N`. -/
def survivalEvent (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (ε : ℝ) (N : ℕ) :
    Set (ℕ → Idx n → EucSpace d) :=
  {x | ∀ s ≤ N, ∀ i, x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε}

/-- The finite-horizon survival event is measurable. Source:
arXiv:2508.09628v1, §5, the event defining `T₂` in `lem: metastab.1`. -/
theorem measurableSet_survivalEvent (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ)
    (ε : ℝ) (N : ℕ) : MeasurableSet (survivalEvent X₀ σ ε N) := by
  have he : survivalEvent X₀ σ ε N = ⋂ s, ⋂ (_ : s ≤ N), ⋂ i,
      {x | x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε} := by
    ext x
    simp [survivalEvent]
  rw [he]
  exact MeasurableSet.iInter fun s => MeasurableSet.iInter fun _ =>
    MeasurableSet.iInter fun i => (Metric.isOpen_ball.add_left.measurableSet).preimage
      ((measurable_pi_apply i).comp (measurable_pi_apply s))

/-- Survival through the integer floor implies survival at every step before
the real horizon. Source: arXiv:2508.09628v1, §5, `eq: metastability.bound`. -/
theorem survivalEvent_subset_real_horizon (X₀ : Idx n → EucSpace d)
    (σ : Idx n → Idx κ) (ε t : ℝ) : survivalEvent X₀ σ ε ⌊t⌋₊ ⊆
      {x | ∀ s : ℕ, (s : ℝ) < t → ∀ i,
        x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε} := by
  intro x hx s hs i
  exact hx s (Nat.le_floor hs.le) i

/-- The count event equals a finite union of intersections at prescribed
times. Source: arXiv:2508.09628v1, §5, auxiliary to `eq: binomial.2`. -/
theorem count_rare_steps_eq_biUnion (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (N M : ℕ) :
    {x | M ≤ (rareStepTimes E N x).card} =
      ⋃ B ∈ (Finset.range N).powersetCard M, ⋂ k ∈ B, E k := by
  classical
  ext x
  constructor
  · intro hx
    obtain ⟨B, hB, hcard⟩ := Finset.exists_subset_card_eq hx
    refine Set.mem_iUnion.mpr ⟨B, Set.mem_iUnion.mpr ⟨?_, ?_⟩⟩
    · exact Finset.mem_powersetCard.mpr ⟨hB.trans (Finset.filter_subset _ _), hcard⟩
    · simp only [Set.mem_iInter]
      exact fun k hk => (Finset.mem_filter.mp (hB hk)).2
  · intro hx
    obtain ⟨B, hxB⟩ := Set.mem_iUnion.mp hx
    obtain ⟨hB, hxB⟩ := Set.mem_iUnion.mp hxB
    obtain ⟨hrange, hcard⟩ := Finset.mem_powersetCard.mp hB
    have hsub : B ⊆ rareStepTimes E N x := by
      intro k hk
      exact Finset.mem_filter.mpr ⟨hrange hk, Set.mem_iInter.mp (Set.mem_iInter.mp hxB k) hk⟩
    simpa only [hcard, Set.mem_ofPred_eq] using Finset.card_le_card hsub

/-- Measurable step events have measurable count tails. Source:
arXiv:2508.09628v1, §5, auxiliary to `eq: binomial.2`. -/
theorem measurableSet_count_rare_steps (E : ℕ → Set (ℕ → Idx n → EucSpace d))
    (hE : ∀ k, MeasurableSet (E k)) (N M : ℕ) :
    MeasurableSet {x | M ≤ (rareStepTimes E N x).card} := by
  rw [count_rare_steps_eq_biUnion]
  exact MeasurableSet.iUnion fun B => MeasurableSet.iUnion fun _ =>
    MeasurableSet.iInter fun k => MeasurableSet.iInter fun _ => hE k

example : ∀ k : ℕ, MeasurableSet ((fun _ : ℕ => ∅) k : Set (ℕ → Idx 1 → EucSpace 1)) :=
  fun _ => MeasurableSet.empty

/-- A path either survives through `N` or accumulates enough stopped bad
steps to account for its first exit. Source: arXiv:2508.09628v1, §5, proof
of `lem: metastab.1`; unlike the printed proof, only steps before the first
exit are charged by the near-vertex probability estimate. -/
theorem survival_or_count_stoppedBadSteps {K : Set (EucSpace d)} (hKv : Convex ℝ K)
    {γ D ρ ε : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1) (hδ : 0 < γ * D)
    (hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ D)
    (X₀ : Idx n → EucSpace d) (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ)
    (hX₀ : ∀ i, X₀ i ∈ K) (hρX : ∀ i, X₀ i ∈ Metric.ball (v (σ i)) ρ)
    (x : ℕ → Idx n → EucSpace d) (h0 : x 0 = X₀) (hK : ∀ s i, x s i ∈ K)
    (hstep : ∀ s i, ∃ j, x (s + 1) i = (1 - γ) • x s i + γ • x s j) (N : ℕ) :
    x ∈ survivalEvent X₀ σ ε N ∨
      ⌈ε / (γ * D)⌉₊ ≤ (rareStepTimes (stoppedBadStep v σ γ (ρ + ε)) N x).card := by
  classical
  by_cases hx : x ∈ survivalEvent X₀ σ ε N
  · exact Or.inl hx
  · right
    have hex : ∃ s ≤ N, ∃ i,
        x s i ∉ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε := by
      simpa only [survivalEvent, Set.mem_ofPred_eq, not_forall, exists_prop] using hx
    obtain ⟨s, hsN, hs⟩ := hex
    have hexit : ∃ s, ∃ i,
        x s i ∉ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε := ⟨s, hs⟩
    let S := Nat.find hexit
    have hSN : S ≤ N := (Nat.find_min' hexit hs).trans hsN
    obtain ⟨i, hi⟩ := Nat.find_spec hexit
    have hbefore : ∀ k < S, ∀ j,
        x k j ∈ groupHull X₀ σ (σ j) + Metric.ball (0 : EucSpace d) ε := by
      intro k hk j
      by_contra h
      exact Nat.find_min hexit hk ⟨j, h⟩
    have hε : ε ≤ γ * D * (badTimes σ γ S x).card :=
      le_of_mem_add_closedBall_of_notMem
        (mem_groupHull_add_closedBall hKv hγ0 hγ1 hD X₀ σ hX₀ x h0 hK hstep S i) hi
    have hcount : ⌈ε / (γ * D)⌉₊ ≤ (badTimes σ γ S x).card :=
      Nat.ceil_le.mpr ((div_le_iff₀ hδ).mpr (by simpa only [mul_comm] using hε))
    refine hcount.trans (Finset.card_le_card ?_)
    intro k hk
    obtain ⟨hkS, hbad⟩ := mem_badTimes.mp hk
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_range.mpr (hkS.trans_le hSN), ?_⟩
    exact ⟨fun j => norm_sub_le_of_mem_groupHull_add_ball hρX (hbefore k hkS j), hbad⟩

/-- All path hypotheses hold for a stationary particle at the origin in a
singleton convex set, with a positive diameter bound and positive radii. -/
example :
    Convex ℝ ({0} : Set (EucSpace 1)) ∧ 0 ≤ (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) ≤ 1 ∧
      0 < (1 / 2 : ℝ) * 1 ∧
      (∀ y ∈ ({0} : Set (EucSpace 1)), ∀ y' ∈ ({0} : Set (EucSpace 1)), ‖y - y'‖ ≤ 1) ∧
      (∀ i : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) i ∈ ({0} : Set (EucSpace 1))) ∧
      (∀ i : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) i ∈
        Metric.ball ((fun _ : Idx 1 => (0 : EucSpace 1)) i) (1 / 8)) ∧
      (fun _ : Idx 1 => (0 : EucSpace 1)) = (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
      (∀ s : ℕ, ∀ i : Idx 1, (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) s i ∈
        ({0} : Set (EucSpace 1))) ∧
      (∀ s : ℕ, ∀ i : Idx 1, ∃ j : Idx 1,
        (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) (s + 1) i =
          (1 - (1 / 2 : ℝ)) • (0 : EucSpace 1) + (1 / 2 : ℝ) •
            (fun _ : Idx 1 => (0 : EucSpace 1)) j) := by
  refine ⟨convex_singleton 0, by norm_num, by norm_num, by norm_num, ?_,
    by simp, fun i => Metric.mem_ball_self (by norm_num), rfl, by simp, ?_⟩
  · rintro y hy y' hy'
    simp only [Set.mem_singleton_iff] at hy hy'
    simp [hy, hy']
  · intro s i
    exact ⟨i, by simp⟩

end Transformer.FrankWolfe
