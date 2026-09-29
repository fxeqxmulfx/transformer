/-
# Attention's forward pass and Frank-Wolfe — how far a path can drift

`lem: metastab.1`: along a path, each step in which every particle moves to a convex combination
of itself and a particle of its own group keeps all particles inside the `r`-neighbourhood of their
group's hull, and any other step moves a particle by at most `γ 𝖽(𝒦)`.  So after `m` "bad" steps
every particle lies within `γ 𝖽(𝒦) m` of its group's hull — pathwise, with no probability.
-/

import Transformer.FrankWolfe.Section5_Metastability

open scoped BigOperators Pointwise
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- The positions particle `i` may reach in one step without leaving its group: the convex
combinations `(1-γ) x_i + γ x_j` with `j` a particle of the same group. -/
def goodSet (σ : Idx n → Idx κ) (γ : ℝ) (z : Idx n → EucSpace d) (i : Idx n) : Set (EucSpace d) :=
  {y | ∃ j, σ j = σ i ∧ y = (1 - γ) • z i + γ • z j}

/-- The step `s → s + 1` of the path `x` is *bad* if some particle leaves the positions of its own
group. -/
def BadStep (σ : Idx n → Idx κ) (γ : ℝ) (s : ℕ) (x : ℕ → Idx n → EucSpace d) : Prop :=
  ∃ i, x (s + 1) i ∉ goodSet σ γ (x s) i

/-- The times before `s` at which the path takes a bad step. -/
noncomputable def badTimes (σ : Idx n → Idx κ) (γ : ℝ) (s : ℕ) (x : ℕ → Idx n → EucSpace d) :
    Finset ℕ := by
  classical exact (Finset.range s).filter fun s' => BadStep σ γ s' x

theorem mem_badTimes {σ : Idx n → Idx κ} {γ : ℝ} {s s' : ℕ} {x : ℕ → Idx n → EucSpace d} :
    s' ∈ badTimes σ γ s x ↔ s' < s ∧ BadStep σ γ s' x := by
  unfold badTimes
  classical
  simp

theorem badTimes_zero (σ : Idx n → Idx κ) (γ : ℝ) (x : ℕ → Idx n → EucSpace d) :
    badTimes σ γ 0 x = ∅ := by
  ext s'
  simp [mem_badTimes]

theorem badTimes_succ_of_bad {σ : Idx n → Idx κ} {γ : ℝ} {s : ℕ} {x : ℕ → Idx n → EucSpace d}
    (h : BadStep σ γ s x) : (badTimes σ γ (s + 1) x).card = (badTimes σ γ s x).card + 1 := by
  have : badTimes σ γ (s + 1) x = insert s (badTimes σ γ s x) := by
    ext s'
    simp only [mem_badTimes, Finset.mem_insert]
    constructor
    · rintro ⟨h1, h2⟩
      rcases Nat.lt_succ_iff_lt_or_eq.mp h1 with h3 | h3
      · exact Or.inr ⟨h3, h2⟩
      · exact Or.inl h3
    · rintro (rfl | ⟨h1, h2⟩)
      · exact ⟨Nat.lt_succ_self _, h⟩
      · exact ⟨h1.trans (Nat.lt_succ_self _), h2⟩
  rw [this, Finset.card_insert_of_notMem (fun hs => (mem_badTimes.mp hs).1.false)]

theorem badTimes_succ_of_not_bad {σ : Idx n → Idx κ} {γ : ℝ} {s : ℕ}
    {x : ℕ → Idx n → EucSpace d} (h : ¬ BadStep σ γ s x) :
    (badTimes σ γ (s + 1) x).card = (badTimes σ γ s x).card := by
  have : badTimes σ γ (s + 1) x = badTimes σ γ s x := by
    ext s'
    simp only [mem_badTimes]
    constructor
    · rintro ⟨h1, h2⟩
      rcases Nat.lt_succ_iff_lt_or_eq.mp h1 with h3 | h3
      · exact ⟨h3, h2⟩
      · exact absurd (h3 ▸ h2) h
    · rintro ⟨h1, h2⟩
      exact ⟨h1.trans (Nat.lt_succ_self _), h2⟩
  rw [this]

/-- The neighbourhood `conv{x_j^0 : σ j = a} + B̄(0, r)` of a group is convex. -/
theorem convex_groupHull_add_closedBall (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (a : Idx κ)
    (r : ℝ) : Convex ℝ (groupHull X₀ σ a + Metric.closedBall (0 : EucSpace d) r) :=
  (convex_convexHull ℝ _).add (convex_closedBall 0 r)

/-- A convex combination of a point of the `r`-neighbourhood of `H ⊆ K` and a point of `K` lies in
the `(r + γ 𝖽)`-neighbourhood, `𝖽` bounding the distances in `K`. -/
theorem comb_mem_add_closedBall {H K : Set (EucSpace d)} (hHK : H ⊆ K) {γ D r : ℝ} (hγ0 : 0 ≤ γ)
    (hγ1 : γ ≤ 1) (hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ D) {u z : EucSpace d}
    (hu : u ∈ H + Metric.closedBall (0 : EucSpace d) r) (hz : z ∈ K) :
    (1 - γ) • u + γ • z ∈ H + Metric.closedBall (0 : EucSpace d) (r + γ * D) := by
  obtain ⟨h, hh, b, hb, rfl⟩ := Set.mem_add.mp hu
  refine Set.mem_add.mpr ⟨h, hh, (1 - γ) • b + γ • (z - h), ?_, ?_⟩
  · rw [mem_closedBall_zero_iff] at hb ⊢
    calc ‖(1 - γ) • b + γ • (z - h)‖ ≤ ‖(1 - γ) • b‖ + ‖γ • (z - h)‖ := norm_add_le _ _
      _ = (1 - γ) * ‖b‖ + γ * ‖z - h‖ := by
          rw [norm_smul, norm_smul, Real.norm_of_nonneg (sub_nonneg.mpr hγ1),
            Real.norm_of_nonneg hγ0]
      _ ≤ (1 - γ) * r + γ * D :=
          add_le_add (mul_le_mul_of_nonneg_left hb (sub_nonneg.mpr hγ1))
            (mul_le_mul_of_nonneg_left (hD z hz h (hHK hh)) hγ0)
      _ ≤ r + γ * D := by
          have : 0 ≤ r := (norm_nonneg _).trans hb
          nlinarith
  · module

/-- **Pathwise drift.**  Along a path that stays in the convex set `K`, starts at `X₀` and whose
every step moves each particle to a convex combination `(1-γ) x_i + γ x_j` of two particles, after
the bad steps before time `s` every particle lies within `γ 𝖽 · #(bad steps)` of the hull of its
group, `𝖽` bounding the distances in `K`: a step that is not bad keeps the neighbourhood (it is
convex), and a bad one enlarges it by at most `γ 𝖽`. -/
theorem mem_groupHull_add_closedBall {K : Set (EucSpace d)} (hKv : Convex ℝ K) {γ D : ℝ}
    (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1) (hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ D)
    (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (hX₀ : ∀ i, X₀ i ∈ K)
    (x : ℕ → Idx n → EucSpace d) (h0 : x 0 = X₀) (hK : ∀ s i, x s i ∈ K)
    (hstep : ∀ s i, ∃ j, x (s + 1) i = (1 - γ) • x s i + γ • x s j) (s : ℕ) (i : Idx n) :
    x s i ∈ groupHull X₀ σ (σ i) +
      Metric.closedBall (0 : EucSpace d) (γ * D * (badTimes σ γ s x).card) := by
  have hHK : ∀ a, groupHull X₀ σ a ⊆ K := fun a =>
    convexHull_min (by rintro _ ⟨j, -, rfl⟩; exact hX₀ j) hKv
  induction s generalizing i with
  | zero =>
    refine Set.mem_add.mpr ⟨x 0 i, ?_, 0, by simp [badTimes_zero], add_zero _⟩
    rw [h0]
    exact subset_convexHull ℝ _ ⟨i, rfl, rfl⟩
  | succ s ih =>
    by_cases hbad : BadStep σ γ s x
    · rw [badTimes_succ_of_bad hbad]
      obtain ⟨j, hj⟩ := hstep s i
      have h := comb_mem_add_closedBall (hHK (σ i)) hγ0 hγ1 hD (ih i) (hK s j)
      have hr : γ * D * (((badTimes σ γ s x).card + 1 : ℕ) : ℝ) =
          γ * D * (badTimes σ γ s x).card + γ * D := by
        push_cast
        ring
      rw [hj, hr]
      exact h
    · rw [badTimes_succ_of_not_bad hbad]
      have hgood : x (s + 1) i ∈ goodSet σ γ (x s) i := by
        by_contra h
        exact hbad ⟨i, h⟩
      obtain ⟨j, hσj, hj⟩ := hgood
      have hjj := ih j
      rw [hσj] at hjj
      rw [hj]
      exact convex_groupHull_add_closedBall X₀ σ (σ i) _ (ih i) hjj (sub_nonneg.mpr hγ1) hγ0
        (by ring)

/-- Leaving the `ε`-neighbourhood of the hull needs a drift of at least `ε`. -/
theorem le_of_mem_add_closedBall_of_notMem {H : Set (EucSpace d)} {r ε : ℝ} {u : EucSpace d}
    (hu : u ∈ H + Metric.closedBall (0 : EucSpace d) r)
    (hn : u ∉ H + Metric.ball (0 : EucSpace d) ε) : ε ≤ r := by
  by_contra h
  push Not at h
  exact hn (Set.add_subset_add_left (Metric.closedBall_subset_ball h) hu)

/-- The `ε`-neighbourhood of the hull of a group lies within `ρ + ε` of its vertex, when the
group sits in the ball of radius `ρ` around it. -/
theorem norm_sub_le_of_mem_groupHull_add_ball {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ}
    {X₀ : Idx n → EucSpace d} {ρ ε : ℝ} (hρX : ∀ i, X₀ i ∈ Metric.ball (v (σ i)) ρ)
    {a : Idx κ} {u : EucSpace d} (hu : u ∈ groupHull X₀ σ a + Metric.ball (0 : EucSpace d) ε) :
    ‖u - v a‖ ≤ ρ + ε := by
  obtain ⟨h, hh, b, hb, rfl⟩ := Set.mem_add.mp hu
  have hHball : groupHull X₀ σ a ⊆ Metric.ball (v a) ρ :=
    convexHull_min (by
      rintro _ ⟨j, hj, rfl⟩
      simpa [hj] using hρX j) (convex_ball _ _)
  have h1 : ‖h - v a‖ < ρ := by
    have := hHball hh
    rwa [Metric.mem_ball, dist_eq_norm] at this
  have h2 : ‖b‖ < ε := by simpa using hb
  calc ‖h + b - v a‖ = ‖(h - v a) + b‖ := by congr 1; abel
    _ ≤ ‖h - v a‖ + ‖b‖ := norm_add_le _ _
    _ ≤ ρ + ε := by linarith

end FrankWolfe
end Transformer
