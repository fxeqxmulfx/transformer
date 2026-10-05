import Transformer.GPTMini.Sparsemax.SupportWindow

/-!
# Score constraints that exclude singleton sparsemax routes

Consequences of arXiv:1602.02068v2, §2.2, Proposition 1, for the actual
causal variational projection. A singleton occurs exactly when its winner
is at least one above every visible competitor. A top-two gap below one
therefore ensures two positive weights without supplying a routing target.

These statements exclude singleton saturation of the score-to-weight map.
They do not assert nonzero derivatives for every outer loss or parameter
map, and they do not require every visible position to have positive weight.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The singleton criterion is necessary as well as sufficient, including
the unit-gap boundary. Source: arXiv:1602.02068v2, §2.2, Proposition 1,
`sparsemax_closedform`, specialized to the causal simplex. -/
theorem sparseWeights_eq_basis_iff_gap {T : ℕ} (scores : Fin T → ℝ)
    (i winner : Fin T) :
    sparseWeights scores i = basis winner ↔ winner ≤ i ∧
      ∀ j, j ≤ i → j ≠ winner → scores j + 1 ≤ scores winner := by
  constructor
  · intro hroute
    have hp : 0 < sparseWeights scores i winner := by
      rw [hroute]
      norm_num [basis]
    have hw := sparseWeights_positive_visible scores i winner hp
    obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold scores i
    have hone : max (scores winner - τ) 0 = 1 := by
      have h := congrFun hroute winner
      simpa [hτ, thresholdWeights, hw, basis] using h
    have hs : 0 ≤ scores winner - τ := by
      by_contra hn
      rw [max_eq_right (le_of_not_ge hn)] at hone
      norm_num at hone
    rw [max_eq_left hs] at hone
    refine ⟨hw, ?_⟩
    intro j hj hn
    have hzero : max (scores j - τ) 0 = 0 := by
      have h := congrFun hroute j
      simpa [hτ, thresholdWeights, hj, basis, hn] using h
    have hle := le_max_left (scores j - τ) 0
    linarith
  · rintro ⟨hw, hgap⟩
    exact sparseWeights_eq_basis_of_gap scores i winner hw hgap

/-- The normalized causal row has a positive coordinate. Source:
arXiv:1602.02068v2, §2.1, the nonempty probability simplex. -/
theorem sparseWeights_exists_positive {T : ℕ} (scores : Fin T → ℝ) (i : Fin T) :
    ∃ j, 0 < sparseWeights scores i j := by
  have hp := (sparseWeights_spec scores i).1
  by_contra hn
  push Not at hn
  have hzero : ∀ j, sparseWeights scores i j = 0 := by
    intro j
    linarith [hp.1 j, hn j]
  have hsum : (∑ j, sparseWeights scores i j) = 0 :=
    Finset.sum_eq_zero (fun j _ => hzero j)
  linarith [hp.2.1]

/-- If every visible candidate has another visible score less than one
below it, the actual row has two distinct positive coordinates. Source:
arXiv:1602.02068v2, §2.2, Proposition 1; this is a derived restriction,
not a claim that every outer training objective receives a gradient. -/
theorem sparseWeights_has_two_positive_of_no_unit_gap {T : ℕ}
    (scores : Fin T → ℝ) (i : Fin T)
    (hnear : ∀ winner, winner ≤ i → ∃ j, j ≤ i ∧ j ≠ winner ∧
      scores winner < scores j + 1) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights scores i j ∧
      0 < sparseWeights scores i k := by
  obtain ⟨winner, hwpos⟩ := sparseWeights_exists_positive scores i
  have hp := (sparseWeights_spec scores i).1
  by_contra hn
  have hzero : ∀ j, j ≠ winner → sparseWeights scores i j = 0 := by
    intro j hj
    by_contra hz
    have hjpos : 0 < sparseWeights scores i j :=
      lt_of_le_of_ne (hp.1 j) (Ne.symm hz)
    exact hn ⟨winner, j, Ne.symm hj, hwpos, hjpos⟩
  have hsum : (∑ j, sparseWeights scores i j) = sparseWeights scores i winner := by
    apply Finset.sum_eq_single winner
    · intro j _ hj
      exact hzero j hj
    · simp
  have hunit : sparseWeights scores i winner = 1 := by
    linarith [hp.2.1]
  have hroute : sparseWeights scores i = basis winner := by
    funext j
    by_cases hj : j = winner
    · subst j
      simpa [basis] using hunit
    · simp [basis, hj, hzero j hj]
  obtain ⟨hw, hgap⟩ := (sparseWeights_eq_basis_iff_gap scores i winner).mp hroute
  obtain ⟨j, hj, hne, hclose⟩ := hnear winner hw
  linarith [hgap j hj hne]

/-- Equal visible scores satisfy the no-unit-gap premises nonvacuously.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : ∀ winner : Fin 2, winner ≤ 1 → ∃ j : Fin 2,
    j ≤ 1 ∧ j ≠ winner ∧ (0 : ℝ) < 0 + 1 := by
  intro winner _
  fin_cases winner
  · exact ⟨1, by decide, by decide, by norm_num⟩
  · exact ⟨0, by decide, by decide, by norm_num⟩

/-- A maximum with a distinct visible competitor within one suffices.
In particular, this holds when the largest and second-largest visible
scores differ by less than one. Source: arXiv:1602.02068v2, §2.2,
Proposition 1, derived using the exact singleton criterion. -/
theorem sparseWeights_has_two_positive_of_top_gap {T : ℕ}
    (scores : Fin T → ℝ) (i winner runner : Fin T)
    (hw : winner ≤ i) (hr : runner ≤ i) (hne : runner ≠ winner)
    (hmax : ∀ j, j ≤ i → scores j ≤ scores winner)
    (hclose : scores winner < scores runner + 1) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights scores i j ∧
      0 < sparseWeights scores i k := by
  apply sparseWeights_has_two_positive_of_no_unit_gap scores i
  intro candidate hc
  by_cases he : candidate = winner
  · subst candidate
    exact ⟨runner, hr, hne, hclose⟩
  · refine ⟨winner, hw, Ne.symm he, ?_⟩
    linarith [hmax candidate hc]

/-- The maximum/competitor premises hold on a genuine two-slot row.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : (0 : Fin 2) ≤ 1 ∧ (1 : Fin 2) ≤ 1 ∧ (1 : Fin 2) ≠ 0 ∧
    (∀ j : Fin 2, j ≤ 1 → (0 : ℝ) ≤ 0) ∧ (0 : ℝ) < 0 + 1 := by
  norm_num

/-- A sub-unit pairwise score range and two visible positions suffice.
Source: arXiv:1602.02068v2, §2.2, Proposition 1; a stronger input
restriction than the top-two gap, suitable for upstream score bounds. -/
theorem sparseWeights_has_two_positive_of_pairwise_gap {T : ℕ}
    (scores : Fin T → ℝ) (i a b : Fin T)
    (ha : a ≤ i) (hb : b ≤ i) (hne : a ≠ b)
    (hgap : ∀ j k, scores j < scores k + 1) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights scores i j ∧
      0 < sparseWeights scores i k := by
  apply sparseWeights_has_two_positive_of_no_unit_gap scores i
  intro winner _
  by_cases he : winner = a
  · subst winner
    exact ⟨b, hb, Ne.symm hne, hgap a b⟩
  · exact ⟨a, ha, Ne.symm he, hgap winner a⟩

/-- Two distinct scores inhabit the upstream pairwise-gap premises.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : (0 : Fin 2) ≤ 1 ∧ (1 : Fin 2) ≤ 1 ∧ (0 : Fin 2) ≠ 1 ∧
    (∀ j k : Fin 2, (j.val : ℝ) / 2 < (k.val : ℝ) / 2 + 1) := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  intro j k
  fin_cases j <;> fin_cases k <;> norm_num

/-- Scores inside a width-4/5 interval, with two identical leaders.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1; a derived
example distinguishing absence of singleton routes from full support. -/
def twoActiveScores : Fin 3 → ℝ :=
  fun j => if j = 2 then -(2 / 5) else 2 / 5

/-- A sub-unit score range is compatible with exact visible zeros.
Source: arXiv:1602.02068v2, §2.2, `sparsemax_closedform`, evaluated
with threshold -1/10 on the actual variational projection. -/
theorem twoActiveScores_projection :
    sparseWeights twoActiveScores 2 = fun j => if j = 2 then 0 else (1 / 2 : ℝ) := by
  have hsum : ∑ j : Fin 3, thresholdWeights twoActiveScores 2 (-(1 / 10)) j = 1 := by
    norm_num [thresholdWeights, twoActiveScores, Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  funext j
  fin_cases j <;> norm_num [thresholdWeights, twoActiveScores]

/-- The same sparse example's raw score differences are at most 4/5.
Source context: arXiv:1602.02068v2, §2.2, derived bounded-score example. -/
theorem twoActiveScores_range :
    ∀ j k, |twoActiveScores j - twoActiveScores k| ≤ (4 / 5 : ℝ) := by
  intro j k
  fin_cases j <;> fin_cases k <;> norm_num [twoActiveScores]

end Transformer.GPTMini.Sparsemax
