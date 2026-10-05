import Transformer.GPTMini.Sparsemax.SupportWindow

/-!
# Full-support sparsemax and relative uniformity

arXiv:1602.02068v2, §2.2, Proposition 1 and equation
`threshold_closedform`, specialized to a full causal support. The exact
relative-error scale is epsilon divided by the number of visible slots.
This is a row theorem, not a claim that counting tasks fail in training.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Visible slots of one causal row. Source: `CausalMHA.forward` at
commit `73f8a0b`, the causal mask applied to Proposition 1 of §2.2. -/
def visiblePositions {T : ℕ} (i : Fin T) : Finset (Fin T) :=
  Finset.univ.filter fun j => j ≤ i

/-- Mean of visible scores, including every slot rather than only the
positive support. Source: arXiv:1602.02068v2, §2.2, `threshold_closedform`,
when the support is full. -/
def visibleScoreMean {T : ℕ} (scores : Fin T → ℝ) (i : Fin T) : ℝ :=
  (∑ j ∈ visiblePositions i, scores j) / (visiblePositions i).card

/-- A causal row has a positive number of visible slots. Source:
`CausalMHA.forward`'s unmasked diagonal, commit `73f8a0b`. -/
theorem visiblePositions_card_pos {T : ℕ} (i : Fin T) :
    0 < (visiblePositions i).card := by
  apply Finset.card_pos.mpr
  exact ⟨i, by simp [visiblePositions]⟩

/-- With full visible support, actual sparsemax weights are `1/N` plus
the score's deviation from the visible mean. Source: arXiv:1602.02068v2,
§2.2, Proposition 1 and `threshold_closedform`; `N` is the causal prefix
cardinality, not the padded context length. -/
theorem sparseWeights_full_support {T : ℕ} (scores : Fin T → ℝ) (i : Fin T)
    (hfull : ∀ j, j ≤ i → 0 < sparseWeights scores i j) :
    ∀ j, j ≤ i → sparseWeights scores i j =
      1 / ((visiblePositions i).card : ℝ) + scores j - visibleScoreMean scores i := by
  classical
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold scores i
  have he (j : Fin T) (hj : j ≤ i) : sparseWeights scores i j = scores j - τ := by
    have hp : 0 < max (scores j - τ) 0 := by
      simpa only [hτ, thresholdWeights, hj, ite_true] using hfull j hj
    have hs : 0 ≤ scores j - τ := by
      by_contra hn
      have hz : max (scores j - τ) 0 = 0 := max_eq_right (le_of_not_ge hn)
      linarith
    simp only [hτ, thresholdWeights, hj, ite_true, max_eq_left hs]
  have hsum : (∑ j ∈ visiblePositions i, sparseWeights scores i j) = 1 := by
    unfold visiblePositions
    rw [Finset.sum_filter]
    have hmask : (∑ j : Fin T, if j ≤ i then sparseWeights scores i j else 0) =
        ∑ j, sparseWeights scores i j := by
      apply Finset.sum_congr rfl
      intro j _
      by_cases hj : j ≤ i
      · simp only [hj, ite_true]
      · simp only [hj, ite_false, (sparseWeights_spec scores i).1.2.2 j hj]
    rw [hmask]
    exact (sparseWeights_spec scores i).1.2.1
  have hτsum : (∑ j ∈ visiblePositions i, (scores j - τ)) = 1 := by
    rw [← hsum]
    apply Finset.sum_congr rfl
    intro j hj
    exact (he j (Finset.mem_filter.mp hj).2).symm
  rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul] at hτsum
  have hn : (0 : ℝ) < (visiblePositions i).card :=
    Nat.cast_pos.mpr (visiblePositions_card_pos i)
  have ht : τ = ((∑ j ∈ visiblePositions i, scores j) - 1) /
      ((visiblePositions i).card : ℝ) := by
    apply (eq_div_iff (ne_of_gt hn)).mpr
    linarith
  intro j hj
  rw [he j hj, ht]
  unfold visibleScoreMean
  field_simp
  ring

/-- A genuinely full two-position support satisfies the formula's premise.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : ∀ j : Fin 2, j ≤ 1 → 0 < sparseWeights (fun _ => 0) 1 j := by
  intro j _
  rw [sparseWeights_two_equal]
  norm_num

/-- Relative uniformity within epsilon is equivalent to visible scores
within `epsilon/N` of their mean. Source: arXiv:1602.02068v2, §2.2,
`threshold_closedform`. The source gives the closed form; this exact
relative-error equivalence is its consequence, with full support explicit. -/
theorem sparseWeights_relative_uniform_iff {T : ℕ} (scores : Fin T → ℝ)
    (i : Fin T) (ε : ℝ) (hfull : ∀ j, j ≤ i → 0 < sparseWeights scores i j) :
    (∀ j, j ≤ i → |sparseWeights scores i j -
      1 / ((visiblePositions i).card : ℝ)| ≤ ε / ((visiblePositions i).card : ℝ)) ↔
    (∀ j, j ≤ i → |scores j - visibleScoreMean scores i| ≤
      ε / ((visiblePositions i).card : ℝ)) := by
  have he (j : Fin T) (hj : j ≤ i) :
      sparseWeights scores i j - 1 / ((visiblePositions i).card : ℝ) =
        scores j - visibleScoreMean scores i := by
    rw [sparseWeights_full_support scores i hfull j hj]
    ring
  constructor
  · intro h j hj
    simpa only [he j hj] using h j hj
  · intro h j hj
    simpa only [he j hj] using h j hj

/-- Full-support premises for the relative-error equivalence are inhabited.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : ∀ j : Fin 2, j ≤ 1 → 0 < sparseWeights (fun _ => 0) 1 j := by
  intro j _
  rw [sparseWeights_two_equal]
  norm_num

/-- Full-support attention equals a uniform average plus the weighted
score deviations. Source: arXiv:1602.02068v2, §2.2, Proposition 1, applied
to attention values as in §4.3. It states the actual row's output. -/
theorem sparseWeights_full_support_average {T d : ℕ} (scores : Fin T → ℝ)
    (i : Fin T) (values : Fin T → EucSpace d)
    (hfull : ∀ j, j ≤ i → 0 < sparseWeights scores i j) :
    (∑ j, sparseWeights scores i j • values j) =
      (1 / ((visiblePositions i).card : ℝ)) • (∑ j ∈ visiblePositions i, values j) +
        ∑ j ∈ visiblePositions i, (scores j - visibleScoreMean scores i) • values j := by
  classical
  have hmask : (∑ j, sparseWeights scores i j • values j) =
      ∑ j ∈ visiblePositions i, sparseWeights scores i j • values j := by
    unfold visiblePositions
    rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j ≤ i
    · simp only [hj, ite_true]
    · simp only [hj, ite_false, (sparseWeights_spec scores i).1.2.2 j hj, zero_smul]
  rw [hmask, Finset.smul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  rw [sparseWeights_full_support scores i hfull j (Finset.mem_filter.mp hj).2]
  rw [show 1 / ((visiblePositions i).card : ℝ) + scores j - visibleScoreMean scores i =
    1 / ((visiblePositions i).card : ℝ) + (scores j - visibleScoreMean scores i) by ring]
  exact add_smul _ _ _

/-- Nonzero value streams are compatible with full sparsemax support.
Source context: arXiv:1602.02068v2, §4.3, attention from simplex weights. -/
example : ∀ j : Fin 2, j ≤ 1 → 0 < sparseWeights (fun _ => 0) 1 j := by
  intro j _
  rw [sparseWeights_two_equal]
  norm_num

end Transformer.GPTMini.Sparsemax
