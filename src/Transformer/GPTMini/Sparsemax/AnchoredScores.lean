import Transformer.GPTMini.Sparsemax.BoundedCoordinates

/-!
# Persistent active anchors with exact sparsemax zeros

Derived construction from arXiv:1602.02068v2, §2.2, Proposition 1.
The first `A` visible slots are anchors with independent bounded scores.
Every other score is strictly below the anchors' lower bound. When
`2 * A * cap < 1`, all anchors remain positive under the actual causal
variational projection, for every finite assignment of their parameters.
Other visible slots may have exactly zero weight.

This changes the score architecture of `Attention` at `73f8a0b`:
anchor coordinates are independent rather than query/key inner products,
and ordinary slots use negative exponential coordinates. The causal
mask and simplex projection are unchanged. All anchors must precede
the current query; this is an explicit prefix requirement.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Independent bounded anchor scores followed by lower ordinary scores.
Derived architecture for arXiv:1602.02068v2, §2.2; the `A` anchor slots
are prepended to `N` ordinary slots before applying the usual causal mask. -/
def anchoredScores {A N : ℕ} (cap : ℝ) (parameters : Fin A → ℝ)
    (ordinary : Fin N → ℝ) : Fin (A + N) → ℝ :=
  Fin.addCases (fun a => boundedCoordinate cap (parameters a))
    (fun n => -cap - Real.exp (ordinary n))

/-- Every anchor score stays in the cap's open interval.
Source context: the derived prefix construction for §2.2 of
arXiv:1602.02068v2; no bound on learned parameters is assumed. -/
theorem anchoredScores_anchor_bounds {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (hc : 0 < cap) (a : Fin A) :
    -cap < anchoredScores cap parameters ordinary (Fin.castAdd N a) ∧
      anchoredScores cap parameters ordinary (Fin.castAdd N a) < cap := by
  simpa only [anchoredScores, Fin.addCases_left] using
    boundedCoordinate_bounds cap (parameters a) hc

/-- The anchor-bound hypotheses have a finite sparse-row instance.
Source context: arXiv:1602.02068v2, §2.2, derived prefix scores. -/
example : -(1 / 8 : ℝ) <
    anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 0 ∧
    anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 0 < 1 / 8 := by
  exact anchoredScores_anchor_bounds _ _ _ (by norm_num) 0

/-- Ordinary slots remain strictly below the anchors' lower bound.
Source context: the derived prefix construction for §2.2 of
arXiv:1602.02068v2; their finite parameters are unrestricted. -/
theorem anchoredScores_ordinary_lt {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (n : Fin N) :
    anchoredScores cap parameters ordinary (Fin.natAdd A n) < -cap := by
  simp only [anchoredScores, Fin.addCases_right]
  linarith [Real.exp_pos (ordinary n)]

/-- Every visible anchor stays active at every finite parameter assignment.
Source: a derived restriction from arXiv:1602.02068v2, §2.2,
`sparsemax_closedform`. If an anchor were inactive, all ordinary slots
would be inactive and the anchors' total mass would be less than one. -/
theorem anchoredScores_anchor_positive {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (i : Fin (A + N))
    (hc : 0 < cap) (hcap : 2 * (A : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin A, Fin.castAdd N a ≤ i) (a : Fin A) :
    0 < sparseWeights (anchoredScores cap parameters ordinary) i (Fin.castAdd N a) := by
  classical
  let scores := anchoredScores cap parameters ordinary
  obtain ⟨τ, hrow⟩ := sparseWeights_exists_threshold scores i
  by_contra hn
  have hp : sparseWeights scores i (Fin.castAdd N a) ≤ 0 := le_of_not_gt hn
  rw [hrow] at hp
  simp only [thresholdWeights, hvisible a, ite_true] at hp
  have haτ : scores (Fin.castAdd N a) ≤ τ := by
    have := le_trans (le_max_left (scores (Fin.castAdd N a) - τ) 0) hp
    linarith
  have hτ : -cap < τ :=
    lt_of_lt_of_le (anchoredScores_anchor_bounds cap parameters ordinary hc a).1 haτ
  have hanchor (b : Fin A) :
      thresholdWeights scores i τ (Fin.castAdd N b) ≤ 2 * cap := by
    simp only [thresholdWeights, hvisible b, ite_true]
    apply max_le
    · linarith [(anchoredScores_anchor_bounds cap parameters ordinary hc b).2]
    · positivity
  have hother (n : Fin N) : thresholdWeights scores i τ (Fin.natAdd A n) = 0 := by
    have hs : scores (Fin.natAdd A n) - τ ≤ 0 := by
      linarith [anchoredScores_ordinary_lt cap parameters ordinary n]
    simp [thresholdWeights, max_eq_right hs]
  have hsum := (sparseWeights_spec scores i).1.2.1
  rw [hrow, Fin.sum_univ_add] at hsum
  have hzero : (∑ n : Fin N, thresholdWeights scores i τ (Fin.natAdd A n)) = 0 :=
    Finset.sum_eq_zero fun n _ => hother n
  rw [hzero, add_zero] at hsum
  have hbound := Finset.sum_le_sum (fun b (_ : b ∈ Finset.univ) => hanchor b)
  rw [hsum] at hbound
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hbound
  nlinarith

/-- Every premise of persistent activity holds with two visible anchors
and a third ordinary slot. Source context: §2.2's derived prefix rule. -/
example : ∀ a : Fin 2,
    0 < sparseWeights
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2
      (Fin.castAdd 1 a) := by
  apply anchoredScores_anchor_positive _ _ _ 2 (by norm_num) (by norm_num)
  intro a
  exact Fin.le_last (Fin.castAdd 1 a)

/-- A sufficiently low ordinary score stays exactly inactive even while
anchor parameters change. Source: §2.2 of arXiv:1602.02068v2,
the one-unit support window applied to the derived prefix construction. -/
theorem anchoredScores_ordinary_zero_of_margin {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (i : Fin (A + N))
    (hc : 0 < cap) (a : Fin A) (ha : Fin.castAdd N a ≤ i) (n : Fin N)
    (hn : 1 ≤ Real.exp (ordinary n)) :
    sparseWeights (anchoredScores cap parameters ordinary) i (Fin.natAdd A n) = 0 := by
  apply sparseWeights_zero_of_score_gap _ i _ (Fin.castAdd N a) ha
  have hb := (anchoredScores_anchor_bounds cap parameters ordinary hc a).1
  simp only [anchoredScores, Fin.addCases_right] at *
  linarith

/-- Margin and visibility premises hold at finite parameters with an
ordinary visible slot. Source context: §2.2's derived anchor scores. -/
example : sparseWeights
    (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 2 = 0 :=
  anchoredScores_ordinary_zero_of_margin _ _ _ 2 (by norm_num) 0 (by decide) 0
    (by norm_num)

/-- The new architecture retains a visible exact zero.
Source context: the derived scores for arXiv:1602.02068v2, §2.2;
the actual variational projection is `(1/2, 1/2, 0)`, not a dense branch. -/
theorem anchoredScores_sparse_example :
    sparseWeights
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 =
      (fun n : Fin 3 => if n = 2 then 0 else 1 / 2) := by
  let scores := anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)
  have hclosed : thresholdWeights scores 2 (-(1 / 2)) =
      (fun n : Fin 3 => if n = 2 then 0 else 1 / 2) := by
    funext n
    fin_cases n <;> norm_num [thresholdWeights, scores, anchoredScores, Fin.addCases,
      boundedCoordinate]
  have hsum : ∑ n : Fin 3, thresholdWeights scores 2 (-(1 / 2)) n = 1 := by
    rw [hclosed]
    norm_num [Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights scores 2 (-(1 / 2)) hsum]
  exact hclosed

end Transformer.GPTMini.Sparsemax
