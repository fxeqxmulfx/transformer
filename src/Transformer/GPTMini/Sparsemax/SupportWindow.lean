import Transformer.GPTMini.Sparsemax.ClosedForm
import Transformer.GPTMini.QKNorm

/-!
# The one-unit support window and QKNorm gain

Consequences of arXiv:1602.02068v2, §2.2, Proposition 1 for the existing
causal simplex projection. These are real-arithmetic statements about one
row, not assertions about score distributions or entire training runs.

Future masked positions are excluded from comparisons with the maximum.
Support is defined by strictly positive weights, so an exact unit score
difference is already enough to exclude the lower-scoring coordinate.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Each coordinate of the actual projection is at most one. Source:
arXiv:1602.02068v2, §2.1, the probability simplex in equation `sparsemax`. -/
theorem sparseWeights_le_one {T : ℕ} (scores : Fin T → ℝ) (i j : Fin T) :
    sparseWeights scores i j ≤ 1 := by
  have hp := (sparseWeights_spec scores i).1
  have h := Finset.single_le_sum (fun k _ => hp.1 k) (Finset.mem_univ j)
  simpa only [hp.2.1] using h

/-- Equal scores project to equal weights on a two-position causal row.
Source: arXiv:1602.02068v2, §2.2, Proposition 1; useful nonempty premises. -/
theorem sparseWeights_two_equal :
    sparseWeights (fun _ : Fin 2 => 0) 1 = fun _ => (1 / 2 : ℝ) := by
  have hsum : ∑ j : Fin 2, thresholdWeights (fun _ => 0) 1 (-(1 / 2)) j = 1 := by
    norm_num [thresholdWeights, Fin.sum_univ_two]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  funext j
  have hj : j ≤ (1 : Fin 2) := by omega
  norm_num [thresholdWeights, hj]

/-- An active coordinate is visible. Source: arXiv:1602.02068v2, §2.1,
equation `sparsemax`, with the repository's causal-simplex restriction. -/
theorem sparseWeights_positive_visible {T : ℕ} (scores : Fin T → ℝ)
    (i j : Fin T) (hp : 0 < sparseWeights scores i j) : j ≤ i := by
  by_contra hj
  have hz := (sparseWeights_spec scores i).1.2.2 j hj
  linarith

/-- The active-coordinate premise is realized by a nontrivial row.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : 0 < sparseWeights (fun _ : Fin 2 => 0) 1 0 := by
  rw [sparseWeights_two_equal]
  norm_num

/-- Every active score is strictly within one of every visible score,
including the maximum. Source: arXiv:1602.02068v2, §2.2, Proposition 1:
an active score exceeds the threshold, while no weight exceeds one. -/
theorem sparseWeights_support_window {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (hp : 0 < sparseWeights scores i j) (hk : k ≤ i) :
    scores k < scores j + 1 := by
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold scores i
  have hj := sparseWeights_positive_visible scores i j hp
  have hpos : 0 < max (scores j - τ) 0 := by
    simpa only [hτ, thresholdWeights, hj, ite_true] using hp
  have hjτ : τ < scores j := by
    by_contra h
    have he : max (scores j - τ) 0 = 0 := max_eq_right (by linarith)
    linarith
  have hle : max (scores k - τ) 0 ≤ 1 := by
    simpa only [hτ, thresholdWeights, hk, ite_true] using sparseWeights_le_one scores i k
  have hmax := le_max_left (scores k - τ) 0
  linarith

/-- Two genuinely competing visible positions satisfy the window premises.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : 0 < sparseWeights (fun _ : Fin 2 => 0) 1 0 ∧ (1 : Fin 2) ≤ 1 := by
  constructor
  · rw [sparseWeights_two_equal]
    norm_num
  · exact le_rfl

/-- A visible competitor at least one above a score excludes that score
from support. Source: arXiv:1602.02068v2, §2.2, Proposition 1. Equality
is excluded too, because support is defined by a strictly positive weight. -/
theorem sparseWeights_zero_of_score_gap {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (hk : k ≤ i) (hgap : scores j + 1 ≤ scores k) :
    sparseWeights scores i j = 0 := by
  have hn := (sparseWeights_spec scores i).1.1 j
  by_contra hzero
  have hp : 0 < sparseWeights scores i j := lt_of_le_of_ne hn (Ne.symm hzero)
  have hw := sparseWeights_support_window scores i j k hp hk
  linarith

/-- The exclusion hypotheses hold with two visible, separated scores.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : (0 : Fin 2) ≤ 1 ∧
    (fun j : Fin 2 => if j = 0 then (2 : ℝ) else 0) 1 + 1 ≤
      (fun j : Fin 2 => if j = 0 then (2 : ℝ) else 0) 0 := by
  norm_num

/-- Scores of two active positions differ by strictly less than one.
Source: arXiv:1602.02068v2, §2.2, Proposition 1, applied in both directions. -/
theorem sparseWeights_active_score_range {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (hj : 0 < sparseWeights scores i j)
    (hk : 0 < sparseWeights scores i k) : |scores j - scores k| < 1 := by
  have hjv := sparseWeights_positive_visible scores i j hj
  have hkv := sparseWeights_positive_visible scores i k hk
  have h₁ := sparseWeights_support_window scores i j k hj hkv
  have h₂ := sparseWeights_support_window scores i k j hk hjv
  exact abs_lt.mpr ⟨by linarith, by linarith⟩

/-- The two-active-position hypotheses hold simultaneously.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : 0 < sparseWeights (fun _ : Fin 2 => 0) 1 0 ∧
    0 < sparseWeights (fun _ : Fin 2 => 0) 1 1 := by
  rw [sparseWeights_two_equal]
  norm_num

/-- QKNorm support retains only normalized similarities within `exp(-α)`
of the largest visible one. Source: arXiv:1602.02068v2, §2.2, Proposition 1,
combined with `GPTMini.score` at commit `73f8a0b`. With norms above `eps`
these inner products are cosines; below it they are clipped similarities.
RoPE-rotated query/key vectors may be supplied as the inputs. -/
theorem qknorm_support_window {T d : ℕ} (α eps : ℝ) (q : EucSpace d)
    (keys : Fin T → EucSpace d) (i j k : Fin T)
    (hp : 0 < sparseWeights (fun l => score α eps q (keys l)) i j)
    (hk : k ≤ i) :
    inner (𝕜 := ℝ) (normL2 eps q) (normL2 eps (keys k)) -
      inner (𝕜 := ℝ) (normL2 eps q) (normL2 eps (keys j)) < Real.exp (-α) := by
  have hw := sparseWeights_support_window (fun l => score α eps q (keys l)) i j k hp hk
  have he : Real.exp α * Real.exp (-α) = 1 := by
    rw [Real.exp_neg, mul_inv_cancel₀ (ne_of_gt (Real.exp_pos α))]
  unfold score at hw
  by_contra h
  have hm := mul_le_mul_of_nonneg_left (le_of_not_gt h) (Real.exp_pos α).le
  nlinarith

/-- The QKNorm support-window hypotheses are inhabited, including epsilon
clipping: zero query/key vectors yield an equal-score, full-support row.
Source context: `GPTMini.score`, commit `73f8a0b`, and Proposition 1. -/
example : 0 < sparseWeights (fun _ : Fin 2 => score 0 1 (0 : EucSpace 1) 0) 1 0 ∧
    (1 : Fin 2) ≤ 1 := by
  have hs : (fun _ : Fin 2 => score 0 1 (0 : EucSpace 1) 0) = fun _ => 0 := by
    funext j
    simp [score, normL2]
  rw [hs, sparseWeights_two_equal]
  norm_num

end Transformer.GPTMini.Sparsemax
