/-
# Selective attention used in the hybrid experiments

Arora et al., arXiv:2312.04927v1, §5, equations `eq:selective` and
`eq:programmatic`.  The selector acts on query rows; the attention weights
inside a selected row still depend on that row's input-dependent scores.
-/

import Transformer.Zoology.Section4_Coyote

open scoped BigOperators

namespace Transformer.Zoology

/-- The rowwise softmax-weighted value output, before any row selection.
Source: §5, equation `eq:selective`. -/
noncomputable def attentionFromScores {n d : ℕ}
    (scores : Fin n → Fin n → ℝ) (values : RealSequence n d) :
    RealSequence n d :=
  fun i q => ∑ j, (Real.exp (scores i j) / ∑ k, Real.exp (scores i k)) *
    values j q

/-- Rowwise selected attention.  The paper's `f(u)[i] ∈ {0,1}` is represented
by `Bool`; the selector may itself depend on the input sequence.
Source: §5, equation `eq:selective`. -/
noncomputable def selectiveAttention {n d : ℕ}
    (scores : RealSequence n d → Fin n → Fin n → ℝ)
    (values : RealSequence n d → RealSequence n d)
    (select : RealSequence n d → Fin n → Bool)
    (u : RealSequence n d) : RealSequence n d :=
  fun i q => if select u i then attentionFromScores (scores u) (values u) i q else 0

/-- Selecting every row recovers full attention exactly.
Source: §5, full-attention choice below equation `eq:selective`. -/
theorem selective_full_eq_attention {n d : ℕ}
    (scores : RealSequence n d → Fin n → Fin n → ℝ)
    (values : RealSequence n d → RealSequence n d)
    (u : RealSequence n d) :
    selectiveAttention scores values (fun _ _ => true) u =
      attentionFromScores (scores u) (values u) := by
  funext i q
  simp [selectiveAttention]

/-- The programmatic selector tests for any earlier occurrence of the same
raw token.  Source: §5, equation `eq:programmatic`. -/
def programmaticSelection {n c : ℕ} (x : TokenSequence n c)
    (i : Fin n) : Bool :=
  decide (∃ j : Fin n, j < i ∧ x j = x i)

/-- Every raw-sequence MQAR hit is selected by the programmatic rule.
Source: §5, equation `eq:programmatic`, combined with §3 Definition
`def: general-AR`. -/
theorem programmatic_selects_recall {n c : ℕ} (x : TokenSequence n c)
    (i : Fin n) (v : Fin c) (h : RawAnswer x i v) :
    programmaticSelection x i = true := by
  obtain ⟨j, hj, _, hkey, _⟩ := h
  simp [programmaticSelection, show ∃ k : Fin n, k < i ∧ x k = x i from
    ⟨j, hj, hkey⟩]

/-- The learned selector's sparse-count auxiliary loss.
Source: §5, paragraph `Learned selection`. -/
noncomputable def selectionAuxiliaryLoss {n : ℕ} (f : Fin n → ℝ) (k : ℕ) : ℝ :=
  max 0 ((∑ i, f i) - k) / n

/-- The auxiliary loss is nonnegative for positive sequence length.
Source: §5, learned-selection auxiliary-loss formula. -/
theorem selectionAuxiliaryLoss_nonneg {n : ℕ} (hn : 0 < n)
    (f : Fin n → ℝ) (k : ℕ) :
    0 ≤ selectionAuxiliaryLoss f k := by
  unfold selectionAuxiliaryLoss
  exact div_nonneg (le_max_left _ _) (le_of_lt (Nat.cast_pos.mpr hn))

/-- The auxiliary penalty vanishes when the selected mass is within budget.
Source: §5, learned-selection auxiliary-loss formula. -/
theorem selectionAuxiliaryLoss_zero {n : ℕ} (f : Fin n → ℝ)
    (k : ℕ) (hbudget : ∑ i, f i ≤ k) :
    selectionAuxiliaryLoss f k = 0 := by
  unfold selectionAuxiliaryLoss
  have h : (∑ i, f i) - k ≤ 0 := sub_nonpos.mpr hbudget
  simp [max_eq_left h]

/-- The length hypothesis and the budget hypothesis above are both
satisfiable for a nonempty sequence with no selected rows. -/
example : (0 : ℕ) < 1 ∧
    (∑ i : Fin 1, (fun _ : Fin 1 => (0 : ℝ)) i) ≤ (0 : ℕ) := by
  norm_num

end Transformer.Zoology
