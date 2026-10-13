import Transformer.GPTMini.Semantics.AdjacencyRoPE

/-!
# Previous-token concentration without an assumed score gap

Source: the original small Basis recall dimensions at cbafbe9,
AdjacencyRoPE's explicit original shifted pair, and CausalMHA.forward's
finite softmax/XSA at f11b6e2. Every non-predecessor visible position has
an explicitly proved score gap. The actual causal softmax therefore
copies a previous raw value with exponentially small error. At positions
whose projected self-value is zero, original XSA leaves this copy intact.

This removes the positional score-gap premise of the earlier conditional
head lemma. Simultaneous raw embedding/QKV realization and the subsequent
key-conditioned latest-write readout are separate encoder obligations.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The original small Basis recall model, retaining width/layers/heads/FFN and its actual vocabulary/context.
Source: domain/basis.py and MQAR.context/vocab at cbafbe9. -/
abbrev recallConfig : Config where
  vocab_size := 548
  n_layers := 2
  n_heads := 4
  d_model := 64
  d_ff := 256
  max_seq_len := 64
  rope_theta := 10000
  divides := by decide
  head_even := by decide
  n_heads_pos := by decide
  n_layers_pos := by decide
  d_model_pos := by decide
  d_ff_pos := by decide
  vocab_pos := by decide
  max_seq_len_pos := by decide
  theta_pos := by norm_num

/-- The concrete finite-temperature predecessor gap in this original head.
Source: the proved actual cosine peak and strongest one-step competitor. -/
noncomputable def adjacentGap (alpha : ℝ) : ℝ :=
  Real.exp alpha * (1 - Real.cos adjacentFrequency)

/-- This concrete positional gap is positive for every finite learned temperature.
Source: positivity of the actual frequency/cosine separation and exp(alpha). -/
theorem adjacentGap_pos (alpha : ℝ) : 0 < adjacentGap alpha :=
  mul_pos (Real.exp_pos alpha) adjacent_cosine_gap_pos

/-- Every discrete causal non-predecessor is below the actual peak by this explicit gap.
Source: original rotated Q/K and the proven concrete frequency; no prepared score-gap hypothesis. -/
theorem adjacent_preScore_gap (alpha eps : ℝ) (heps : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 128) (i selected : Fin T) (hprev : selected.val + 1 = i.val) :
    ∀ j : Fin T, j ≠ selected → j.val ≤ i.val →
      preScore recallConfig alpha eps
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) i j ≤
        preScore recallConfig alpha eps
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) i selected - adjacentGap alpha := by
  intro j hne hj
  have hd : (j.val : ℤ) - i.val + 1 ≠ 0 := by
    intro hz
    have he : j.val = selected.val := by omega
    exact hne (Fin.ext he)
  have hi : (i.val : ℝ) < 128 := by exact_mod_cast (lt_of_lt_of_le i.isLt hT)
  have hj' : (j.val : ℝ) ≤ i.val := by exact_mod_cast hj
  have hbound : |(((j.val : ℤ) - i.val + 1 : ℤ) : ℝ)| ≤ 128 := by
    simp only [Int.cast_add, Int.cast_sub, Int.cast_natCast, Int.cast_one]
    apply abs_le.mpr
    constructor <;> linarith [Nat.cast_nonneg (α := ℝ) j.val]
  have hc := adjacent_cosine_displacement _ hd hbound
  simp only [Int.cast_add, Int.cast_sub, Int.cast_natCast, Int.cast_one] at hc
  dsimp only [preScore]
  change score (d_head := 16) alpha eps (applyRope 16 10000 (i.val : ℝ) adjacentQuery)
    (applyRope 16 10000 (j.val : ℝ) adjacentDirection) ≤
      score (d_head := 16) alpha eps (applyRope 16 10000 (i.val : ℝ) adjacentQuery)
        (applyRope 16 10000 (selected.val : ℝ) adjacentDirection) - adjacentGap alpha
  rw [adjacent_score _ _ _ _ heps, adjacent_score_peak _ _ heps _ _ hprev]
  have hm := mul_le_mul_of_nonneg_left hc (Real.exp_pos alpha).le
  unfold adjacentGap
  nlinarith

example : (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    (0 : Fin 2).val + 1 = (1 : Fin 2).val := by norm_num

/-- The predecessor is the unique causal maximizer of the actual original positional score.
Source: the derived positive gap, including the unmasked self competitor and every earlier visible token. -/
theorem adjacent_previous_unique (alpha eps : ℝ) (heps : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 128) (i selected : Fin T) (hprev : selected.val + 1 = i.val) :
    ∀ j : Fin T, j ≠ selected → j.val ≤ i.val →
      preScore recallConfig alpha eps
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) i j <
        preScore recallConfig alpha eps
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
          (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) i selected := by
  intro j hne hj
  have h := adjacent_preScore_gap alpha eps heps hT i selected hprev j hne hj
  have hp := adjacentGap_pos alpha
  linarith

example : (1 / 100000 : ℝ) ≤ 1 ∧ (64 : ℕ) ≤ 128 ∧
    (62 : Fin 64).val + 1 = (63 : Fin 64).val := by norm_num

/-- The actual softmax mass outside the predecessor is exponentially small, at the unchanged original mask.
Source: tail_mass_le applied with the now-derived original positional gap. -/
theorem adjacent_tail_mass_le (alpha eps : ℝ) (heps : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 128) (i selected : Fin T) (hprev : selected.val + 1 = i.val) :
    1 - causalAttnWeights recallConfig alpha eps
        (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
        (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) i selected ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap alpha) :=
  tail_mass_le recallConfig alpha eps _ _ _ i selected (by omega)
    (adjacent_preScore_gap alpha eps heps hT i selected hprev)

example : (1 / 100000 : ℝ) ≤ 1 ∧ (3 : ℕ) ≤ 128 ∧
    (1 : Fin 3).val + 1 = (2 : Fin 3).val := by norm_num

/-- Original softmax and XSA copy the predecessor with a derived finite error when self-values are zero.
Source: the actual attentionHead, not a new shift operator; the raw value array remains explicit. -/
theorem adjacent_head_copy (alpha eps B : ℝ) (heps : eps ≤ 1) (hB : 0 ≤ B) {T : ℕ}
    (hT : T ≤ 128) (v : Fin T → EucSpace recallConfig.head_dim) (i selected : Fin T)
    (hprev : selected.val + 1 = i.val) (hself : v i = 0)
    (hvalues : ∀ j, j ≠ selected → ‖v j - v selected‖ ≤ B) :
    ‖attentionHead recallConfig alpha eps (fun _ => adjacentQuery) (fun _ => adjacentDirection)
        v (fun r => (r.val : ℝ)) i - v selected‖ ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap alpha) * B := by
  have ho := output_error_le recallConfig alpha eps
    (fun r => applyRope 16 10000 (r.val : ℝ) adjacentQuery)
    (fun r => applyRope 16 10000 (r.val : ℝ) adjacentDirection) v i selected B hvalues
  have ht := mul_le_mul_of_nonneg_right
    (adjacent_tail_mass_le alpha eps heps hT i selected hprev) hB
  have h := ho.trans ht
  unfold attentionHead xsaProjection
  simp only [hself, normL2, norm_zero, smul_zero, inner_zero_right, sub_zero, ite_true]
  exact h

example : (1 / 100000 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun i : Fin 2 => if i = 0 then adjacentDirection else 0) 1 = 0 ∧
    (∀ j : Fin 2, j ≠ 0 →
      ‖(if j = 0 then adjacentDirection else 0) - adjacentDirection‖ ≤ (1 : ℝ)) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by simp, ?_⟩
  intro j hj
  simp only [ite_eq_right hj, zero_sub, norm_neg, adjacent_vectors_norm.1]
  norm_num

end Transformer.GPTMini.Semantics
