import Transformer.GPTMini.Semantics.DepthReadoutScores
import Transformer.GPTMini.TokenInterface.Decoding

/-!
# Strict original depth tied readout and greedy answer

Source: original tied unembed/final RMSNorm at f11b6e2, actual raw-word
state at 59cf964 and the original 36-entry depth vocabulary. The true
semantic label score is at least five, the opposite label at most minus
three and every remaining token at most two. All scores derive from
the same actual token-local embedding table and computed hidden state.

The true final RMS multiplier is positive, so all strict comparisons
survive original normalization and full-vocabulary greedy decoding.
The result returns the independent ordered-pattern semantic target on
every raw word position, without an encoder, route or logit hypothesis.
It proves real arithmetic behavior of these given ordinary parameters;
floating-point/AdamW guarantees and convex trainability do not follow.
Complete ModelParams/loop and checked integer-prefix coupling remain.
The criterion rejects both-pattern and neither-pattern cases as well
as the wrong active start; final-position visibility retains neutrals.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- The actual correct label defeats its opposite quantitatively for every raw word and visible prefix.
Source: preserved true constant one, genuine tied label branches and derived decision margin four. -/
theorem depthWordFinalState_label_scores (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    let x := depthWordFinalState mode depthLabelGain eps word positions i
    if DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i then
      5 ≤ inner (𝕜 := ℝ) x (depthRawEmbedding mode depthLabelGain 16) ∧
        inner (𝕜 := ℝ) x (depthRawEmbedding mode depthLabelGain 15) ≤ -3
    else 5 ≤ inner (𝕜 := ℝ) x (depthRawEmbedding mode depthLabelGain 15) ∧
        inner (𝕜 := ℝ) x (depthRawEmbedding mode depthLabelGain 16) ≤ -3 := by
  have hm := depthWordFinalState_label_margin mode eps heps hclip word hT positions i
  have hc := (depthWordFinalState_raw mode depthLabelGain eps heps hclip word hT positions i).1
  have hl := depthRawEmbedding_labels mode depthLabelGain (depthWordFinalState mode depthLabelGain eps word positions i)
  dsimp only
  rw [hl.1, hl.2, hc]
  by_cases hp : DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i
  · rw [ite_eq_left hp] at hm ⊢
    constructor <;> linarith
  · rw [ite_eq_right hp] at hm ⊢
    constructor <;> linarith

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- The true raw-word answer strictly defeats every other actual tied vocabulary entry, including all filler and input tokens.
Source: genuine correct/opposite label bounds five/minus-three and derived nonlabel cap two across all 36 entries. -/
theorem depthWordFinalState_tied_margin (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length)
    (token : Fin 36) (hne : token ≠ depthWordAnswer mode word i) :
    inner (𝕜 := ℝ) (depthWordFinalState mode depthLabelGain eps word positions i) (depthRawEmbedding mode depthLabelGain token) <
      inner (𝕜 := ℝ) (depthWordFinalState mode depthLabelGain eps word positions i)
        (depthRawEmbedding mode depthLabelGain (depthWordAnswer mode word i)) := by
  have hl := depthWordFinalState_label_scores mode eps heps hclip word hT positions i
  by_cases hp : DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i
  · rw [depthWordAnswer, ite_eq_left hp] at hne ⊢
    rw [ite_eq_left hp] at hl
    have hn : token.val ≠ 16 := by intro he; apply hne; exact Fin.ext he
    by_cases hr : token.val = 15
    · have he : token = 15 := Fin.ext hr
      rw [he]
      linarith [hl.1, hl.2]
    · have ho := depthWordFinalState_nonlabel_score mode depthLabelGain eps heps hclip word hT positions i token hn hr
      linarith [hl.1]
  · rw [depthWordAnswer, ite_eq_right hp] at hne ⊢
    rw [ite_eq_right hp] at hl
    have hn : token.val ≠ 15 := by intro he; apply hne; exact Fin.ext he
    by_cases ha : token.val = 16
    · have he : token = 16 := Fin.ext ha
      rw [he]
      linarith [hl.1, hl.2]
    · have ho := depthWordFinalState_nonlabel_score mode depthLabelGain eps heps hclip word hT positions i token ha hn
      linarith [hl.1]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 ∧
    (0 : Fin 36) ≠ depthWordAnswer .easy [some false, none, some true] 2 := by
  refine ⟨by norm_num, by norm_num, by decide, ?_⟩
  unfold depthWordAnswer
  split_ifs <;> decide

/-- The original final RMSNorm preserves every strict whole-vocabulary true-answer comparison.
Source: actual final norm lower bound one and the positive genuine original RMS multiplier. -/
theorem depthWordFinalState_rms_margin (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length)
    (token : Fin 36) (hne : token ≠ depthWordAnswer mode word i) :
    inner (𝕜 := ℝ) (rmsNormEps eps (depthWordFinalState mode depthLabelGain eps word positions i)) (depthRawEmbedding mode depthLabelGain token) <
      inner (𝕜 := ℝ) (rmsNormEps eps (depthWordFinalState mode depthLabelGain eps word positions i))
        (depthRawEmbedding mode depthLabelGain (depthWordAnswer mode word i)) := by
  rw [depthScale_rms, real_inner_smul_left, real_inner_smul_left]
  exact mul_lt_mul_of_pos_left (depthWordFinalState_tied_margin mode eps heps hclip word hT positions i token hne)
    (depthScale_pos mode eps heps.le _ (depthWordFinalState_bounds mode depthLabelGain eps heps hclip word hT positions i).1.1)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    (0 : Fin 36) ≠ depthWordAnswer .hard [none, some false, some true] 3 := by
  refine ⟨by norm_num, by norm_num, by decide, ?_⟩
  unfold depthWordAnswer
  split_ifs <;> decide

/-- Greedy decoding of the genuine final normalized tied scores returns the independent raw-word semantic answer.
Source: actual strict margins over every other token and the original deterministic whole-vocabulary decoder. -/
theorem depthWordFinalState_greedy (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    TokenInterface.bestToken (depthConfig mode).vocab_pos (fun token =>
      inner (𝕜 := ℝ) (rmsNormEps eps (depthWordFinalState mode depthLabelGain eps word positions i)) (depthRawEmbedding mode depthLabelGain token)) =
        depthWordAnswer mode word i := by
  apply TokenInterface.bestToken_of_strict
  exact depthWordFinalState_rms_margin mode eps heps hclip word hT positions i

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- At the true final position, independent readout presence is exactly full-word ordered-pattern presence.
Source: the unchanged inclusive causal mask and genuine last-position bound, retaining all original neutral positions. -/
theorem depthReadoutPresence_last (mode : Mode) (word : List (Option Bool)) (i : Fin word.length)
    (hlast : i.val + 1 = word.length) (b : Fin 2) :
    DepthReadoutPresence mode word b i ↔ ∃ j : Fin word.length, DepthOccurrence (depthDetectorCount mode) (!depthBranchLetter b) word j := by
  constructor
  · rintro ⟨j, _, hp⟩
    exact ⟨j, hp⟩
  · rintro ⟨j, hp⟩
    exact ⟨j, by have hj := j.isLt; omega, hp⟩

example : (2 : Fin ([none, some false, some true] : List (Option Bool)).length).val + 1 =
    ([none, some false, some true] : List (Option Bool)).length := by decide

/-- The independent final target is exactly the full-word B-pattern/no-A-pattern criterion.
Source: real last-position presence equivalence and the original positive/negative branch letters. -/
theorem depthWordAnswer_last (mode : Mode) (word : List (Option Bool)) (i : Fin word.length)
    (hlast : i.val + 1 = word.length) :
    depthWordAnswer mode word i = if (∃ j : Fin word.length, DepthOccurrence (depthDetectorCount mode) true word j) ∧
      ¬(∃ j : Fin word.length, DepthOccurrence (depthDetectorCount mode) false word j) then 16 else 15 := by
  unfold depthWordAnswer
  rw [depthReadoutPresence_last mode word i hlast 0, depthReadoutPresence_last mode word i hlast 1]
  norm_num [depthBranchLetter]

example : (2 : Fin ([some false, none, some true] : List (Option Bool)).length).val + 1 =
    ([some false, none, some true] : List (Option Bool)).length := by decide

end Transformer.GPTMini.Semantics
