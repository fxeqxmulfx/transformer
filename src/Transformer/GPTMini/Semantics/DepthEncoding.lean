import Transformer.GPTMini.Semantics.DepthInvariant

/-!
# Actual original depth detector encoder and ordered-state induction

Source: original blockForward at f11b6e2 and the neutral-preserving
Basis ordered occurrence recurrence at cbafbe9. The real complete
detector transition writes the next opposite-predecessor occurrence
with amplitude in [r^2,128], and zero under independent absence.
All head/FFN conditions come from the incoming explicit invariant.

The complete original block also preserves raw constant/types, keeps
future stage channels empty and derives the next actual growth bound
2+288*(level+1). The invariant is proved for the real output array;
it is not used to define that array or supplied as a model encoder.

The actual raw-word base and this substantive step derive the entire
detector encoder invariant. Easy uses one detector and hard three;
both leave room for readout within their unchanged original layer
budgets. Full ModelParams/readout/integer coupling remains to be done.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- The true next original block represents precisely the independent next ordered-occurrence level.
Source: real type-and-strictly-earlier feature transition, exact raw/source semantics and quantitative true FFN bounds. -/
theorem depthDetectorBlock_representation (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) (word : List (Option Bool)) (hT : word.length ≤ 128)
    (positions : Fin word.length → ℝ) (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨stage.val, by have hs := stage.isLt; omega⟩ word x) :
    DepthLevelRepresentation mode ⟨stage.val + 1, by have hs := stage.isLt; omega⟩ word
      (blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x) := by
  have hinput := depthStateInvariant_detectorInput mode stage word x h
  have hnorm := depthStateInvariant_domain mode _ word x h
  have hn (i : Fin word.length) : ‖x i‖ ≤ 4096 := (hnorm i).trans (by norm_num)
  have hpos (b : Fin 2) (i : Fin word.length) :
      0 < blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
        (depthFeatureCoordinate mode stage b) ↔ DepthOccurrence (stage.val + 1) (depthBranchLetter b) word i := by
    rw [depthDetectorBlock_feature_ordered mode stage eps heps hclip hT positions x hinput hn i b,
      depthStateInvariant_type_iff mode _ word x h b i, depthOccurrence_succ]
    constructor
    · rintro ⟨hc, j, hj, hf⟩
      exact ⟨hc, j, hj, (depthStateInvariant_source_iff mode stage word x h b j).mp hf⟩
    · rintro ⟨hc, j, hj, hf⟩
      exact ⟨hc, j, hj, (depthStateInvariant_source_iff mode stage word x h b j).mpr hf⟩
  intro b i
  rw [depthLevelCoordinate_successor]
  have hb := depthDetectorBlock_feature_bounds mode stage eps heps hclip hT positions x hinput hnorm i b
  constructor
  · intro hp
    have hf := (hpos b i).mpr hp
    rcases hb.1 with hz | hg
    · rw [hz] at hf
      exact False.elim ((lt_irrefl 0) hf)
    · exact ⟨hg, hb.2⟩
  · intro hp
    rcases hb.1 with hz | hg
    · exact hz
    · have hr := sq_pos_of_pos depthScaleLower_bounds.1
      have hf : 0 < blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
          (depthFeatureCoordinate mode stage b) := by linarith
      exact False.elim (hp ((hpos b i).mp hf))

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .easy 0 [some false, none, some true]
      (depthWordInput .easy 1 [some false, none, some true]) := by
  exact ⟨by norm_num, by norm_num, by decide, depthStateInvariant_input .easy 1 _⟩

/-- One genuine original detector block preserves every clause of the quantified ordered-state invariant.
Source: complete real occurrence representation, both actual protected residuals, fresh future stages and proved M+288 growth.
The next norm/zero-self/presence requirements are derived for the actual output, not assumed. -/
theorem depthStateInvariant_step (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) (word : List (Option Bool)) (hT : word.length ≤ 128)
    (positions : Fin word.length → ℝ) (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨stage.val, by have hs := stage.isLt; omega⟩ word x) :
    DepthStateInvariant mode ⟨stage.val + 1, by have hs := stage.isLt; omega⟩ word
      (blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x) := by
  refine ⟨?_, ?_, depthDetectorBlock_representation mode stage eps heps hclip word hT positions x h, ?_, ?_⟩
  · intro i
    rw [depthDetectorBlock_preserves_raw mode stage eps positions x i 0 (by decide)]
    exact h.1 i
  · intro b i
    change blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthCoordinate mode ⟨1 + b.val, by have hb := b.isLt; omega⟩) = _
    rw [depthDetectorBlock_preserves_raw mode stage eps positions x i
      ⟨1 + b.val, by have hb := b.isLt; omega⟩ (by change 1 + b.val < 3; have hb := b.isLt; omega)]
    exact h.2.1 b i
  · intro other hfuture part i
    have hne : other ≠ stage := by
      intro he
      subst other
      change stage.val + 1 ≤ stage.val at hfuture
      omega
    rw [depthDetectorBlock_preserves_stage mode stage other hne eps positions x i part]
    exact h.2.2.2.1 other (by
      change stage.val ≤ other.val
      change stage.val + 1 ≤ other.val at hfuture
      omega) part i
  · intro i
    have hs : (stage.val : ℝ) ≤ 2 := by
      have hn : stage.val ≤ 2 := by have hs := stage.isLt; omega
      exact_mod_cast hn
    have hn := depthDetectorBlock_norm mode stage eps heps hclip hT positions x
      (depthStateInvariant_detectorInput mode stage word x h)
      (2 + 288 * (stage.val : ℝ)) (by linarith) h.2.2.2.2 i
    change ‖blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i‖ ≤
      2 + 288 * ((stage.val + 1 : ℕ) : ℝ)
    rw [Nat.cast_add, Nat.cast_one]
    linarith

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .hard 0 [some false, none, some true]
      (depthWordInput .hard 1 [some false, none, some true]) := by
  exact ⟨by norm_num, by norm_num, by decide, depthStateInvariant_input .hard 1 _⟩

/-- One detector for E_2, three for E_4, before the separate original readout block.
Source: the independent ordered occurrence lengths two/four, with the raw letter already being level zero. -/
def depthDetectorCount : Mode → ℕ
  | .easy => 1
  | .hard => 3

/-- Every used detector stage is available and one readout block fits the unchanged original model.
Source: actual easy two-layer and hard six-layer Config records; unused hard layers can be identities. -/
theorem depthDetectorCount_slots (mode : Mode) :
    depthDetectorCount mode ≤ 3 ∧ depthDetectorCount mode + 1 ≤ (depthConfig mode).n_layers := by
  cases mode <;> decide

/-- The genuine detector prefix, computed only from local raw embeddings and original blockForward.
Source: fixed shared detector parameters at each allowed stage; this helper freezes after that prefix and does not include readout. -/
noncomputable def depthWordState (mode : Mode) (gain eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) : ℕ → Fin word.length → EucSpace (depthConfig mode).d_model
  | 0 => depthWordInput mode gain word
  | n + 1 => if h : n < depthDetectorCount mode then
      blockForward (depthConfig mode) (depthDetectorBlock mode ⟨n, by have hc := (depthDetectorCount_slots mode).1; omega⟩)
        eps positions (depthWordState mode gain eps word positions n)
    else depthWordState mode gain eps word positions n

/-- Every used real detector-prefix state derives the full independent ordered and quantitative invariant from the raw word.
Source: actual token-local embedding base and genuine block-preservation induction; no encoder/routing/output hypothesis is supplied. -/
theorem depthWordState_invariant (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (n : ℕ) (hn : n ≤ depthDetectorCount mode) :
    DepthStateInvariant mode ⟨n, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word
      (depthWordState mode gain eps word positions n) := by
  revert hn
  induction n with
  | zero =>
      intro hn
      exact depthStateInvariant_input mode gain word
  | succ n ih =>
      intro hn
      have hprev : n ≤ depthDetectorCount mode := by omega
      have hnext : n < depthDetectorCount mode := by omega
      have h := ih hprev
      rw [depthWordState, dite_eq_left hnext]
      exact depthStateInvariant_step mode ⟨n, by have hc := (depthDetectorCount_slots mode).1; omega⟩
        eps heps hclip word hT positions _ h

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, none, some true] : List (Option Bool)).length ≤ 128 ∧ 1 ≤ depthDetectorCount .easy := by
  exact ⟨by norm_num, by norm_num, by decide, by decide⟩

end Transformer.GPTMini.Semantics
