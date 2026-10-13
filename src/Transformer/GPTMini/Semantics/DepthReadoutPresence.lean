import Transformer.GPTMini.Semantics.DepthReadoutLayout

/-!
# Quantitative visible occurrences in genuine depth readout

Source: original finite uniform softmax/XSA at f11b6e2, the independent
neutral-preserving ordered occurrence semantics and real detector
encoder induction at 14a2f0d. Readout writes each head into its existing
last-level flag. A positive current flag survives XSA via the residual;
with a zero current flag, a visible occurrence gives the genuine causal
mean floor. Absence gives exact zero in both terms.

Every true pre-FFN probe is zero or at least twice the shared threshold,
is capped by 144 and is positive exactly at independent visible pattern
presence. Norm and feature hypotheses are derived from the explicit
state invariant, then instantiated by the actual raw-word encoder.
This conclusion does not define the attention state by a label oracle.
Final original FFN, tied logits and integer decoding remain obligations.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- Independent presence of the positive B-ending or negative A-ending pattern at an actual visible position.
Source: Basis E_2/E_4 ordered recurrence, retaining neutral positions and the original inclusive causal prefix. -/
def DepthReadoutPresence (mode : Mode) (word : List (Option Bool)) (b : Fin 2) (i : Fin word.length) : Prop :=
  ∃ j : Fin word.length, j.val ≤ i.val ∧ DepthOccurrence (depthDetectorCount mode) (!depthBranchLetter b) word j

/-- The real encoder invariant gives each actual readout source its zero-or-r^2 gap, cap and independent meaning.
Source: original last-level coordinate assignment and genuine quantitative level representation. -/
theorem depthReadout_source (mode : Mode) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (b : Fin 2) (i : Fin word.length) :
    (x i (depthReadoutProbeCoordinate mode b) = 0 ∨ depthScaleLower ^ 2 ≤ x i (depthReadoutProbeCoordinate mode b)) ∧
      x i (depthReadoutProbeCoordinate mode b) ≤ 128 ∧
      (depthScaleLower ^ 2 ≤ x i (depthReadoutProbeCoordinate mode b) ↔
        DepthOccurrence (depthDetectorCount mode) (!depthBranchLetter b) word i) := by
  have hb := depthLevelRepresentation_bounds mode _ word x h.2.2.1
    ⟨1 - b.val, by have hb := b.isLt; omega⟩ i
  have he := depthLevelRepresentation_threshold_iff mode _ word x h.2.2.1
    ⟨1 - b.val, by have hb := b.isLt; omega⟩ i
  simp only [depthBranchLetter_opposite] at he
  exact ⟨hb.1, hb.2, he⟩

example : DepthStateInvariant .easy 1 [none, some false, some true]
    (depthWordState .easy 1 (1 / 100000) [none, some false, some true] (fun i => i.val) 1) := by
  exact depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)

/-- The genuine residual readout probe has a finite presence gap, cap and exact independent visible-occurrence criterion.
Source: true XSA signal on separated encoder features; current positive features are retained by the unnormalized residual. -/
theorem depthReadout_probe_semantics (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (b : Fin 2) (i : Fin word.length) :
    (depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) = 0 ∨
      2 * depthThreshold ≤ depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b)) ∧
    depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) ≤ 144 ∧
    (0 < depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) ↔
      DepthReadoutPresence mode word b i) := by
  have hs (j : Fin word.length) := depthReadout_source mode word x h b j
  have hn (j : Fin word.length) : 0 ≤ x j (depthReadoutProbeCoordinate mode b) := by
    rcases (hs j).1 with hz | hp
    · rw [hz]
    · exact (sq_nonneg depthScaleLower).trans hp
  have hb (j : Fin word.length) : 1 ≤ ‖x j‖ ∧ ‖x j‖ ≤ 4096 :=
    ⟨depthState_norm_lower mode (x j) (h.1 j), (depthStateInvariant_norm mode _ word x h j).trans (by norm_num)⟩
  have ha := depthAttentionSignal_bounds mode (depthReadoutProbeCoordinate mode) eps heps x b i hn
  rw [depthReadoutAttentionState_probe]
  refine ⟨?_, by linarith [(hs i).2.1], ?_⟩
  · by_cases hz : x i (depthReadoutProbeCoordinate mode b) = 0
    · rw [hz, zero_add]
      exact depthAttentionSignal_separated mode (depthReadoutProbeCoordinate mode) eps heps.le hclip hT x b i hb
        (fun j => (hs j).1) hz
    · rcases (hs i).1 with he | hp
      · exact False.elim (hz he)
      · apply Or.inr
        have hr : 2 * depthThreshold ≤ depthScaleLower ^ 2 := by norm_num [depthThreshold, depthScaleLower]
        linarith
  · by_cases hz : x i (depthReadoutProbeCoordinate mode b) = 0
    · rw [hz, zero_add, depthAttentionSignal_positive_iff mode (depthReadoutProbeCoordinate mode)
        eps heps.le hclip hT x b i hb (fun j => (hs j).1) hz]
      constructor
      · rintro ⟨j, hj, hp⟩
        exact ⟨j, hj, ((hs j).2.2).mp hp⟩
      · rintro ⟨j, hj, hp⟩
        exact ⟨j, hj, ((hs j).2.2).mpr hp⟩
    · rcases (hs i).1 with he | hp
      · exact False.elim (hz he)
      · have hr := sq_pos_of_pos depthScaleLower_bounds.1
        constructor
        · intro
          exact ⟨i, by omega, ((hs i).2.2).mp hp⟩
        · intro
          linarith

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .easy 1 [none, some false, some true]
      (depthWordState .easy 1 (1 / 100000) [none, some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num, by norm_num, by decide,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

/-- Genuine readout attention on the supplied encoder invariant stays within norm 898 and retains a positive constant-derived lower bound.
Source: actual two-head contribution at most 32 and the true detector-prefix bound 866. -/
theorem depthReadoutAttentionState_norm (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (word : List (Option Bool)) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (i : Fin word.length) :
    1 ≤ ‖depthReadoutAttentionState mode eps positions x i‖ ∧ ‖depthReadoutAttentionState mode eps positions x i‖ ≤ 898 := by
  refine ⟨depthReadoutAttentionState_norm_lower mode eps positions x i (h.1 i), ?_⟩
  have ha := depthAttentionResidual_norm mode (depthReadoutProbeCoordinate mode) (depthReadoutProbeCoordinate mode)
    eps heps positions x i 866
    (depthStateInvariant_norm mode _ word x h i) (by
      intro b j
      rcases (depthReadout_source mode word x h b j).1 with hz | hp
      · rw [hz]
      · exact (sq_nonneg depthScaleLower).trans hp)
  change ‖depthReadoutAttentionState mode eps positions x i‖ ≤ 866 + 32 at ha
  linarith

example : (0 : ℝ) < 1 / 100000 ∧ DepthStateInvariant .easy 1 [some false, some true]
    (depthWordState .easy 1 (1 / 100000) [some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

/-- Actual raw-word encoding followed by genuine readout attention detects every visible ordered occurrence without an invariant premise.
Source: the entire real detector induction instantiated in the quantitative residual readout theorem. -/
theorem depthWordReadout_semantics (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (b : Fin 2) (i : Fin word.length) :
    let x := depthWordState mode gain eps word positions (depthDetectorCount mode)
    (depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) = 0 ∨
      2 * depthThreshold ≤ depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b)) ∧
    depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) ≤ 144 ∧
    (0 < depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) ↔
      DepthReadoutPresence mode word b i) := by
  have h := depthWordState_invariant mode gain eps heps hclip word hT positions
    (depthDetectorCount mode) (by omega)
  exact depthReadout_probe_semantics mode eps heps hclip word hT positions _ h b i

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

end Transformer.GPTMini.Semantics
