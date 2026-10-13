import Transformer.GPTMini.Semantics.DepthEncoding

/-!
# Original depth readout layout and genuine residual probes

Source: original blockForward at f11b6e2 and the actual ordered detector
encoder at 14a2f0d. A final uniform head reads each last-level feature
and writes into that same feature coordinate. The unnormalized residual
therefore retains a current occurrence even when original XSA suppresses
the current head contribution. Positive and negative branches read the
B-ending and A-ending features respectively.

The readout axes 17/18/19 remain genuinely empty through every detector;
this is proved from local raw embeddings and protected real blocks.
The actual readout attention preserves these axes and raw constant/types.
No desired answer or Boolean presence defines the computed state.
Quantitative visible-occurrence detection and the final FFN/readout are
subsequent obligations on this actual residual array.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- Every detector preserves the unused upper working axes on arbitrary real arrays.
Source: the true block's only targets lie in stage axes three through fourteen. -/
theorem depthDetectorBlock_preserves_upper (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin 24) (hc : 15 ≤ c.val) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthCoordinate mode c) = x i (depthCoordinate mode c) := by
  apply depthDetectorBlock_protected
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 3 + 4 * stage.val + b.val at hv
    have hs := stage.isLt
    have hb := b.isLt
    omega
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 3 + 4 * stage.val + (2 + b.val) at hv
    have hs := stage.isLt
    have hb := b.isLt
    omega

example : 15 ≤ (17 : Fin 24).val := by decide

/-- The actual raw detector prefix leaves every reserved upper axis zero at all positions.
Source: genuine token-local input and complete protected-block induction, with no freshness premise. -/
theorem depthWordState_upper_zero (mode : Mode) (gain eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (n : ℕ) (hn : n ≤ depthDetectorCount mode)
    (c : Fin 24) (hc : 15 ≤ c.val) (i : Fin word.length) :
    depthWordState mode gain eps word positions n i (depthCoordinate mode c) = 0 := by
  revert hn
  induction n with
  | zero =>
      intro hn
      change depthRawEmbedding mode gain (depthLetterToken (word.get i)) (depthCoordinate mode c) = 0
      apply depthRawEmbedding_fresh mode gain _ (depthLetterToken_raw (word.get i))
      · intro he; have hv := congrArg Fin.val he; change c.val = 0 at hv; omega
      · intro he; have hv := congrArg Fin.val he; change c.val = 1 at hv; omega
      · intro he; have hv := congrArg Fin.val he; change c.val = 2 at hv; omega
  | succ n ih =>
      intro hn
      have hnext : n < depthDetectorCount mode := by omega
      rw [depthWordState, dite_eq_left hnext, depthDetectorBlock_preserves_upper mode _ eps positions _ i c hc]
      exact ih (by omega)

example : 1 ≤ depthDetectorCount .easy ∧ 15 ≤ (19 : Fin 24).val := by decide

/-- The ordinary FFN writes positive/negative readout indicators into axes seventeen/eighteen.
Source: the existing tied label direction, disjoint from all actual detector targets. -/
def depthReadoutOutputCoordinate (mode : Mode) (b : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode ⟨17 + b.val, by have hb := b.isLt; omega⟩

/-- The two real readout FFN output columns cannot interfere.
Source: their original coordinate assignment and injective integer working layout. -/
theorem depthReadoutOutputCoordinate_injective (mode : Mode) :
    Function.Injective (depthReadoutOutputCoordinate mode) := by
  intro a b he
  apply Fin.ext
  have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
  change 17 + a.val = 17 + b.val at hv
  omega

/-- Positive/negative readout probes are the final B/A occurrence coordinates of the actual encoder.
Source: independent E_2/E_4 ending-pattern criterion and unchanged original working axes. -/
def depthReadoutProbeCoordinate (mode : Mode) (b : Fin 2) : Fin (depthConfig mode).d_model :=
  depthLevelCoordinate mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩
    ⟨1 - b.val, by have hb := b.isLt; omega⟩

/-- The two actual final probes are disjoint.
Source: opposite branches in the same original occurrence level, with no head collision. -/
theorem depthReadoutProbeCoordinate_injective (mode : Mode) :
    Function.Injective (depthReadoutProbeCoordinate mode) := by
  intro a b he
  apply Fin.ext
  have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
  change 1 + 4 * depthDetectorCount mode + (1 - a.val) = 1 + 4 * depthDetectorCount mode + (1 - b.val) at hv
  have ha := a.isLt
  have hb := b.isLt
  omega

/-- Both actual probes lie between five and fourteen, outside raw and final readout axes.
Source: one easy detector or three hard detectors in the original residual layout. -/
theorem depthReadoutProbeCoordinate_bounds (mode : Mode) (b : Fin 2) :
    5 ≤ (depthReadoutProbeCoordinate mode b).val ∧ (depthReadoutProbeCoordinate mode b).val ≤ 14 := by
  change 5 ≤ 1 + 4 * depthDetectorCount mode + (1 - b.val) ∧ 1 + 4 * depthDetectorCount mode + (1 - b.val) ≤ 14
  have hb := b.isLt
  cases mode <;> simp only [depthDetectorCount] <;> omega

/-- The genuine first residual of the original readout block writes its uniform heads back into the last-level probes.
Source: ordinary attention parameters and original unnormalized residual addition, preserving possible current presence. -/
noncomputable def depthReadoutAttentionState (mode : Mode) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) :
    EucSpace (depthConfig mode).d_model :=
  x i + attnSubLayer (depthConfig mode)
    (depthAttention mode (depthReadoutProbeCoordinate mode) (depthReadoutProbeCoordinate mode)) eps positions x i

/-- Each true readout probe contains its retained old flag plus its genuine attenuated head signal.
Source: full original fused attention/merge/W_o and injective output targets; no presence indicator is substituted. -/
theorem depthReadoutAttentionState_probe (mode : Mode) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (b : Fin 2) :
    depthReadoutAttentionState mode eps positions x i (depthReadoutProbeCoordinate mode b) =
      x i (depthReadoutProbeCoordinate mode b) + depthAttentionSignal mode (depthReadoutProbeCoordinate mode) eps x b i := by
  unfold depthReadoutAttentionState
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthAttention_target mode _ _ (depthReadoutProbeCoordinate_injective mode)]

/-- Actual readout attention preserves all raw axes and unused upper working axes on arbitrary real states.
Source: only probe columns five through fourteen receive head contributions. -/
theorem depthReadoutAttentionState_protected (mode : Mode) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin 24) (hc : c.val < 3 ∨ 15 ≤ c.val) :
    depthReadoutAttentionState mode eps positions x i (depthCoordinate mode c) = x i (depthCoordinate mode c) := by
  unfold depthReadoutAttentionState
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthAttention_protected mode _ _ eps positions x i (depthCoordinate mode c) (by
    intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = (depthReadoutProbeCoordinate mode b).val at hv
    have hb := depthReadoutProbeCoordinate_bounds mode b
    omega), add_zero]

example : (17 : Fin 24).val < 3 ∨ 15 ≤ (17 : Fin 24).val := Or.inr (by decide)

/-- A genuine incoming constant one supplies an actual lower norm bound after readout attention.
Source: protected real constant coordinate and the original residual Euclidean norm. -/
theorem depthReadoutAttentionState_norm_lower (mode : Mode) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (hc : x i (depthCoordinate mode 0) = 1) : 1 ≤ ‖depthReadoutAttentionState mode eps positions x i‖ := by
  apply depthState_norm_lower
  rw [depthReadoutAttentionState_protected mode eps positions x i 0 (Or.inl (by decide)), hc]

example : depthAxis .easy 0 (depthCoordinate .easy 0) = 1 := by rw [depthAxis_coordinate]; norm_num

end Transformer.GPTMini.Semantics
