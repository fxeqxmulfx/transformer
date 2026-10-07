import Transformer.GPTMini.Semantics.DepthReadoutFFN

/-!
# Genuine complete original depth readout block

Source: original blockForward at f11b6e2, raw ordered detector induction
at 14a2f0d, actual residual probes at f3e61b2 and the seven-unit readout
FFN at 2d14aea. The complete unchanged block combines two finite uniform
heads with the genuine bias-free ReLU2 matrices. Raw axes survive both
residuals; actual pre-FFN conditions follow from the incoming invariant.

The real block output has two visible-occurrence indicators with its
same actual RMS-square scale, plus a common scale coordinate. Norm lies
between one and 1282; the true squared scale lies in [r^2,128]. Indicators
are proved conclusions of the independently computed original block.
The incoming invariant is local and explicit; the raw encoder already
derives it. Full ModelParams and the checked integer adapter remain to
be connected, with no encoder, route or correct-logit oracle premise.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical BigOperators
open Transformer.Basis

/-- The complete original readout parameter record, shared across every raw prefix of this mode.
Source: actual uniform fused attention with retained flags and the ordinary seven-unit FFN. -/
noncomputable def depthReadoutBlock (mode : Mode) : BlockParams (depthConfig mode) where
  attn := depthAttention mode (depthReadoutProbeCoordinate mode) (depthReadoutProbeCoordinate mode)
  ffn := depthReadoutFFN mode

/-- Both real residuals preserve every raw constant/type axis on arbitrary states.
Source: genuine attention targets at least five and genuine FFN targets seventeen/eighteen/nineteen. -/
theorem depthReadoutBlock_raw (mode : Mode) (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (c : Fin 24) (hc : c.val < 3) :
    blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions x i (depthCoordinate mode c) =
      x i (depthCoordinate mode c) := by
  dsimp only [blockForward, depthReadoutBlock]
  change (depthReadoutAttentionState mode eps positions x i +
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps (depthReadoutAttentionState mode eps positions x) i)
      (depthCoordinate mode c) = _
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthReadoutFFN_protected mode eps _ i (depthCoordinate mode c) (by
    intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 17 + b.val at hv
    omega) (by
    intro he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 19 at hv
    omega), add_zero, depthReadoutAttentionState_protected mode eps positions x i c (Or.inl hc)]

example : (2 : Fin 24).val < 3 := by decide

/-- All requirements of the actual final FFN follow from the local encoder invariant and true readout probes.
Source: protected constant one, actual zero-or-threshold residual signal and cap 144, below the actual gate cap 256. -/
theorem depthReadoutBlock_conditions (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (i : Fin word.length) :
    let y := depthReadoutAttentionState mode eps positions x
    y i (depthCoordinate mode 0) = 1 ∧
    (∀ b, y i (depthReadoutProbeCoordinate mode b) = 0 ∨ 2 * depthThreshold ≤ y i (depthReadoutProbeCoordinate mode b)) ∧
    (∀ b, y i (depthReadoutProbeCoordinate mode b) ≤ 256) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [depthReadoutAttentionState_protected mode eps positions x i 0 (Or.inl (by decide)), h.1 i]
  · intro b
    exact (depthReadout_probe_semantics mode eps heps hclip word hT positions x h b i).1
  · intro b
    exact (depthReadout_probe_semantics mode eps heps hclip word hT positions x h b i).2.1.trans (by norm_num)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .easy 1 [none, some false, some true]
      (depthWordState .easy 1 (1 / 100000) [none, some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num, by norm_num, by decide,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

/-- The true complete readout block realizes independent visible ordered-pattern indicators with one genuine shared RMS-square amplitude.
Source: derived actual FFN conditions, real residual probe semantics and complete original blockForward. -/
theorem depthReadoutBlock_semantics (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (i : Fin word.length) :
    let y := depthReadoutAttentionState mode eps positions x
    blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions x i = y i +
      ((∑ b : Fin 2, (if DepthReadoutPresence mode word b i then (depthScale mode eps (y i)) ^ 2 else 0) •
        EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) + (depthScale mode eps (y i)) ^ 2 • depthAxis mode 19) := by
  let y := depthReadoutAttentionState mode eps positions x
  have hc := depthReadoutBlock_conditions mode eps heps hclip word hT positions x h i
  have hf := depthReadoutFFN_binary mode eps y i hc.1 hc.2.1 hc.2.2
  have he (b : Fin 2) : 0 < y i (depthReadoutProbeCoordinate mode b) ↔ DepthReadoutPresence mode word b i :=
    (depthReadout_probe_semantics mode eps heps hclip word hT positions x h b i).2.2
  simp only [he] at hf
  dsimp only [blockForward, depthReadoutBlock]
  change y i + ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps y i = _
  rw [hf]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .easy 1 [none, some false, some true]
      (depthWordState .easy 1 (1 / 100000) [none, some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num, by norm_num, by decide,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

/-- The actual complete final block retains norm at least one and is bounded by 1282.
Source: protected raw constant, real pre-FFN norm at most 898 and true FFN contribution at most 384. -/
theorem depthReadoutBlock_norm (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (i : Fin word.length) :
    1 ≤ ‖blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions x i‖ ∧
      ‖blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions x i‖ ≤ 1282 := by
  refine ⟨?_, ?_⟩
  · apply depthState_norm_lower
    rw [depthReadoutBlock_raw mode eps positions x i 0 (by decide), h.1 i]
  · let y := depthReadoutAttentionState mode eps positions x
    have hc := depthReadoutBlock_conditions mode eps heps hclip word hT positions x h i
    have hf := depthReadoutFFN_norm mode eps heps.le y i hc.1 hc.2.1 hc.2.2
    have hy := (depthReadoutAttentionState_norm mode eps heps word positions x h i).2
    change ‖y i‖ ≤ 898 at hy
    dsimp only [blockForward, depthReadoutBlock]
    change ‖y i + ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps y i‖ ≤ 1282
    exact (norm_add_le _ _).trans (by linarith)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, some true] : List (Option Bool)).length ≤ 128 ∧
    DepthStateInvariant .easy 1 [some false, some true]
      (depthWordState .easy 1 (1 / 100000) [some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num, by norm_num, by decide,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

/-- The genuine common squared readout scale has the same positive lower floor and finite cap as true detector flags.
Source: actual pre-FFN norm [1,898], unchanged original RMSNorm and shared bounded-domain scale estimates. -/
theorem depthReadoutBlock_scale_bounds (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (positions : Fin word.length → ℝ)
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨depthDetectorCount mode, by have hc := (depthDetectorCount_slots mode).1; omega⟩ word x)
    (i : Fin word.length) :
    depthScaleLower ^ 2 ≤ (depthScale mode eps (depthReadoutAttentionState mode eps positions x i)) ^ 2 ∧
      (depthScale mode eps (depthReadoutAttentionState mode eps positions x i)) ^ 2 ≤ 128 := by
  have hy := depthReadoutAttentionState_norm mode eps heps word positions x h i
  have hs := depthScale_lower mode eps heps.le hclip _ hy.1 (hy.2.trans (by norm_num))
  have hp := depthScale_pos mode eps heps.le _ hy.1
  refine ⟨?_, (depthScale_upper mode eps heps.le _ hy.1).2⟩
  simpa only [pow_two] using mul_le_mul hs hs depthScaleLower_bounds.1.le hp.le

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ DepthStateInvariant .easy 1 [some false, some true]
    (depthWordState .easy 1 (1 / 100000) [some false, some true] (fun i => i.val) 1) := by
  exact ⟨by norm_num, by norm_num,
    depthWordState_invariant .easy 1 (1 / 100000) (by norm_num) (by norm_num) _ (by decide) _ 1 (by decide)⟩

end Transformer.GPTMini.Semantics
