import Transformer.GPTMini.Semantics.DepthDetector

/-!
# Fixed original depth detector blocks and their working layout

Source: original GPTMini blockForward at f11b6e2, raw Basis E_2/E_4
semantics at cbafbe9 and DepthRecurrence's opposite-ending transition.
Each of three possible detector stages uses two original heads and six
ordinary FFN units. Its inputs are the previous opposite-ending flags;
its attention signals and new flags occupy fresh residual coordinates.

The layout uses axes 0 through 14, within the unchanged 64/128 widths.
Axis zero and the raw A/B axes one/two are protected through every real
detector block. Unwritten other stages are also protected. Parameters
are shared over positions and do not inspect a word or desired answer.

These are actual BlockParams records and actual blockForward identities.
Signal gaps, feature meanings and bounded state induction remain proof
obligations; no oracle is used to define a block or its residual stream.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- A genuine raw A/B type axis, with branch zero for A and one for B.
Source: depthRawEmbedding's original token-local axes one and two. -/
def depthTypeCoordinate (mode : Mode) (branch : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode ⟨1 + branch.val, by have hb := branch.isLt; omega⟩

/-- Four disjoint working channels per possible detector stage.
Source: two real attention signals followed by two actual FFN outputs, not additional model width. -/
def depthStageAxis (stage : Fin 3) (part : Fin 4) : Fin 24 :=
  ⟨3 + 4 * stage.val + part.val, by have hs := stage.isLt; have hp := part.isLt; omega⟩

/-- Distinct stage/part assignments do not interfere in the original residual stream.
Source: the explicit four-channel integer layout. -/
theorem depthStageAxis_eq (stage other : Fin 3) (part next : Fin 4) :
    depthStageAxis stage part = depthStageAxis other next ↔ stage = other ∧ part = next := by
  constructor
  · intro he
    have hv := congrArg (fun c : Fin 24 => c.val) he
    change 3 + 4 * stage.val + part.val = 3 + 4 * other.val + next.val at hv
    have hp := part.isLt
    have hn := next.isLt
    exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- The two actual attention signal targets for this detector stage.
Source: parts zero/one of the disjoint residual assignment. -/
def depthSignalCoordinate (mode : Mode) (stage : Fin 3) (branch : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode (depthStageAxis stage ⟨branch.val, by have hb := branch.isLt; omega⟩)

/-- The two fresh ordinary FFN flag targets for this detector stage.
Source: parts two/three of the same actual residual assignment. -/
def depthFeatureCoordinate (mode : Mode) (stage : Fin 3) (branch : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode (depthStageAxis stage ⟨2 + branch.val, by have hb := branch.isLt; omega⟩)

/-- The genuine previous opposite-ending coordinate read by each of the two heads.
Source: DepthRecurrence's alternating successor; stage zero reads the opposite raw type directly. -/
def depthSourceCoordinate (mode : Mode) (stage : Fin 3) (branch : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode ⟨2 + 4 * stage.val - branch.val, by have hs := stage.isLt; omega⟩

/-- The actual first detector reads the opposite raw token's embedding coordinate.
Source: the source-row assignment at stage zero. -/
theorem depthSourceCoordinate_initial (mode : Mode) (branch : Fin 2) :
    depthSourceCoordinate mode 0 branch =
      depthTypeCoordinate mode ⟨1 - branch.val, by have hb := branch.isLt; omega⟩ := by
  apply congrArg (depthCoordinate mode)
  apply Fin.ext
  change 2 + 4 * (0 : Fin 3).val - branch.val = 1 + (1 - branch.val)
  have hb := branch.isLt
  norm_num
  omega

/-- Every later real detector reads the previous stage's opposite-ending feature.
Source: the same fixed fused-QKV row, coupled to the prior genuine FFN target. -/
theorem depthSourceCoordinate_successor (mode : Mode) (stage : Fin 2) (branch : Fin 2) :
    depthSourceCoordinate mode ⟨stage.val + 1, by have hs := stage.isLt; omega⟩ branch =
      depthFeatureCoordinate mode ⟨stage.val, by have hs := stage.isLt; omega⟩
        ⟨1 - branch.val, by have hb := branch.isLt; omega⟩ := by
  apply congrArg (depthCoordinate mode)
  apply Fin.ext
  change 2 + 4 * (stage.val + 1) - branch.val = 3 + 4 * stage.val + (2 + (1 - branch.val))
  have hb := branch.isLt
  omega

/-- The two simultaneous actual signal targets are distinct.
Source: stage-axis injection and the original residual-coordinate embedding. -/
theorem depthSignalCoordinate_injective (mode : Mode) (stage : Fin 3) :
    Function.Injective (depthSignalCoordinate mode stage) := by
  intro a b he
  have hp := (depthStageAxis_eq stage stage _ _).mp (depthCoordinate_injective mode he)
  exact Fin.ext (congrArg (fun c : Fin 4 => c.val) hp.2)

/-- The two simultaneous actual FFN feature targets are distinct.
Source: the same disjoint stage layout, so actual output columns have no cross-branch leakage. -/
theorem depthFeatureCoordinate_injective (mode : Mode) (stage : Fin 3) :
    Function.Injective (depthFeatureCoordinate mode stage) := by
  intro a b he
  have hp := (depthStageAxis_eq stage stage _ _).mp (depthCoordinate_injective mode he)
  have hv := congrArg (fun c : Fin 4 => c.val) hp.2
  apply Fin.ext
  change 2 + a.val = 2 + b.val at hv
  omega

/-- The complete ordinary original detector block, without a new operator or hidden state definition.
Source: actual simultaneous fused attention and six-unit FFN records in the fixed layout. -/
noncomputable def depthDetectorBlock (mode : Mode) (stage : Fin 3) : BlockParams (depthConfig mode) where
  attn := depthAttention mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)
  ffn := depthPresenceFFN mode (depthSignalCoordinate mode stage) (depthTypeCoordinate mode)
    (depthFeatureCoordinate mode stage)

/-- Every actual residual coordinate outside this block's signal/feature columns is preserved on arbitrary states.
Source: both genuine protected sublayer contributions and original two-residual blockForward. -/
theorem depthDetectorBlock_protected (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin (depthConfig mode).d_model) (hs : ∀ b, c ≠ depthSignalCoordinate mode stage b)
    (hf : ∀ b, c ≠ depthFeatureCoordinate mode stage b) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i c = x i c := by
  dsimp only [blockForward, depthDetectorBlock]
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthAttention_protected _ _ _ _ _ _ _ _ hs,
    depthPresenceFFN_protected mode (depthSignalCoordinate mode stage) (depthTypeCoordinate mode)
      (depthFeatureCoordinate mode stage) eps _ i c hf]
  simp only [add_zero]

example : (∀ b, depthCoordinate .easy 0 ≠ depthSignalCoordinate .easy 0 b) ∧
    (∀ b, depthCoordinate .easy 0 ≠ depthFeatureCoordinate .easy 0 b) := by
  constructor
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig .easy).d_model => c.val) he
    change 0 = 3 + 4 * (0 : Fin 3).val + b.val at hv
    omega
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig .easy).d_model => c.val) he
    change 0 = 3 + 4 * (0 : Fin 3).val + (2 + b.val) at hv
    omega

/-- The protected constant and both raw token types survive every complete real detector stage.
Source: axes below three are disjoint from all actual block outputs. -/
theorem depthDetectorBlock_preserves_raw (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin 24) (hc : c.val < 3) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthCoordinate mode c) = x i (depthCoordinate mode c) := by
  apply depthDetectorBlock_protected
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 3 + 4 * stage.val + b.val at hv
    omega
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change c.val = 3 + 4 * stage.val + (2 + b.val) at hv
    omega

example : (0 : Fin 24).val < 3 := by decide

/-- Actual feature targets are disjoint from both actual attention signal columns of their own stage.
Source: two signal parts followed by two feature parts in the unchanged residual layout. -/
theorem depthFeatureCoordinate_signal_ne (mode : Mode) (stage : Fin 3) (a b : Fin 2) :
    depthFeatureCoordinate mode stage a ≠ depthSignalCoordinate mode stage b := by
  intro he
  have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
  change 3 + 4 * stage.val + (2 + a.val) = 3 + 4 * stage.val + b.val at hv
  have hb := b.isLt
  omega

/-- All genuine signal/feature coordinates of another stage survive this complete original block.
Source: stage/part injectivity and both real residual protection identities. -/
theorem depthDetectorBlock_preserves_stage (mode : Mode) (stage other : Fin 3) (hne : other ≠ stage)
    (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model)
    (i : Fin T) (part : Fin 4) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthCoordinate mode (depthStageAxis other part)) = x i (depthCoordinate mode (depthStageAxis other part)) := by
  apply depthDetectorBlock_protected
  · intro b he
    exact hne ((depthStageAxis_eq other stage _ _).mp (depthCoordinate_injective mode he)).1
  · intro b he
    exact hne ((depthStageAxis_eq other stage _ _).mp (depthCoordinate_injective mode he)).1

example : (1 : Fin 3) ≠ 0 := by decide

end Transformer.GPTMini.Semantics
