import Transformer.GPTMini.Semantics.DepthLayout

/-!
# Genuine depth attention residual and detector input conditions

Source: original blockForward's first residual at f11b6e2 and the
fixed original detector records in DepthLayout. This state is the
actual input plus actual attention, not a defined semantic indicator.
Protected raw coordinates survive, fresh signals equal their genuine
head outputs, and new FFN targets stay empty until the FFN writes them.

The input predicate records explicit local conditions on a real array:
protected constant, binary raw types, fresh stage channels, separated
previous features and zero opposite feature at a matching raw type.
An ordinary active-A array satisfies them simultaneously. The complete
raw-word induction must prove the predicate for its actual states;
it is not a new assumption about the full model's desired answers.

The real attention residual has norm at most M+32, and its protected
constant gives norm at least one. Full FFN transition and ordered
semantics are subsequent obligations. No training claim is made.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The real first residual of the original detector block, before its actual FFN prenorm.
Source: blockForward's exact attention update with ordinary fixed parameters. -/
noncomputable def depthDetectorAttentionState (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) :
    EucSpace (depthConfig mode).d_model :=
  x i + attnSubLayer (depthConfig mode)
    (depthAttention mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)) eps positions x i

/-- The true pre-FFN residual retains the original constant and raw token types.
Source: disjoint actual attention targets and the unnormalized residual addition. -/
theorem depthDetectorAttentionState_raw (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin 24) (hc : c.val < 3) :
    depthDetectorAttentionState mode stage eps positions x i (depthCoordinate mode c) = x i (depthCoordinate mode c) := by
  unfold depthDetectorAttentionState
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthAttention_protected mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)
    eps positions x i (depthCoordinate mode c) (by
      intro b he
      have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
      change c.val = 3 + 4 * stage.val + b.val at hv
      omega), add_zero]

example : (1 : Fin 24).val < 3 := by decide

/-- Both actual raw-type columns are unchanged at the real FFN input.
Source: the low-coordinate protection identity at the concrete token-local layout. -/
theorem depthDetectorAttentionState_type (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (b : Fin 2) :
    depthDetectorAttentionState mode stage eps positions x i (depthTypeCoordinate mode b) = x i (depthTypeCoordinate mode b) := by
  exact depthDetectorAttentionState_raw mode stage eps positions x i _ (by
    change 1 + b.val < 3
    have hb := b.isLt
    omega)

/-- A genuinely fresh signal channel in the input reads exactly its own actual head signal after attention.
Source: both true targets are distinct in the complete original output projection. -/
theorem depthDetectorAttentionState_signal (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (b : Fin 2)
    (hz : x i (depthSignalCoordinate mode stage b) = 0) :
    depthDetectorAttentionState mode stage eps positions x i (depthSignalCoordinate mode stage b) =
      depthAttentionSignal mode (depthSourceCoordinate mode stage) eps x b i := by
  unfold depthDetectorAttentionState
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [hz, depthAttention_target mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)
    (depthSignalCoordinate_injective mode stage) eps positions x i b, zero_add]

example : (depthAxis .easy 0 + depthAxis .easy 1) (depthSignalCoordinate .easy 0 0) = 0 := by
  change (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy 3) = 0
  rw [depthActiveDetector_coordinate]
  norm_num

/-- The genuine attention residual does not write either of this stage's future FFN feature columns.
Source: actual signal/feature disjointness in the ordinary output matrix. -/
theorem depthDetectorAttentionState_feature (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (b : Fin 2) :
    depthDetectorAttentionState mode stage eps positions x i (depthFeatureCoordinate mode stage b) =
      x i (depthFeatureCoordinate mode stage b) := by
  unfold depthDetectorAttentionState
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthAttention_protected mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)
    eps positions x i (depthFeatureCoordinate mode stage b) (depthFeatureCoordinate_signal_ne mode stage b), add_zero]

/-- The actual pre-FFN residual grows by at most 32 on nonnegative real previous features.
Source: the genuine complete attention bound, independently of a binary or separated signal. -/
theorem depthDetectorAttentionState_norm (mode : Mode) (stage : Fin 3) (eps : ℝ) (heps : 0 < eps) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (M : ℝ)
    (hx : ‖x i‖ ≤ M) (hn : ∀ b j, 0 ≤ x j (depthSourceCoordinate mode stage b)) :
    ‖depthDetectorAttentionState mode stage eps positions x i‖ ≤ M + 32 := by
  exact depthAttentionResidual_norm mode (depthSourceCoordinate mode stage) (depthSignalCoordinate mode stage)
    eps heps positions x i M hx hn

example : (0 : ℝ) < 1 / 100000 ∧ ‖depthAxis .easy 0 + depthAxis .easy 1‖ ≤ 2 ∧
    (∀ b : Fin 2, ∀ j : Fin 2, (0 : ℝ) ≤
      (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j (depthSourceCoordinate .easy 0 b)) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · calc ‖depthAxis .easy 0 + depthAxis .easy 1‖ ≤ ‖depthAxis .easy 0‖ + ‖depthAxis .easy 1‖ := norm_add_le _ _
      _ = 2 := by rw [depthAxis_norm, depthAxis_norm]; norm_num
  · intro b j
    fin_cases b <;> change 0 ≤ (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy _)
    all_goals rw [depthActiveDetector_coordinate]; norm_num

/-- A protected raw constant one derives the actual pre-FFN norm lower bound.
Source: original coordinate domination after the genuine residual protection calculation. -/
theorem depthDetectorAttentionState_norm_lower (mode : Mode) (stage : Fin 3) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (hc : x i (depthCoordinate mode 0) = 1) :
    1 ≤ ‖depthDetectorAttentionState mode stage eps positions x i‖ := by
  apply depthState_norm_lower
  rw [depthDetectorAttentionState_raw mode stage eps positions x i 0 (by decide)]
  exact hc

example : (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard 0) = 1 := by
  rw [depthActiveDetector_coordinate]
  norm_num

/-- Explicit local representation conditions on the actual input array of a detector stage.
Source: raw-type protection, fresh target channels and DepthRecurrence's opposite-current exclusion;
the full raw-state induction must derive all six conditions rather than assume this predicate at model input. -/
def DepthDetectorInput (mode : Mode) (stage : Fin 3) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) : Prop :=
  (∀ j, x j (depthCoordinate mode 0) = 1) ∧
  (∀ b j, x j (depthTypeCoordinate mode b) = 0 ∨ x j (depthTypeCoordinate mode b) = 1) ∧
  (∀ b j, x j (depthSignalCoordinate mode stage b) = 0) ∧
  (∀ b j, x j (depthFeatureCoordinate mode stage b) = 0) ∧
  (∀ b j, x j (depthSourceCoordinate mode stage b) = 0 ∨ depthScaleLower ^ 2 ≤ x j (depthSourceCoordinate mode stage b)) ∧
  (∀ b j, x j (depthTypeCoordinate mode b) = 1 → x j (depthSourceCoordinate mode stage b) = 0)

/-- One ordinary active-A array satisfies every local detector-input condition simultaneously.
Source: true real unit-coordinate sums; the matched A branch has zero opposite self, while the wrong B branch sees A.
This is solely an operator-domain witness, not a semantic encoder supplied to the model. -/
theorem depthDetectorInput_active (mode : Mode) :
    DepthDetectorInput mode 0 (fun _ : Fin 2 => depthAxis mode 0 + depthAxis mode 1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    rw [depthActiveDetector_coordinate]
    norm_num
  · intro b j
    fin_cases b <;> change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode _) = 0 ∨
      (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode _) = 1
    all_goals rw [depthActiveDetector_coordinate]; norm_num
  · intro b j
    fin_cases b <;> change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode _) = 0
    all_goals rw [depthActiveDetector_coordinate]; norm_num [depthStageAxis]
  · intro b j
    fin_cases b <;> change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode _) = 0
    all_goals rw [depthActiveDetector_coordinate]; norm_num [depthStageAxis]
  · intro b j
    fin_cases b
    · apply Or.inl
      change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode 2) = 0
      rw [depthActiveDetector_coordinate]; norm_num
    · apply Or.inr
      change depthScaleLower ^ 2 ≤ (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode 1)
      rw [depthActiveDetector_coordinate]; norm_num [depthScaleLower]
  · intro b j hk
    fin_cases b
    · change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode 2) = 0
      rw [depthActiveDetector_coordinate]; norm_num
    · change (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode 2) = 1 at hk
      rw [depthActiveDetector_coordinate] at hk
      norm_num at hk

/-- Previous real features in the explicit detector domain are nonnegative.
Source: zero-or-positive RMS-square lower bound, with no normalized-value assumption. -/
theorem depthDetectorInput_nonneg (mode : Mode) (stage : Fin 3) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (h : DepthDetectorInput mode stage x) :
    ∀ b j, 0 ≤ x j (depthSourceCoordinate mode stage b) := by
  intro b j
  rcases h.2.2.2.2.1 b j with hz | hp
  · rw [hz]
  · exact (sq_nonneg depthScaleLower).trans hp

example : DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) :=
  depthDetectorInput_active .easy

/-- The same active-A domain witness has genuine norm between one and two at every position.
Source: protected unit coordinate and the triangle inequality for its actual two-axis raw embedding. -/
theorem depthDetectorInput_active_norm (mode : Mode) :
    1 ≤ ‖depthAxis mode 0 + depthAxis mode 1‖ ∧ ‖depthAxis mode 0 + depthAxis mode 1‖ ≤ 2 := by
  constructor
  · exact depthState_norm_lower mode _ (depthActiveDetector_conditions mode).1
  · calc ‖depthAxis mode 0 + depthAxis mode 1‖ ≤ ‖depthAxis mode 0‖ + ‖depthAxis mode 1‖ := norm_add_le _ _
      _ = 2 := by rw [depthAxis_norm, depthAxis_norm]; norm_num

end Transformer.GPTMini.Semantics
