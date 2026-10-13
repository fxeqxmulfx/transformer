import Transformer.GPTMini.Semantics.DepthResidual

/-!
# Complete real detector transitions from explicit input conditions

Source: original blockForward/ffnSubLayer at f11b6e2, the unchanged
six-unit FFN and the actual head separation proved in DepthSignalPresence.
All constant/type/gap/cap requirements are now derived at the true
pre-FFN residual from explicit conditions on the incoming real array.
The gap is derived only at a matching raw type, where its genuine
opposite self-feature is zero. Wrong types need only the true cap.

The complete original block writes zero or its actual pre-FFN RMS
square into each fresh flag. Its norm grows by at most 288, including
both residuals. The raw-state induction must still establish all
incoming conditions from the validated word and identify these flags
with the independent ordered-occurrence semantics. No correct logits,
semantic encoder, optimizer success or convexity is assumed or claimed.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators Classical
open Transformer.Basis

/-- Every true local FFN requirement follows from actual incoming coordinates and norm bounds.
Source: genuine residual protection, actual head gap at zero opposite self and the RMS-derived signal cap sixteen. -/
theorem depthDetector_preFFN_conditions (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4096) (i : Fin T) :
    depthDetectorAttentionState mode stage eps positions x i (depthCoordinate mode 0) = 1 ∧
    (∀ b, depthDetectorAttentionState mode stage eps positions x i (depthTypeCoordinate mode b) = 0 ∨
      depthDetectorAttentionState mode stage eps positions x i (depthTypeCoordinate mode b) = 1) ∧
    (∀ b, depthDetectorAttentionState mode stage eps positions x i (depthTypeCoordinate mode b) = 1 →
      depthDetectorAttentionState mode stage eps positions x i (depthSignalCoordinate mode stage b) = 0 ∨
      2 * depthThreshold ≤ depthDetectorAttentionState mode stage eps positions x i (depthSignalCoordinate mode stage b)) ∧
    (∀ b, depthDetectorAttentionState mode stage eps positions x i (depthSignalCoordinate mode stage b) ≤ 256) := by
  have hbound (j : Fin T) : 1 ≤ ‖x j‖ ∧ ‖x j‖ ≤ 4096 :=
    ⟨depthState_norm_lower mode (x j) (hinput.1 j), hnorm j⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [depthDetectorAttentionState_raw mode stage eps positions x i 0 (by decide)]
    exact hinput.1 i
  · intro b
    rw [depthDetectorAttentionState_type]
    exact hinput.2.1 b i
  · intro b hk
    rw [depthDetectorAttentionState_type] at hk
    rw [depthDetectorAttentionState_signal mode stage eps positions x i b (hinput.2.2.1 b i)]
    exact depthAttentionSignal_separated mode (depthSourceCoordinate mode stage) eps heps.le hclip hT x b i
      hbound (hinput.2.2.2.2.1 b) (hinput.2.2.2.2.2 b i hk)
  · intro b
    rw [depthDetectorAttentionState_signal mode stage eps positions x i b (hinput.2.2.1 b i)]
    exact (depthAttentionSignal_bounds mode (depthSourceCoordinate mode stage) eps heps x b i
      (depthDetectorInput_nonneg mode stage x hinput b)).2.trans (by norm_num)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4096) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)

/-- The actual full block writes each separated presence flag with its own true position-dependent RMS square.
Source: complete real FFN formula, distinct actual output columns and genuine fresh residual channels.
The Boolean expression is derived as the conclusion, never used to define the model's state. -/
theorem depthDetectorBlock_feature (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4096) (i : Fin T) (b : Fin 2) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) =
      if x i (depthTypeCoordinate mode b) = 1 ∧
        0 < depthAttentionSignal mode (depthSourceCoordinate mode stage) eps x b i
      then (depthScale mode eps (depthDetectorAttentionState mode stage eps positions x i)) ^ 2 else 0 := by
  let y := depthDetectorAttentionState mode stage eps positions x
  have hp := depthDetector_preFFN_conditions mode stage eps heps hclip hT positions x hinput hnorm i
  have hf := depthPresenceFFN_binary mode (depthSignalCoordinate mode stage) (depthTypeCoordinate mode)
    (depthFeatureCoordinate mode stage) eps y i hp.1 hp.2.1 hp.2.2.1 hp.2.2.2
  dsimp only [blockForward, depthDetectorBlock]
  change (y i + ffnSubLayer (depthConfig mode) (depthPresenceFFN mode (depthSignalCoordinate mode stage)
    (depthTypeCoordinate mode) (depthFeatureCoordinate mode stage)) eps y i) (depthFeatureCoordinate mode stage b) = _
  simp only [WithLp.ofLp_add, Pi.add_apply]
  rw [depthDetectorAttentionState_feature mode stage eps positions x i b, hinput.2.2.2.1 b i, hf, zero_add]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, PiLp.single_apply, smul_eq_mul]
  rw [Fintype.sum_eq_single b]
  · simp only [ite_true, mul_one]
    rw [depthDetectorAttentionState_type mode stage eps positions x i b,
      depthDetectorAttentionState_signal mode stage eps positions x i b (hinput.2.2.1 b i)]
  · intro other ho
    have hne : depthFeatureCoordinate mode stage b ≠ depthFeatureCoordinate mode stage other :=
      fun he => ho ((depthFeatureCoordinate_injective mode stage) he).symm
    rw [ite_eq_right hne, mul_zero]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .hard 0 (fun _ : Fin 2 => depthAxis .hard 0 + depthAxis .hard 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .hard 0 + depthAxis .hard 1) j‖ ≤ 4096) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .hard, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .hard).2.trans (by norm_num)

/-- The actual written flag is positive exactly when the genuine type/presence detector succeeds.
Source: the full real block formula and the strictly positive true RMS multiplier at the protected pre-FFN constant.
No Boolean state is supplied as an input to the ordinary block. -/
theorem depthDetectorBlock_feature_positive_iff (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4096) (i : Fin T) (b : Fin 2) :
    0 < blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) ↔ x i (depthTypeCoordinate mode b) = 1 ∧
        0 < depthAttentionSignal mode (depthSourceCoordinate mode stage) eps x b i := by
  have hscale := depthScale_pos mode eps heps.le (depthDetectorAttentionState mode stage eps positions x i)
    (depthDetectorAttentionState_norm_lower mode stage eps positions x i (hinput.1 i))
  rw [depthDetectorBlock_feature mode stage eps heps hclip hT positions x hinput hnorm i b]
  by_cases hp : x i (depthTypeCoordinate mode b) = 1 ∧
      0 < depthAttentionSignal mode (depthSourceCoordinate mode stage) eps x b i
  · rw [ite_eq_left hp]
    constructor
    · intro
      exact hp
    · intro
      exact sq_pos_of_pos hscale
  · rw [ite_eq_right hp]
    constructor
    · intro hz
      exact False.elim ((lt_irrefl 0) hz)
    · intro hz
      exact False.elim (hp hz)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4096) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)

/-- One complete original detector block grows the actual incoming norm by at most 288.
Source: real attention norm at most 32 plus real two-branch FFN norm at most 256, with all FFN conditions derived.
The bounded incoming domain is explicit and must be discharged by raw-state induction. -/
theorem depthDetectorBlock_norm (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (M : ℝ) (hM : M ≤ 4096) (hnorm : ∀ j, ‖x j‖ ≤ M) (i : Fin T) :
    ‖blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i‖ ≤ M + 288 := by
  let y := depthDetectorAttentionState mode stage eps positions x
  have hp := depthDetector_preFFN_conditions mode stage eps heps hclip hT positions x hinput (fun j => (hnorm j).trans hM) i
  have ha := depthDetectorAttentionState_norm mode stage eps heps positions x i M (hnorm i)
    (depthDetectorInput_nonneg mode stage x hinput)
  have hf := depthPresenceFFN_norm mode (depthSignalCoordinate mode stage) (depthTypeCoordinate mode)
    (depthFeatureCoordinate mode stage) eps heps.le y i hp.1 hp.2.1 hp.2.2.1 hp.2.2.2
  dsimp only [blockForward, depthDetectorBlock]
  change ‖y i + ffnSubLayer (depthConfig mode) (depthPresenceFFN mode (depthSignalCoordinate mode stage)
    (depthTypeCoordinate mode) (depthFeatureCoordinate mode stage)) eps y i‖ ≤ M + 288
  calc ‖y i + ffnSubLayer (depthConfig mode) (depthPresenceFFN mode (depthSignalCoordinate mode stage)
        (depthTypeCoordinate mode) (depthFeatureCoordinate mode stage)) eps y i‖
      ≤ ‖y i‖ + ‖ffnSubLayer (depthConfig mode) (depthPresenceFFN mode (depthSignalCoordinate mode stage)
        (depthTypeCoordinate mode) (depthFeatureCoordinate mode stage)) eps y i‖ := norm_add_le _ _
    _ ≤ M + 288 := by linarith

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (2 : ℝ) ≤ 4096 ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 2) := by
  exact ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy,
    by norm_num, fun j => (depthDetectorInput_active_norm .easy).2⟩

end Transformer.GPTMini.Semantics
