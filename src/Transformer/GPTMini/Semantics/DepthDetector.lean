import Transformer.GPTMini.Semantics.DepthSignalPresence

/-!
# Complete original depth detector FFN with shared finite weights

Source: original ffnSubLayer/W_in/ReLU2/W_out at f11b6e2 and
DepthMatrices' genuine simultaneous six-unit construction. The
shared threshold and gain from DepthNormalization produce exactly
zero or the true squared RMS multiplier on matching-type presence
inputs. No bias, extra unit or input-dependent parameter is added.

The raw constant, binary local type, matching-type signal gap and cap
remain explicit local hypotheses. Actual depth attention now derives
the signal gap and cap; raw/ordered block induction must establish
the whole simultaneous representation at its real pre-FFN residual.
Wrong types need only the cap, even if XSA attenuates their signal.

Protected output coordinates receive exactly zero even outside that
representation domain. In the domain and at nonnegative epsilon,
the actual two-branch FFN contribution has norm at most 256. These
are original-operator computations, not a training or convexity claim.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators Classical
open Transformer.Basis

/-- A complete ordinary original FFN parameter record, using six of the unchanged hidden units.
Source: DepthMatrices' actual shared matrices at threshold r^3/256, cap 256 and finite plateau gain. -/
noncomputable def depthPresenceFFN (mode : Mode) (signal kind output : Fin 2 → Fin (depthConfig mode).d_model) :
    FFNParams (depthConfig mode) where
  W_in := depthDetectorIn (by have h := depthConfig_ffn mode; omega)
    (depthCoordinate mode 0) signal kind depthThreshold 256
  W_out := depthDetectorOut (by have h := depthConfig_ffn mode; omega) output depthDetectorGain

/-- A genuine protected constant one gives residual norm at least one in the original model.
Source: actual coordinate norm domination, needed for subsequent genuine RMS amplitude bounds. -/
theorem depthState_norm_lower (mode : Mode) (x : EucSpace (depthConfig mode).d_model)
    (hc : x (depthCoordinate mode 0) = 1) : 1 ≤ ‖x‖ := by
  have h := PiLp.norm_apply_le x (depthCoordinate mode 0)
  rw [hc, Real.norm_eq_abs, abs_one] at h
  exact h

example : depthAxis .easy 0 (depthCoordinate .easy 0) = 1 := by rw [depthAxis_coordinate]; norm_num

/-- The ordinary raw A embedding has only its protected constant and local A type coordinates.
Source: depthRawEmbedding's real axis sum; this is an active-type control for the local detector domain. -/
theorem depthActiveDetector_coordinate (mode : Mode) (c : Fin 24) :
    (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode c) =
      (if c = 0 then 1 else 0) + (if c = 1 then 1 else 0) := by
  change depthAxis mode 0 (depthCoordinate mode c) + depthAxis mode 1 (depthCoordinate mode c) = _
  rw [depthAxis_coordinate, depthAxis_coordinate]

/-- A real active-A residual simultaneously satisfies the constant, binary raw-type and zero initial-signal conditions.
Source: the genuine raw embedding control; the matching branch is present, so the domain is not wrong-type only. -/
theorem depthActiveDetector_conditions (mode : Mode) :
    (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode 0) = 1 ∧
    (∀ b : Fin 2, (depthAxis mode 0 + depthAxis mode 1)
      (depthCoordinate mode (if b.val = 0 then 1 else 2)) = 0 ∨
      (depthAxis mode 0 + depthAxis mode 1) (depthCoordinate mode (if b.val = 0 then 1 else 2)) = 1) ∧
    (∀ b : Fin 2, (depthAxis mode 0 + depthAxis mode 1)
      (depthCoordinate mode (if b.val = 0 then 3 else 4)) = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [depthActiveDetector_coordinate]
    norm_num
  · intro b
    by_cases hb : b.val = 0
    · rw [ite_eq_left hb, depthActiveDetector_coordinate]
      norm_num
    · rw [ite_eq_right hb, depthActiveDetector_coordinate]
      norm_num
  · intro b
    by_cases hb : b.val = 0
    · rw [ite_eq_left hb, depthActiveDetector_coordinate]
      norm_num
    · rw [ite_eq_right hb, depthActiveDetector_coordinate]
      norm_num

/-- The actual full FFN sublayer has the genuine continuous gate formula with its real position-dependent RMS square.
Source: original prenorm followed by the simultaneous actual six-unit matrices. -/
theorem depthPresenceFFN_raw (mode : Mode) (signal kind output : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) :
    ffnSubLayer (depthConfig mode) (depthPresenceFFN mode signal kind output) eps x i =
      ∑ branch : Fin 2, (depthDetectorGain * (depthScale mode eps (x i)) ^ 2 *
        depthTypeStep depthThreshold 256 (x i (signal branch)) (x i (kind branch)) (x i (depthCoordinate mode 0))) •
        EuclideanSpace.single (output branch) 1 := by
  unfold ffnSubLayer depthPresenceFFN
  rw [depthDetectorFFN_rms]
  rfl

/-- With a gap only on matching types, the actual original FFN contributes exactly zero or its same true squared RMS amplitude.
Source: actual matrix/prenorm formula and proved three-hinge type/presence plateau; the Boolean expression is only the conclusion. -/
theorem depthPresenceFFN_binary (mode : Mode) (signal kind output : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (hc : x i (depthCoordinate mode 0) = 1) (hk : ∀ b, x i (kind b) = 0 ∨ x i (kind b) = 1)
    (hs : ∀ b, x i (kind b) = 1 → x i (signal b) = 0 ∨ 2 * depthThreshold ≤ x i (signal b))
    (hu : ∀ b, x i (signal b) ≤ 256) :
    ffnSubLayer (depthConfig mode) (depthPresenceFFN mode signal kind output) eps x i =
      ∑ branch : Fin 2, (if x i (kind branch) = 1 ∧ 0 < x i (signal branch)
        then (depthScale mode eps (x i)) ^ 2 else 0) • EuclideanSpace.single (output branch) 1 := by
  rw [depthPresenceFFN_raw]
  apply Finset.sum_congr rfl
  intro branch hb
  rw [hc]
  rcases hk branch with hz | hone
  · rw [hz, depthTypeStep_wrong_type depthThreshold 256 (x i (signal branch)) 1
      depthThreshold_bounds.1.le (by norm_num) (by simpa only [mul_one] using hu branch)]
    norm_num
  · have hg := depthTypeStep_binary depthThreshold 256 (x i (signal branch)) (x i (kind branch)) 1
      depthThreshold_bounds.1 (by norm_num) (Or.inr hone)
      (by simpa only [mul_one] using hs branch hone) (by simpa only [mul_one] using hu branch)
    rw [hg]
    split_ifs
    · congr 1
      calc depthDetectorGain * (depthScale mode eps (x i)) ^ 2 * (2 * (depthThreshold * 1) ^ 2) =
          (depthDetectorGain * (2 * depthThreshold ^ 2)) * (depthScale mode eps (x i)) ^ 2 := by ring
        _ = (depthScale mode eps (x i)) ^ 2 := by rw [depthDetectorGain_plateau.2, one_mul]
    · simp only [mul_zero]

example : (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy 0) = 1 ∧
    (∀ b : Fin 2, (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 1 else 2)) = 0 ∨
      (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 1 else 2)) = 1) ∧
    (∀ b : Fin 2, (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 1 else 2)) = 1 →
      (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 3 else 4)) = 0 ∨
      2 * depthThreshold ≤ (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 3 else 4))) ∧
    (∀ b : Fin 2, (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy (if b.val = 0 then 3 else 4)) ≤ 256) := by
  have h := depthActiveDetector_conditions .easy
  refine ⟨h.1, h.2.1, fun b _ => Or.inl (h.2.2 b), ?_⟩
  intro b
  rw [h.2.2 b]
  norm_num

/-- Every actual FFN contribution outside the two assigned output coordinates is zero on arbitrary real states.
Source: genuine shared output matrix, without a binary-feature or signal-gap premise. -/
theorem depthPresenceFFN_protected (mode : Mode) (signal kind output : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (c : Fin (depthConfig mode).d_model) (hprotected : ∀ b, c ≠ output b) :
    ffnSubLayer (depthConfig mode) (depthPresenceFFN mode signal kind output) eps x i c = 0 := by
  unfold ffnSubLayer depthPresenceFFN
  exact depthDetectorFFN_protected _ _ signal kind output depthThreshold 256 depthDetectorGain (rmsNormEps eps (x i)) c hprotected

example : ∀ b : Fin 2, (0 : Fin 64) ≠ (if b = 0 then 5 else 6) := by intro b; split_ifs <;> decide

/-- The genuine simultaneous detector FFN has norm at most 256 on its real separated representation domain.
Source: each actual output amplitude is zero or squared RMS at most 128; output targets need not be distinct. -/
theorem depthPresenceFFN_norm (mode : Mode) (signal kind output : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T)
    (hc : x i (depthCoordinate mode 0) = 1) (hk : ∀ b, x i (kind b) = 0 ∨ x i (kind b) = 1)
    (hs : ∀ b, x i (kind b) = 1 → x i (signal b) = 0 ∨ 2 * depthThreshold ≤ x i (signal b))
    (hu : ∀ b, x i (signal b) ≤ 256) :
    ‖ffnSubLayer (depthConfig mode) (depthPresenceFFN mode signal kind output) eps x i‖ ≤ 256 := by
  have hscale := (depthScale_upper mode eps heps (x i) (depthState_norm_lower mode (x i) hc)).2
  rw [depthPresenceFFN_binary mode signal kind output eps x i hc hk hs hu]
  calc ‖∑ b : Fin 2, (if x i (kind b) = 1 ∧ 0 < x i (signal b) then (depthScale mode eps (x i)) ^ 2 else 0) •
        EuclideanSpace.single (output b) (1 : ℝ)‖
      ≤ ∑ b : Fin 2, ‖(if x i (kind b) = 1 ∧ 0 < x i (signal b) then (depthScale mode eps (x i)) ^ 2 else 0) •
        EuclideanSpace.single (output b) (1 : ℝ)‖ := norm_sum_le _ _
    _ ≤ ∑ b : Fin 2, (128 : ℝ) := by
      apply Finset.sum_le_sum
      intro b hb
      split_ifs
      · simp only [norm_smul, PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one,
          abs_of_nonneg (sq_nonneg (depthScale mode eps (x i)))]
        exact hscale
      · simp only [zero_smul, norm_zero]
        norm_num
    _ = 256 := by rw [Fin.sum_univ_two]; norm_num

example : (0 : ℝ) ≤ 1 / 100000 ∧ (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard 0) = 1 ∧
    (∀ b : Fin 2, (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 1 else 2)) = 0 ∨
      (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 1 else 2)) = 1) ∧
    (∀ b : Fin 2, (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 1 else 2)) = 1 →
      (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 3 else 4)) = 0 ∨
      2 * depthThreshold ≤ (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 3 else 4))) ∧
    (∀ b : Fin 2, (depthAxis .hard 0 + depthAxis .hard 1) (depthCoordinate .hard (if b.val = 0 then 3 else 4)) ≤ 256) := by
  have h := depthActiveDetector_conditions .hard
  refine ⟨by norm_num, h.1, h.2.1, fun b _ => Or.inl (h.2.2 b), ?_⟩
  intro b
  rw [h.2.2 b]
  norm_num

end Transformer.GPTMini.Semantics
