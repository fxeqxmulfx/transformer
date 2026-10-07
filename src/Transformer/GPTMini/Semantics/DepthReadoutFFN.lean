import Transformer.GPTMini.Semantics.DepthReadoutMatrices

/-!
# Actual depth readout FFN amplitudes, protection and bounds

Source: original ReLU2_FFN at f11b6e2 and the genuine seven-unit matrices
at a705ed2. The seventh unit adds the actual pre-FFN RMS square in axis
nineteen. The other six rows are the original two presence gates with
both type coordinates equal to the protected constant.

On genuine separated probes, the true complete FFN writes zero or that
same RMS square in axes seventeen/eighteen, protects all other axes and
has contribution norm at most 384. Local numerical conditions are
explicit, with simultaneous real-vector witnesses; the actual encoder
and readout attention already derive them from raw words. Semantic and
complete-model coupling are subsequent steps, not hidden assumptions
in the definition of these actual matrices or outputs.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical BigOperators
open Transformer.Basis

/-- The genuine complete readout FFN is the original six-unit sublayer plus its actual common RMS-square coordinate.
Source: evaluated seven-unit/prenorm formula at the same true constant-one input. -/
theorem depthReadoutFFN_extra (mode : Mode) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (hc : x i (depthCoordinate mode 0) = 1) :
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps x i =
      ffnSubLayer (depthConfig mode) (depthReadoutBaseFFN mode) eps x i +
        (depthScale mode eps (x i)) ^ 2 • depthAxis mode 19 := by
  change relu2FFN (depthReadoutIn mode) (depthReadoutOut mode) (rmsNormEps eps (x i)) = _
  rw [depthReadoutFFN_rms mode eps (x i) hc]
  unfold depthReadoutBaseFFN
  rw [depthPresenceFFN_raw]
  simp only [hc]

example : depthAxis .easy 0 (depthCoordinate .easy 0) = 1 := by rw [depthAxis_coordinate]; norm_num

/-- The true constant unit's output is exactly the common actual pre-FFN RMS square.
Source: six detector output columns seventeen/eighteen are disjoint from common column nineteen. -/
theorem depthReadoutFFN_common (mode : Mode) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (hc : x i (depthCoordinate mode 0) = 1) :
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps x i (depthCoordinate mode 19) =
      (depthScale mode eps (x i)) ^ 2 := by
  rw [depthReadoutFFN_extra mode eps x i hc]
  simp only [WithLp.ofLp_add, Pi.add_apply, PiLp.smul_apply, smul_eq_mul]
  unfold depthReadoutBaseFFN
  rw [depthPresenceFFN_protected mode (depthReadoutProbeCoordinate mode) (fun _ => depthCoordinate mode 0)
    (depthReadoutOutputCoordinate mode) eps x i (depthCoordinate mode 19) (by
    intro b he
    have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
    change 19 = 17 + b.val at hv
    have hb := b.isLt
    omega), depthAxis_coordinate]
  norm_num

example : depthAxis .hard 0 (depthCoordinate .hard 0) = 1 := by rw [depthAxis_coordinate]; norm_num

/-- Only the three genuine readout output coordinates can receive a complete FFN contribution on arbitrary states.
Source: full original matrix/activation formula; no constant, gap, norm or correct-output premise is needed. -/
theorem depthReadoutFFN_protected (mode : Mode) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (c : Fin (depthConfig mode).d_model)
    (ho : ∀ b, c ≠ depthReadoutOutputCoordinate mode b) (hc : c ≠ depthCoordinate mode 19) :
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps x i c = 0 := by
  change relu2FFN (depthReadoutIn mode) (depthReadoutOut mode) (rmsNormEps eps (x i)) c = 0
  rw [depthReadoutFFN_apply]
  simp only [WithLp.ofLp_add, Pi.add_apply, WithLp.ofLp_sum, Finset.sum_apply,
    PiLp.smul_apply, PiLp.single_apply, smul_eq_mul, depthAxis]
  rw [ite_eq_right hc, mul_zero, add_zero]
  apply Finset.sum_eq_zero
  intro b hb
  rw [ite_eq_right (ho b), mul_zero]

example : (∀ b, depthCoordinate .easy 0 ≠ depthReadoutOutputCoordinate .easy b) ∧
    depthCoordinate .easy 0 ≠ depthCoordinate .easy 19 := by
  constructor
  · intro b he
    have hv := congrArg (fun c : Fin (depthConfig .easy).d_model => c.val) he
    change 0 = 17 + b.val at hv
    omega
  · intro he
    have hv := congrArg (fun c : Fin (depthConfig .easy).d_model => c.val) he
    change 0 = 19 at hv
    omega

/-- A concrete actual constant vector simultaneously satisfies the readout's constant and zero-probe requirements.
Source: protected unit axis zero and genuine probe axes five through fourteen. -/
theorem depthReadoutFFN_control (mode : Mode) :
    depthAxis mode 0 (depthCoordinate mode 0) = 1 ∧
      ∀ b, depthAxis mode 0 (depthReadoutProbeCoordinate mode b) = 0 := by
  refine ⟨by rw [depthAxis_coordinate]; norm_num, ?_⟩
  intro b
  unfold depthAxis
  rw [PiLp.single_apply, ite_eq_right]
  intro he
  have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
  have hb := depthReadoutProbeCoordinate_bounds mode b
  change (depthReadoutProbeCoordinate mode b).val = 0 at hv
  omega

/-- Genuine separated probes make the actual seven-unit sublayer write zero or its same common RMS-square amplitude.
Source: original six-unit binary theorem plus the evaluated actual common unit; indicators occur only in the conclusion. -/
theorem depthReadoutFFN_binary (mode : Mode) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (hc : x i (depthCoordinate mode 0) = 1)
    (hs : ∀ b, x i (depthReadoutProbeCoordinate mode b) = 0 ∨ 2 * depthThreshold ≤ x i (depthReadoutProbeCoordinate mode b))
    (hu : ∀ b, x i (depthReadoutProbeCoordinate mode b) ≤ 256) :
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps x i =
      (∑ b : Fin 2, (if 0 < x i (depthReadoutProbeCoordinate mode b) then (depthScale mode eps (x i)) ^ 2 else 0) •
        EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) + (depthScale mode eps (x i)) ^ 2 • depthAxis mode 19 := by
  rw [depthReadoutFFN_extra mode eps x i hc]
  unfold depthReadoutBaseFFN
  rw [depthPresenceFFN_binary mode _ _ _ eps x i hc (fun _ => Or.inr hc) (fun b _ => hs b) hu]
  simp only [hc, true_and]

example : depthAxis .easy 0 (depthCoordinate .easy 0) = 1 ∧
    (∀ b, depthAxis .easy 0 (depthReadoutProbeCoordinate .easy b) = 0 ∨
      2 * depthThreshold ≤ depthAxis .easy 0 (depthReadoutProbeCoordinate .easy b)) ∧
    (∀ b, depthAxis .easy 0 (depthReadoutProbeCoordinate .easy b) ≤ 256) := by
  have h := depthReadoutFFN_control .easy
  exact ⟨h.1, fun b => Or.inl (h.2 b), fun b => by rw [h.2 b]; norm_num⟩

/-- The actual complete readout FFN contribution has norm at most 384 on the true separated-probe domain.
Source: genuine six-unit norm at most 256 and constant-unit squared RMS at most 128. -/
theorem depthReadoutFFN_norm (mode : Mode) (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (hc : x i (depthCoordinate mode 0) = 1)
    (hs : ∀ b, x i (depthReadoutProbeCoordinate mode b) = 0 ∨ 2 * depthThreshold ≤ x i (depthReadoutProbeCoordinate mode b))
    (hu : ∀ b, x i (depthReadoutProbeCoordinate mode b) ≤ 256) :
    ‖ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps x i‖ ≤ 384 := by
  have hd := depthPresenceFFN_norm mode (depthReadoutProbeCoordinate mode) (fun _ => depthCoordinate mode 0)
    (depthReadoutOutputCoordinate mode) eps heps x i hc (fun _ => Or.inr hc) (fun b _ => hs b) hu
  change ‖ffnSubLayer (depthConfig mode) (depthReadoutBaseFFN mode) eps x i‖ ≤ 256 at hd
  have hu := (depthScale_upper mode eps heps (x i) (depthState_norm_lower mode (x i) hc)).2
  have hsquare : ‖(depthScale mode eps (x i)) ^ 2 • depthAxis mode 19‖ ≤ 128 := by
    rw [norm_smul, depthAxis_norm, Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), mul_one]
    exact hu
  rw [depthReadoutFFN_extra mode eps x i hc]
  exact (norm_add_le _ _).trans (by linarith)

example : (0 : ℝ) ≤ 1 / 100000 ∧ depthAxis .hard 0 (depthCoordinate .hard 0) = 1 ∧
    (∀ b, depthAxis .hard 0 (depthReadoutProbeCoordinate .hard b) = 0 ∨
      2 * depthThreshold ≤ depthAxis .hard 0 (depthReadoutProbeCoordinate .hard b)) ∧
    (∀ b, depthAxis .hard 0 (depthReadoutProbeCoordinate .hard b) ≤ 256) := by
  have h := depthReadoutFFN_control .hard
  exact ⟨by norm_num, h.1, fun b => Or.inl (h.2 b), fun b => by rw [h.2 b]; norm_num⟩

/-- On an actual constant-only control, all six presence units vanish and the true seventh unit supplies the common scale.
Source: the complete original binary matrix theorem and evaluated control probes. -/
theorem depthReadoutFFN_control_output (mode : Mode) (eps : ℝ) :
    ffnSubLayer (depthConfig mode) (depthReadoutFFN mode) eps (fun _ : Fin 1 => depthAxis mode 0) 0 =
      (depthScale mode eps (depthAxis mode 0)) ^ 2 • depthAxis mode 19 := by
  have h := depthReadoutFFN_control mode
  rw [depthReadoutFFN_binary mode eps (fun _ : Fin 1 => depthAxis mode 0) 0 h.1
    (fun b => Or.inl (h.2 b)) (fun b => by rw [h.2 b]; norm_num)]
  simp only [h.2, lt_irrefl, ite_false, zero_smul, Finset.sum_const_zero, zero_add]

end Transformer.GPTMini.Semantics
