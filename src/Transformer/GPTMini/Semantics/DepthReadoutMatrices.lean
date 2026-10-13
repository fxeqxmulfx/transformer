import Transformer.GPTMini.Semantics.DepthReadoutPresence

/-!
# Seven-unit ordinary original depth readout matrices

Source: original bias-free ReLU2_FFN at f11b6e2 and the simultaneous
six-row depthDetectorIn/depthDetectorOut construction. Both readout
presence gates use the protected constant as their type coordinate.
One additional, disjoint ordinary hidden unit reads that same constant
and writes its squared ReLU activation into readout axis nineteen.

These are genuine shared W_in/W_out matrices in the unchanged original
FFN types, fitting both width configurations. Every assigned input row
is evaluated and the complete actual FFN formula is derived on arbitrary
real inputs. Subsequent prenorm and presence theorems must derive the
common positive RMS-square amplitude and the two semantic indicators;
neither indicators nor desired labels define this network.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical BigOperators
open Transformer.Basis

/-- The original six-unit presence matrices with both type rows equal to the protected constant.
Source: actual readout probes and original output axes seventeen/eighteen. -/
noncomputable def depthReadoutBaseFFN (mode : Mode) : FFNParams (depthConfig mode) :=
  depthPresenceFFN mode (depthReadoutProbeCoordinate mode) (fun _ => depthCoordinate mode 0)
    (depthReadoutOutputCoordinate mode)

/-- The same six original hidden-row indices, explicitly shared with the readout matrices.
Source: three finite-difference units per branch in the original unchanged FFN width. -/
noncomputable def depthReadoutHiddenIndex (mode : Mode) (b : Fin 2) (r : Fin 3) : Fin (depthConfig mode).d_ff :=
  depthDetectorIndex (by have h := depthConfig_ffn mode; omega) b r

/-- One additional original FFN unit, disjoint from all six detector rows.
Source: seventh unit within the original 256/512 hidden coordinates. -/
def depthReadoutCommonIndex (mode : Mode) : Fin (depthConfig mode).d_ff :=
  ⟨6, by have h := depthConfig_ffn mode; omega⟩

/-- The common-scale unit cannot change any actual detector preactivation.
Source: assigned detector rows zero through five and common row six. -/
theorem depthReadoutCommonIndex_ne (mode : Mode) (b : Fin 2) (r : Fin 3) :
    depthReadoutCommonIndex mode ≠ depthReadoutHiddenIndex mode b r := by
  intro he
  have hv := congrArg (fun c : Fin (depthConfig mode).d_ff => c.val) he
  change 6 = 3 * b.val + r.val at hv
  have hb := b.isLt
  have hr := r.isLt
  omega

/-- The actual input matrix adds one constant-reading row to the original six-unit detector.
Source: a single ordinary rank-one matrix term, with no bias or extra operator. -/
noncomputable def depthReadoutIn (mode : Mode) : EucSpace (depthConfig mode).d_model →L[ℝ] EucSpace (depthConfig mode).d_ff :=
  (depthReadoutBaseFFN mode).W_in + (EuclideanSpace.proj (depthCoordinate mode 0)).smulRight
    (EuclideanSpace.single (depthReadoutCommonIndex mode) 1)

/-- The actual output matrix adds the common hidden activation to readout axis nineteen.
Source: original W_out type and the tied-label common-scale coordinate. -/
noncomputable def depthReadoutOut (mode : Mode) : EucSpace (depthConfig mode).d_ff →L[ℝ] EucSpace (depthConfig mode).d_model :=
  (depthReadoutBaseFFN mode).W_out + (EuclideanSpace.proj (depthReadoutCommonIndex mode)).smulRight (depthAxis mode 19)

/-- Every real detector preactivation is unchanged by the actual seventh-row insertion.
Source: original six-row formula and disjoint common hidden index. -/
theorem depthReadoutIn_detector (mode : Mode) (x : EucSpace (depthConfig mode).d_model) (b : Fin 2) (r : Fin 3) :
    depthReadoutIn mode x (depthReadoutHiddenIndex mode b r) =
      depthDetectorForm (depthCoordinate mode 0) (depthReadoutProbeCoordinate mode b) (depthCoordinate mode 0) 256 x -
        (r.val : ℝ) * depthThreshold * x (depthCoordinate mode 0) := by
  unfold depthReadoutIn
  simp only [add_apply, WithLp.ofLp_add, Pi.add_apply, ContinuousLinearMap.smulRight_apply,
    EuclideanSpace.proj, PiLp.proj_apply, PiLp.smul_apply, PiLp.single_apply, smul_eq_mul]
  have hn : depthReadoutHiddenIndex mode b r ≠ depthReadoutCommonIndex mode :=
    fun he => depthReadoutCommonIndex_ne mode b r he.symm
  simp only [ite_eq_right hn, mul_zero, add_zero]
  exact depthDetectorIn_coordinate (by have h := depthConfig_ffn mode; omega) (depthCoordinate mode 0)
    (depthReadoutProbeCoordinate mode) (fun _ => depthCoordinate mode 0) depthThreshold 256 x b r

/-- The genuine extra preactivation is exactly the same protected raw constant.
Source: unused original detector row six has zero contribution, and the inserted rank-one row reads the constant. -/
theorem depthReadoutIn_common (mode : Mode) (x : EucSpace (depthConfig mode).d_model) :
    depthReadoutIn mode x (depthReadoutCommonIndex mode) = x (depthCoordinate mode 0) := by
  have hz : (depthReadoutBaseFFN mode).W_in x (depthReadoutCommonIndex mode) = 0 := by
    unfold depthReadoutBaseFFN depthPresenceFFN depthDetectorIn
    simp only [sum_apply, ContinuousLinearMap.smulRight_apply, WithLp.ofLp_sum, Finset.sum_apply,
      PiLp.smul_apply, PiLp.single_apply, smul_eq_mul]
    apply Finset.sum_eq_zero
    intro b hb
    apply Finset.sum_eq_zero
    intro r hr
    change _ * (if depthReadoutCommonIndex mode = depthReadoutHiddenIndex mode b r then 1 else 0) = 0
    rw [ite_eq_right (depthReadoutCommonIndex_ne mode b r), mul_zero]
  unfold depthReadoutIn
  simp only [add_apply, WithLp.ofLp_add, Pi.add_apply, ContinuousLinearMap.smulRight_apply,
    EuclideanSpace.proj, PiLp.proj_apply, PiLp.smul_apply, PiLp.single_apply, ite_true, smul_eq_mul, hz, mul_one, zero_add]

/-- Complete ordinary readout FFN parameters, using exactly seven assigned original units.
Source: the genuine simultaneous input/output matrices above, without state- or label-dependent parameters. -/
noncomputable def depthReadoutFFN (mode : Mode) : FFNParams (depthConfig mode) where
  W_in := depthReadoutIn mode
  W_out := depthReadoutOut mode

/-- The true seven-unit original FFN has the continuous two-gate formula plus the real squared-ReLU common activation.
Source: actual matrix rows, original componentwise ReLU2 and original output matrix; valid for arbitrary real vectors. -/
theorem depthReadoutFFN_apply (mode : Mode) (x : EucSpace (depthConfig mode).d_model) :
    relu2FFN (depthReadoutIn mode) (depthReadoutOut mode) x =
      (∑ b : Fin 2, (depthDetectorGain * depthTypeStep depthThreshold 256
        (x (depthReadoutProbeCoordinate mode b)) (x (depthCoordinate mode 0)) (x (depthCoordinate mode 0))) •
          EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) +
      relu2 (x (depthCoordinate mode 0)) • depthAxis mode 19 := by
  have hd : (depthReadoutBaseFFN mode).W_out (relu2Vec (depthReadoutIn mode x)) =
      ∑ b : Fin 2, (depthDetectorGain * depthTypeStep depthThreshold 256
        (x (depthReadoutProbeCoordinate mode b)) (x (depthCoordinate mode 0)) (x (depthCoordinate mode 0))) •
          EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1 := by
    unfold depthReadoutBaseFFN depthPresenceFFN depthDetectorOut
    simp only [sum_apply, ContinuousLinearMap.smulRight_apply, smul_apply, EuclideanSpace.proj,
      PiLp.proj_apply, smul_eq_mul, relu2Vec_apply, smul_smul]
    apply Finset.sum_congr rfl
    intro b hb
    change (∑ r : Fin 3, (depthHingeCoefficient r * relu2 (depthReadoutIn mode x (depthReadoutHiddenIndex mode b r)) *
      depthDetectorGain) • EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) = _
    simp only [Fin.sum_univ_three, depthHingeCoefficient]
    rw [depthReadoutIn_detector mode x b 0, depthReadoutIn_detector mode x b 1, depthReadoutIn_detector mode x b 2]
    norm_num
    rw [← neg_smul, ← add_smul, ← add_smul]
    unfold depthTypeStep depthStep
    rw [depthDetectorForm_apply]
    congr 1
    ring_nf
  unfold relu2FFN depthReadoutOut
  rw [add_apply, hd]
  simp only [ContinuousLinearMap.smulRight_apply, EuclideanSpace.proj, PiLp.proj_apply,
    relu2Vec_apply]
  rw [depthReadoutIn_common]

/-- The actual prenormed readout FFN has one common true RMS-square scale for both continuous gates and its constant unit.
Source: the evaluated seven-unit original matrices, genuine RMSNorm and positive quadratic homogeneity. -/
theorem depthReadoutFFN_rms (mode : Mode) (eps : ℝ) (x : EucSpace (depthConfig mode).d_model)
    (hc : x (depthCoordinate mode 0) = 1) :
    relu2FFN (depthReadoutIn mode) (depthReadoutOut mode) (rmsNormEps eps x) =
      (∑ b : Fin 2, (depthDetectorGain * (depthScale mode eps x) ^ 2 * depthTypeStep depthThreshold 256
        (x (depthReadoutProbeCoordinate mode b)) 1 1) • EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) +
      (depthScale mode eps x) ^ 2 • depthAxis mode 19 := by
  have hs : 0 ≤ depthScale mode eps x := by
    unfold depthScale
    exact div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  rw [depthReadoutFFN_apply, depthScale_rms]
  simp only [PiLp.smul_apply, smul_eq_mul, hc, mul_one]
  have hr : relu2 (depthScale mode eps x) = (depthScale mode eps x) ^ 2 := by
    unfold relu2
    rw [max_eq_right hs]
  rw [hr]
  congr 1
  apply Finset.sum_congr rfl
  intro b hb
  have ht := depthTypeStep_scale depthThreshold 256 (x (depthReadoutProbeCoordinate mode b)) 1 1 (depthScale mode eps x) hs
  simp only [mul_one] at ht
  rw [ht]
  congr 1
  ring

example : depthAxis .easy 0 (depthCoordinate .easy 0) = 1 := by rw [depthAxis_coordinate]; norm_num

end Transformer.GPTMini.Semantics
