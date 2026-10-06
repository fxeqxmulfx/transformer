import Transformer.GPTMini.Semantics.CountFFN

/-!
# A simultaneous parity and completion decoder in the original FFN

Source: ReLU2_FFN.forward at f11b6e2 and Parity.solve's answer/EOS phases
at cbafbe9. The 68 proved parity units are retained. One additional unit
reads the negative protected phase coordinate: it is inactive at SEP and
active at a supplied answer. Its ordinary output column writes to an
independent EOS residual direction. Both signals coexist in one FFN with
the original 256 hidden units and 64 residual coordinates.

The result below evaluates the actual linear input matrix, coordinatewise
ReLU2 and linear output matrix. The two scales are ordinary finite output
weights. Tied embeddings and sufficient readout margins must still be
verified; a nonzero EOS channel alone does not prove a final token answer.
The real-arithmetic and shared-epsilon conventions remain unchanged.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- EOS uses a fifth residual coordinate, independently of ONE, BOS, phase and parity.
Source: the new completion decoder in the unchanged 64-wide model. -/
noncomputable def eosUnit : EucSpace 64 := EuclideanSpace.single 4 1

/-- One spare FFN unit reads negative completion phase, while all count hinges retain their inputs.
Source: the original W_in operator type, with exactly 69 assigned activation coordinates. -/
noncomputable def completionFFNIn : EucSpace 64 →L[ℝ] EucSpace countConfig.d_ff :=
  parityFFNIn + (-phaseProbe).smulRight (EuclideanSpace.single ⟨68, by decide⟩ 1)

/-- Ordinary output weights scale the parity signal and the independent EOS signal.
Source: W_out at f11b6e2; no task function or discrete label is called by this matrix. -/
noncomputable def completionFFNOut (parityScale eosScale : ℝ) :
    EucSpace countConfig.d_ff →L[ℝ] EucSpace 64 :=
  parityScale • parityFFNOut +
    (innerSL ℝ (EuclideanSpace.single ⟨68, by decide⟩ 1)).smulRight (eosScale • eosUnit)

/-- The original FFN parameter record with both decoder channels.
Source: FFNParams at f11b6e2, without adding layers, biases or an alternate optimizer. -/
noncomputable def completionFFN (parityScale eosScale : ℝ) : FFNParams countConfig where
  W_in := completionFFNIn
  W_out := completionFFNOut parityScale eosScale

/-- The additional phase column cannot alter any actual parity preactivation.
Source: the proved disjointness of the 68 hinge coordinates and coordinate 68. -/
theorem completionFFNIn_hinge (x : EucSpace 64) (k : Fin 17) (r : Fin 4) :
    (completionFFNIn x).ofLp (bumpIndex k r) = (parityFFNIn x).ofLp (bumpIndex k r) := by
  simp only [completionFFNIn, add_apply, ContinuousLinearMap.smulRight_apply,
    WithLp.ofLp_add, Pi.add_apply, WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply,
    ite_eq_right (bumpIndex_ne_spare k r), smul_zero, add_zero]

/-- The spare unit reads exactly the negative phase, since the parity matrix is zero there.
Source: the actual extra input column and parityFFNIn_spare. -/
theorem completionFFNIn_phase (x : EucSpace 64) :
    (completionFFNIn x).ofLp ⟨68, by decide⟩ = -phaseProbe x := by
  simp only [completionFFNIn, add_apply, ContinuousLinearMap.smulRight_apply,
    WithLp.ofLp_add, Pi.add_apply, WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply,
    ite_true, smul_eq_mul, mul_one, neg_apply]
  have hz : (parityFFNIn x).ofLp ⟨68, by decide⟩ = 0 := parityFFNIn_spare x
  rw [hz, zero_add]

/-- The actual parity output matrix is unaffected by activation of the separate completion unit.
Source: its 68 distinct output probes; this checks simultaneous realizability in one FFN. -/
theorem completion_preserves_parity (x : EucSpace 64) :
    parityFFNOut (relu2Vec (completionFFNIn x)) = parityFFNOut (relu2Vec (parityFFNIn x)) := by
  unfold parityFFNOut
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, relu2Vec_apply]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro r _
  exact congrArg (fun t : ℝ => relu2 t • ((countSign k.val * bumpCoeff r) • parityUnit))
    (completionFFNIn_hinge x k r)

/-- The unchanged FFN evaluates to both the proved count decoder and the actual phase hinge.
Source: its explicit input/output matrices and original relu2FFN, not a desired-output premise. -/
theorem completionFFN_apply (parityScale eosScale : ℝ) (x : EucSpace 64) :
    relu2FFN completionFFNIn (completionFFNOut parityScale eosScale) x =
      (parityScale * paritySpline (inner (𝕜 := ℝ) TokenInterface.controlUnit x)
        (inner (𝕜 := ℝ) bosUnit x)) • parityUnit +
      (eosScale * relu2 (-phaseProbe x)) • eosUnit := by
  unfold relu2FFN completionFFNOut
  rw [add_apply, smul_apply, completion_preserves_parity]
  have hp := parityFFN_apply x
  change parityFFNOut (relu2Vec (parityFFNIn x)) = _ at hp
  rw [hp]
  simp only [ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, relu2Vec_apply,
    smul_smul]
  congr 1
  calc
    _ = (relu2 (-phaseProbe x) * eosScale) • eosUnit :=
      congrArg (fun t : ℝ => (relu2 t * eosScale) • eosUnit) (completionFFNIn_phase x)
    _ = _ := by congr 1; ring

/-- Prompt phase +1 leaves the true completion unit inactive.
Source: the SEP phase in ratioEmbedding, using the actual nonpositive ReLU2 branch. -/
theorem completionFFN_prompt_phase (parityScale eosScale : ℝ) (x : EucSpace 64)
    (hphase : 0 ≤ phaseProbe x) :
    relu2FFN completionFFNIn (completionFFNOut parityScale eosScale) x =
      (parityScale * paritySpline (inner (𝕜 := ℝ) TokenInterface.controlUnit x)
        (inner (𝕜 := ℝ) bosUnit x)) • parityUnit := by
  rw [completionFFN_apply, relu2_of_nonpos _ (by linarith : -phaseProbe x ≤ 0)]
  simp

example : 0 ≤ phaseProbe phaseUnit := by
  norm_num [phaseProbe, phaseUnit, innerSL_apply_apply, EuclideanSpace.inner_single_left]

/-- Negative supplied-answer phase gives a strictly positive EOS signal at any positive output scale.
Source: the positive ReLU2 branch in the ordinary additional FFN unit. -/
theorem completionFFN_answer_signal (eosScale : ℝ) (hscale : 0 < eosScale)
    (x : EucSpace 64) (hphase : phaseProbe x < 0) :
    0 < eosScale * relu2 (-phaseProbe x) := by
  have hp : 0 < -phaseProbe x := by linarith
  rw [relu2_of_pos _ hp]
  exact mul_pos hscale (sq_pos_of_pos hp)

example : (0 : ℝ) < 1 ∧ phaseProbe (-phaseUnit) < 0 := by
  norm_num [phaseProbe, phaseUnit, innerSL_apply_apply, EuclideanSpace.inner_single_left]

/-- The actual FFN prenorm multiplies the completion hinge by the square of its positive common scale.
Source: RMSNorm and the proved positive homogeneity of the original ReLU2 activation. -/
theorem completion_phase_rms (eps : ℝ) (x : EucSpace 64) :
    relu2 (-phaseProbe (rmsNormEps eps x)) =
      (Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 * relu2 (-phaseProbe x) := by
  simp only [rmsNormEps, phaseProbe, innerSL_apply_apply, real_inner_smul_right]
  norm_num only [Nat.cast_ofNat]
  have harg : -(8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps) * inner (𝕜 := ℝ) phaseUnit x) =
      (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) * (-inner (𝕜 := ℝ) phaseUnit x) := by ring
  rw [harg, relu2_mul_nonneg _ _ (by positivity)]

/-- Both decoded outputs are orthogonal to completion phase, so the FFN cannot erase the phase code.
Source: parityUnit and eosUnit occupy residual coordinates three and four, independently of phase coordinate one. -/
theorem completionFFN_phase_preserved (parityScale eosScale : ℝ) (x : EucSpace 64) :
    phaseProbe (x + relu2FFN completionFFNIn (completionFFNOut parityScale eosScale) x) =
      phaseProbe x := by
  rw [completionFFN_apply, phaseProbe.map_add]
  have horth : phaseProbe
      ((parityScale * paritySpline (inner (𝕜 := ℝ) TokenInterface.controlUnit x)
          (inner (𝕜 := ℝ) bosUnit x)) • parityUnit +
        (eosScale * relu2 (-phaseProbe x)) • eosUnit) = 0 := by
    simp [phaseProbe, innerSL_apply_apply, inner_add_right,
      phaseUnit, parityUnit, eosUnit, EuclideanSpace.inner_single_left]
  rw [horth, add_zero]

/-!
The active phase unit is independent of the parity sign. For tied readout,
the original embedding table must assign distinct EVEN/ODD code directions
and an EOS direction. The earlier ratioParams feature control deliberately
did not contain those output codes, so these channel formulas alone are
not a theorem that its final greedy decoder already solves the task.
-/

end Transformer.GPTMini.Semantics
