import Transformer.GPTMini.Semantics.RecallGateScalar

/-!
# Realize the recall table gate in the original FFN matrices

Source: W_in/ReLU2/W_out at f11b6e2 and MQAR's fixed table size
at cbafbe9. The gate is an ordinary linear form in the protected
constant, value-type flag and actual causal BOS marker. Each copied
key coordinate has two actual preactivations g+eta*z and g-eta*z.

Sixteen of the original 256 units compute the eight gated coordinates.
The true output matrix writes them into fresh coordinates 27..34,
leaving the raw code/type and predecessor-copy channels available.
No cancellation of an unknown RMS factor is assumed. The actual
prenorm formula instead retains its homogeneous square explicitly.
These are finite shared matrices, not a switch on semantic table
membership. Gate outcomes on raw inputs are proved subsequently.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- An ordinary coordinate functional in the original residual dimension.
Source: the actual bias-free matrix rows, expressed as real inner products. -/
noncomputable def recallProbe (c : Fin 64) : EucSpace 64 →L[ℝ] ℝ := innerSL ℝ (recallUnit c)

/-- Every input coordinate is faithfully read by this matrix row.
Source: the real Euclidean unit-coordinate inner product. -/
theorem recallProbe_apply (c : Fin 64) (x : EucSpace 64) : recallProbe c x = x c := by
  unfold recallProbe recallUnit
  rw [innerSL_apply_apply, EuclideanSpace.inner_single_left, map_one, one_mul]

/-- Value-type exclusion and the fixed table cutoff are one bias-free linear form.
Source: protected residual axes 0/36 and the genuine reciprocal-prefix marker on axis 26. -/
noncomputable def recallGateForm (P : ℕ) : EucSpace 64 →L[ℝ] ℝ :=
  recallProbe 26 - recallTableThreshold P • recallProbe 0 -
    (2 : ℝ) • (recallProbe 0 - recallProbe 36)

/-- The actual row uses only internally computed real coordinates.
Source: the explicitly stated bias-free ordinary matrix row. -/
theorem recallGateForm_apply (P : ℕ) (x : EucSpace 64) :
    recallGateForm P x = x 26 - recallTableThreshold P * x 0 - 2 * (x 0 - x 36) := by
  simp only [recallGateForm, sub_apply, smul_apply, recallProbe_apply, smul_eq_mul]

/-- Two units per copied coordinate fit into the unchanged 256-unit FFN.
Source: the explicit contiguous sixteen-unit assignment. -/
def recallGateIndex (c : Fin 8) (r : Fin 2) : Fin recallConfig.d_ff :=
  ⟨2 * c.val + r.val, by change 2 * c.val + r.val < 256; have hc := c.isLt; have hr := r.isLt; omega⟩

/-- Distinct pairs cannot interfere in the actual hidden coordinates.
Source: the two-unit assignment and the precise bound r < 2. -/
theorem recallGateIndex_eq (c d : Fin 8) (r t : Fin 2) :
    recallGateIndex c r = recallGateIndex d t ↔ c = d ∧ r = t := by
  rw [Fin.ext_iff]
  change 2 * c.val + r.val = 2 * d.val + t.val ↔ c = d ∧ r = t
  constructor
  · intro h
    have hr := r.isLt
    have ht := t.isLt
    exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- The signed copied-coordinate perturbation distinguishes the two actual input rows.
Source: the two quadratic preactivations of RecallGateScalar. -/
def recallGateSign (r : Fin 2) : ℝ := if r = 0 then 1 else -1

/-- Actual rank-one input matrices write all sixteen linear preactivations simultaneously.
Source: the original W_in type, with no bias and no extra hidden units. -/
noncomputable def recallGateIn (P : ℕ) : EucSpace 64 →L[ℝ] EucSpace recallConfig.d_ff :=
  ∑ c : Fin 8, ∑ r : Fin 2,
    (recallGateForm P + (recallGateSign r * (recallTableMargin P / 8)) •
      recallProbe (recallSlotIndex 18 c)).smulRight
        (EuclideanSpace.single (recallGateIndex c r) 1)

/-- The true output matrix subtracts each pair and writes it to the fresh gated-key slot.
Source: original W_out, with finite gain beta and the strictly positive fixed table margin. -/
noncomputable def recallGateOut (P : ℕ) (beta : ℝ) :
    EucSpace recallConfig.d_ff →L[ℝ] EucSpace 64 :=
  ∑ c : Fin 8,
    (innerSL ℝ (EuclideanSpace.single (recallGateIndex c 0) 1 : EucSpace recallConfig.d_ff) -
      innerSL ℝ (EuclideanSpace.single (recallGateIndex c 1) 1 : EucSpace recallConfig.d_ff)).smulRight
        ((beta / (4 * (recallTableMargin P / 8))) • recallUnit (recallSlotIndex 27 c))

/-- A coordinate of the genuine input matrix is its exact gate-plus-signed-copy preactivation.
Source: all finite rank-one sums, with the proved hidden-coordinate injectivity discharged. -/
theorem recallGateIn_coordinate (P : ℕ) (x : EucSpace 64) (c : Fin 8) (r : Fin 2) :
    recallGateIn P x (recallGateIndex c r) =
      recallGateForm P x + recallGateSign r * (recallTableMargin P / 8) * x (recallSlotIndex 18 c) := by
  unfold recallGateIn
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, add_apply, smul_apply,
    recallProbe_apply, smul_eq_mul, WithLp.ofLp_sum, Finset.sum_apply,
    WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply]
  rw [Fintype.sum_eq_single c]
  · rw [Fintype.sum_eq_single r]
    · simp
    · intro t ht
      have hn : recallGateIndex c r ≠ recallGateIndex c t :=
        fun he => ht ((recallGateIndex_eq c c r t).mp he).2.symm
      simp only [ite_eq_right hn, mul_zero]
  · intro d hd
    apply Finset.sum_eq_zero
    intro t _
    have hn : recallGateIndex c r ≠ recallGateIndex d t :=
      fun he => hd ((recallGateIndex_eq c d r t).mp he).1.symm
    simp only [ite_eq_right hn, mul_zero]

/-- The actual ReLU2_FFN produces the eight stated gated products, without a semantic oracle.
Source: true W_in, coordinatewise quadratic activation and the simultaneous W_out. -/
theorem recallGateFFN_apply (P : ℕ) (beta : ℝ) (x : EucSpace 64) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) x =
      ∑ c : Fin 8,
        (beta * recallGateProduct (recallTableMargin P / 8) (recallGateForm P x)
          (x (recallSlotIndex 18 c))) • recallUnit (recallSlotIndex 27 c) := by
  unfold relu2FFN recallGateOut
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, sub_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, relu2Vec_apply, smul_smul]
  apply Finset.sum_congr rfl
  intro c _
  change ((relu2 (recallGateIn P x (recallGateIndex c 0)) -
    relu2 (recallGateIn P x (recallGateIndex c 1))) *
    (beta / (4 * (recallTableMargin P / 8)))) • recallUnit (recallSlotIndex 27 c) = _
  rw [recallGateIn_coordinate, recallGateIn_coordinate]
  simp only [recallGateSign, ite_true, ite_eq_right (by decide : (1 : Fin 2) ≠ 0),
    one_mul, neg_one_mul]
  unfold recallGateProduct
  congr 1
  ring_nf

/-- The original FFN prenorm leaves the table test homogeneous and records its genuine squared scale.
Source: actual RMSNorm at f11b6e2, followed by the proved shared finite matrices. -/
theorem recallGateFFN_rms (P : ℕ) (beta eps : ℝ) (x : EucSpace 64) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x) =
      ∑ c : Fin 8,
        (beta * (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 *
          recallGateProduct (recallTableMargin P / 8) (recallGateForm P x)
            (x (recallSlotIndex 18 c))) • recallUnit (recallSlotIndex 27 c) := by
  rw [recallGateFFN_apply]
  unfold rmsNormEps
  norm_num only [Nat.cast_ofNat]
  simp only [map_smul, PiLp.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro c _
  change (beta * recallGateProduct (recallTableMargin P / 8)
    ((8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) * recallGateForm P x)
    ((8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) * x (recallSlotIndex 18 c))) • _ = _
  rw [recallGateProduct_scale _ _ _ _ (by positivity : 0 ≤ 8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps))]
  congr 1
  ring

end Transformer.GPTMini.Semantics
