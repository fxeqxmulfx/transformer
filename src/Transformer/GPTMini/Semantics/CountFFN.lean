import Transformer.GPTMini.Semantics.CountSpline

/-!
# Realize the homogeneous count decoder in the original FFN matrices

Source: ReLU2_FFN.forward at f11b6e2, using the new four-hinge construction
in CountSpline. The 17 count centers each occupy four distinct coordinates
among the original 256 hidden units. Every input preactivation is a linear
form in the ONE and BOS residual coordinates. The ordinary output matrix
adds the signed hinge contributions into a new residual output coordinate.

The result evaluates the actual relu2FFN operator, including the finite
matrix coordinate layout. It is not a replacement FFN calling an oracle
or computing integer parity directly. Unused FFN coordinates do not
contribute to the result. EOS, tied embedding choices, prenorm margins
and the full autoregressive integer function remain separate obligations.
No assertion about joint parameter convexity or AdamW is made here.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open TokenInterface

/-- The first 68 of the original 256 FFN coordinates contain four hinges for each count.
Source: the explicit matrix realization, with no increase to the original width. -/
def bumpIndex (k : Fin 17) (r : Fin 4) : Fin countConfig.d_ff :=
  ⟨4 * k.val + r.val, by change 4 * k.val + r.val < 256; have hk := k.isLt; have hr := r.isLt; omega⟩

/-- Different count/hinge pairs occupy different actual FFN coordinates.
Source: the contiguous four-coordinate layout and the bounds r < 4. -/
theorem bumpIndex_eq (k j : Fin 17) (r t : Fin 4) :
    bumpIndex k r = bumpIndex j t ↔ k = j ∧ r = t := by
  rw [Fin.ext_iff]
  change 4 * k.val + r.val = 4 * j.val + t.val ↔ k = j ∧ r = t
  constructor
  · intro h
    have hr := r.isLt
    have ht := t.isLt
    have hk : k.val = j.val := by omega
    have hrt : r.val = t.val := by omega
    exact ⟨Fin.ext hk, Fin.ext hrt⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- Read a coordinate of a finite sum of the assigned hinge units without interference.
Source: the proved injectivity of the actual FFN coordinate assignment. -/
theorem bump_sum_coordinate (f : Fin 17 → Fin 4 → ℝ) (k : Fin 17) (r : Fin 4) :
    (∑ j : Fin 17, ∑ t : Fin 4,
        f j t • (EuclideanSpace.single (bumpIndex j t) 1 : EucSpace countConfig.d_ff))
      (bumpIndex k r) = f k r := by
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul,
    Pi.smul_apply, PiLp.single_apply, smul_eq_mul]
  rw [Fintype.sum_eq_single k]
  · rw [Fintype.sum_eq_single r]
    · simp
    · intro t ht
      have hne : bumpIndex k r ≠ bumpIndex k t := by
        intro heq
        exact ht ((bumpIndex_eq k k r t).mp heq).2.symm
      simp only [ite_eq_right hne, mul_zero]
  · intro j hj
    apply Finset.sum_eq_zero
    intro t _
    have hne : bumpIndex k r ≠ bumpIndex j t := by
      intro heq
      exact hj ((bumpIndex_eq k j r t).mp heq).1.symm
    simp only [ite_eq_right hne, mul_zero]

/-- The four fixed relative shifts are coefficients of the internal BOS coordinate.
Source: countBumpHom's bias-free linear forms; the shifts are independent of a raw input word. -/
noncomputable def bumpShift (r : Fin 4) : ℝ :=
  if r.val = 0 then -3 / 4 else if r.val = 1 then -1 / 4 else
    if r.val = 2 then 1 / 4 else 3 / 4

/-- Coefficients of the four quadratic hinges, including the exact center normalization.
Source: CountSpline's explicit four-term finite difference. -/
noncomputable def bumpCoeff (r : Fin 4) : ℝ :=
  (8 / 3 : ℝ) * (if r.val = 0 then 1 else if r.val = 1 then -3 else
    if r.val = 2 then 3 else -1)

/-- Parity output uses a fourth residual coordinate, separate from count, BOS and phase.
Source: the new decoder's ordinary output direction in the original 64-wide stream. -/
noncomputable def parityUnit : EucSpace 64 := EuclideanSpace.single 3 1

/-- The original bias-free input matrix writes all 68 linear forms into the assigned FFN coordinates.
Source: the actual W_in operator type at f11b6e2, realized as a finite sum of rank-one matrices. -/
noncomputable def parityFFNIn : EucSpace 64 →L[ℝ] EucSpace countConfig.d_ff :=
  ∑ k : Fin 17, ∑ r : Fin 4,
    ((innerSL ℝ controlUnit) - ((k.val : ℝ) + bumpShift r) • (innerSL ℝ bosUnit)).smulRight
      (EuclideanSpace.single (bumpIndex k r) 1)

/-- The original output matrix combines signed activations into the parity residual coordinate.
Source: W_out at f11b6e2, with no nonlinear oracle in its definition. -/
noncomputable def parityFFNOut : EucSpace countConfig.d_ff →L[ℝ] EucSpace 64 :=
  ∑ k : Fin 17, ∑ r : Fin 4,
    (innerSL ℝ (EuclideanSpace.single (bumpIndex k r) 1)).smulRight
      ((countSign k.val * bumpCoeff r) • parityUnit)

/-- A scalar preactivation of the actual input matrix is exactly its count-minus-shift-times-BOS form.
Source: the finite rank-one matrix sum and its proved disjoint coordinate assignment. -/
theorem parityFFNIn_coordinate (x : EucSpace 64) (k : Fin 17) (r : Fin 4) :
    parityFFNIn x (bumpIndex k r) =
      inner (𝕜 := ℝ) controlUnit x - ((k.val : ℝ) + bumpShift r) * inner (𝕜 := ℝ) bosUnit x := by
  unfold parityFFNIn
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, sub_apply, smul_apply,
    innerSL_apply_apply, smul_eq_mul]
  exact bump_sum_coordinate _ k r

/-- The four actual activation coordinates realize one homogeneous count bump.
Source: the explicit matrix coefficients and shifts, using the unchanged scalar ReLU2. -/
theorem bump_four_activations (a b : ℝ) (k : Fin 17) :
    (∑ r : Fin 4, relu2 (a - ((k.val : ℝ) + bumpShift r) * b) *
      (countSign k.val * bumpCoeff r)) = countSign k.val * countBumpHom a b (k.val : ℝ) := by
  rw [Fin.sum_univ_four]
  norm_num [bumpShift, bumpCoeff]
  unfold countBumpHom
  ring_nf

/-- The actual original FFN computes the proved homogeneous parity signal from its own two input coordinates.
Source: relu2FFN's true W_in, coordinatewise ReLU2 and W_out, with all finite matrix sums discharged. -/
theorem parityFFN_apply (x : EucSpace 64) :
    relu2FFN parityFFNIn parityFFNOut x =
      paritySpline (inner (𝕜 := ℝ) controlUnit x) (inner (𝕜 := ℝ) bosUnit x) • parityUnit := by
  unfold relu2FFN parityFFNOut
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, relu2Vec_apply,
    smul_smul]
  have hcoord (k : Fin 17) (r : Fin 4) :
      (parityFFNIn x).ofLp (bumpIndex k r) =
        inner (𝕜 := ℝ) controlUnit x - ((k.val : ℝ) + bumpShift r) * inner (𝕜 := ℝ) bosUnit x :=
    parityFFNIn_coordinate x k r
  simp_rw [hcoord]
  simp_rw [← Finset.sum_smul]
  unfold paritySpline
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  exact bump_four_activations _ _ k

/-- The true FFN prenorm preserves the bounded-count decoder and scales its margin quadratically.
Source: RMSNorm followed by the original ReLU2_FFN; the two premises are local input coordinates. -/
theorem parityFFN_rms_count (eps : ℝ) (x : EucSpace 64) (s : ℝ) (hs : 0 ≤ s)
    (n : ℕ) (hn : n ≤ 16) (ha : inner (𝕜 := ℝ) controlUnit x = s * n)
    (hb : inner (𝕜 := ℝ) bosUnit x = s) :
    relu2FFN parityFFNIn parityFFNOut (rmsNormEps eps x) =
      ((Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps) * s) ^ 2 * countSign n) • parityUnit := by
  rw [parityFFN_apply]
  simp only [rmsNormEps, real_inner_smul_right, ha, hb]
  have hscale : 0 ≤ Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps) * s := by positivity
  norm_num only [Nat.cast_ofNat] at hscale ⊢
  rw [← mul_assoc, paritySpline_exact _ hscale n hn]

example : (0 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 16 ∧
    inner (𝕜 := ℝ) controlUnit ((2 : ℝ) • controlUnit + bosUnit) = 1 * 2 ∧
    inner (𝕜 := ℝ) bosUnit ((2 : ℝ) • controlUnit + bosUnit) = 1 := by
  norm_num [inner_add_right, real_inner_smul_right, controlUnit, bosUnit,
    EuclideanSpace.inner_single_left]

/-- Coordinate 68 is disjoint from every parity hinge and remains available for completion phase.
Source: the actual 4*k+r coordinate layout occupies only indices zero through 67. -/
theorem bumpIndex_ne_spare (k : Fin 17) (r : Fin 4) :
    bumpIndex k r ≠ (⟨68, by decide⟩ : Fin countConfig.d_ff) := by
  intro h
  have hi := congrArg Fin.val h
  change 4 * k.val + r.val = 68 at hi
  have hk := k.isLt
  have hr := r.isLt
  omega

/-- The actual parity input matrix is zero at the spare phase coordinate.
Source: the proved index separation, independent of the raw input or desired answer. -/
theorem parityFFNIn_spare (x : EucSpace 64) :
    parityFFNIn x ⟨68, by decide⟩ = 0 := by
  unfold parityFFNIn
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, WithLp.ofLp_sum,
    Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply, smul_eq_mul]
  apply Finset.sum_eq_zero
  intro k _
  apply Finset.sum_eq_zero
  intro r _
  rw [ite_eq_right (Ne.symm (bumpIndex_ne_spare k r)), mul_zero]

end Transformer.GPTMini.Semantics
