import Transformer.GPTMini.Semantics.RecallRawBinding

/-!
# An ordinary linear matrix inserts genuine copied keys into slow RoPE pairs

Source: the original sixteen-coordinate head at f11b6e2 and its
contiguous half-pairing. Compact raw and copied codes are interleaved
eight-coordinate vectors; the matching representation occupies the
actual pairs four through seven. An ordinary shared matrix must
perform this coordinate change on imperfect real copies as well as
on categorical symbol codes. A symbol-only nonlinear decoder would
not realize the required original QKV projection.

This finite rank-one matrix retains all eight real coordinates. Its
norm and distance calculations are exact, so arbitrary copy errors
are not amplified before the true rotary operator. Its symbol-code
output is the previously verified actual matching geometry. Further
QKV coupling and prenorm/QKNorm saturation remain separate steps.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Even compact coordinates go to 4..7 and odd coordinates to their actual rotary partners 12..15.
Source: apply_rope's two contiguous halves at the original head dimension sixteen. -/
def recallRotaryIndex (c : Fin 8) : Fin 16 :=
  if c.val % 2 = 0 then ⟨4 + c.val / 2, by have hc := c.isLt; omega⟩
  else ⟨12 + c.val / 2, by have hc := c.isLt; omega⟩

/-- All eight actual destination coordinates are distinct.
Source: disjoint half intervals and recovery of compact parity and quotient. -/
theorem recallRotaryIndex_injective : Function.Injective recallRotaryIndex := by decide

/-- The genuine shared matrix inserts every real compact coordinate, including imperfect softmax copies.
Source: finite ordinary rank-one matrices of the original head projection type. -/
noncomputable def recallRotaryInsert : EucSpace 8 →L[ℝ] EucSpace 16 :=
  ∑ c : Fin 8, (innerSL ℝ (EuclideanSpace.single c 1)).smulRight
    (EuclideanSpace.single (recallRotaryIndex c) 1)

/-- The complete matrix output is the stated sum of eight actual head axes.
Source: evaluated rank-one input/output operators, with no symbol or task assumptions. -/
theorem recallRotaryInsert_apply (code : EucSpace 8) :
    recallRotaryInsert code =
      ∑ c : Fin 8, code c • (EuclideanSpace.single (recallRotaryIndex c) 1 : EucSpace 16) := by
  simp only [recallRotaryInsert, sum_apply, ContinuousLinearMap.smulRight_apply,
    innerSL_apply_apply, EuclideanSpace.inner_single_left, map_one, one_mul]

/-- Each selected real compact coordinate is faithfully present at its assigned actual head index.
Source: the complete finite matrix sum and proved destination injectivity. -/
theorem recallRotaryInsert_at (code : EucSpace 8) (c : Fin 8) :
    recallRotaryInsert code (recallRotaryIndex c) = code c := by
  rw [recallRotaryInsert_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul,
    Pi.smul_apply, PiLp.single_apply, smul_eq_mul]
  rw [Fintype.sum_eq_single c]
  · simp
  · intro j hj
    have hn : recallRotaryIndex c ≠ recallRotaryIndex j :=
      fun he => hj (recallRotaryIndex_injective he).symm
    simp only [ite_eq_right hn, mul_zero]

/-- The actual linear insertion preserves every real inner product, rather than only categorical code norms.
Source: evaluated faithful destination coordinates and the eight-coordinate real Euclidean inner product. -/
theorem recallRotaryInsert_inner (x y : EucSpace 8) :
    inner (𝕜 := ℝ) (recallRotaryInsert x) (recallRotaryInsert y) = inner (𝕜 := ℝ) x y := by
  rw [recallRotaryInsert_apply x]
  simp only [sum_inner, real_inner_smul_left, EuclideanSpace.inner_single_left,
    map_one, one_mul, recallRotaryInsert_at]
  rw [PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial, mul_comm]

/-- Every actual inserted imperfect compact key has its exact original norm.
Source: the matrix's full evaluated inner product and nonnegative real norms. -/
theorem recallRotaryInsert_norm (x : EucSpace 8) : ‖recallRotaryInsert x‖ = ‖x‖ := by
  have hs := recallRotaryInsert_inner x x
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq] at hs
  nlinarith [norm_nonneg (recallRotaryInsert x), norm_nonneg x]

/-- The ordinary matching insertion preserves softmax-copy error exactly.
Source: true matrix linear subtraction and the proved isometry for all real compact vectors. -/
theorem recallRotaryInsert_dist (x y : EucSpace 8) :
    ‖recallRotaryInsert x - recallRotaryInsert y‖ = ‖x - y‖ := by
  rw [← map_sub, recallRotaryInsert_norm]

/-- All actual head coordinates have the explicit interleaved-to-half-pair matrix formula.
Source: finite expansion of every original sixteen-coordinate matrix row. -/
theorem recallRotaryInsert_coordinate (code : EucSpace 8) (i : Fin 16) :
    recallRotaryInsert code i =
      if h : 4 ≤ i.val ∧ i.val < 8 then code ⟨2 * (i.val - 4), by omega⟩
      else if h : 12 ≤ i.val then code ⟨2 * (i.val - 12) + 1, by omega⟩ else 0 := by
  rw [recallRotaryInsert_apply, Fin.sum_univ_eight]
  fin_cases i <;> norm_num [recallRotaryIndex, Pi.add_apply, Pi.smul_apply, Pi.single_apply]
  all_goals rfl

/-- The true linear matrix sends every raw compact symbol code to its already verified rotary matching representation.
Source: explicit coordinate formulas on both sides, with every original half-pair and categorical digit retained. -/
theorem recallRotaryInsert_code (digits : Fin 4 → Fin 4) :
    recallRotaryInsert (recallCode digits) = recallRotaryCode digits := by
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext i
  change recallRotaryInsert (recallCode digits) i = recallRotaryCode digits i
  rw [recallRotaryInsert_coordinate]
  fin_cases i <;> norm_num [recallCode, recallRotaryCode]

/-- The complete ordinary matrix commutes with every position-dependent positive or negative scalar amplitude.
Source: actual continuous linear map structure, without an assumption of equal record amplitudes. -/
theorem recallRotaryInsert_scaled_code (s : ℝ) (digits : Fin 4 → Fin 4) :
    recallRotaryInsert (s • recallCode digits) = s • recallRotaryCode digits := by
  rw [map_smul, recallRotaryInsert_code]

/-- The original rotary transformation also preserves the exact inserted copied-key error.
Source: actual RoPE's real isometry, composed with the ordinary matching insertion. -/
theorem recallRotaryInsert_rotated_dist (position : ℝ) (x y : EucSpace 8) :
    ‖applyRope 16 10000 position (recallRotaryInsert x) -
      applyRope 16 10000 position (recallRotaryInsert y)‖ = ‖x - y‖ := by
  rw [applyRope_dist, recallRotaryInsert_dist]

/-- This insertion cannot erase a real norm-two symbol, even when its amplitude differs across table positions.
Source: genuine matrix isometry, compact-code norm two and exact norm homogeneity. -/
theorem recallRotaryInsert_scaled_norm (s : ℝ) (digits : Fin 4 → Fin 4) :
    ‖recallRotaryInsert (s • recallCode digits)‖ = 2 * |s| := by
  rw [recallRotaryInsert_norm, norm_smul, Real.norm_eq_abs, recallCode_norm]
  ring

/-- Every excluded zero compact key remains exactly zero in the real matching head.
Source: the ordinary linear insertion, needed for nonrecords and post-table false writes. -/
theorem recallRotaryInsert_zero : recallRotaryInsert 0 = 0 := by
  exact map_zero recallRotaryInsert

/-- All unused original fast-pair coordinates stay zero even for imperfect real copied keys.
Source: the complete evaluated sixteen-coordinate insertion matrix and its exact half-pair support. -/
theorem recallRotaryInsert_outside (code : EucSpace 8) (i : Fin 16)
    (hout : i.val < 4 ∨ (8 ≤ i.val ∧ i.val < 12)) : recallRotaryInsert code i = 0 := by
  rw [recallRotaryInsert_coordinate]
  have hn : ¬(4 ≤ i.val ∧ i.val < 8) := by omega
  have hm : ¬12 ≤ i.val := by omega
  rw [dite_eq_right hn, dite_eq_right hm]

example : (0 : Fin 16).val < 4 ∨ (8 ≤ (0 : Fin 16).val ∧ (0 : Fin 16).val < 12) := by decide

/-- The actual linear projection of every raw symbol still has the exact original norm two.
Source: faithful compact symbol encoding and isometry of the genuine eight-to-sixteen matrix. -/
theorem recallRotaryInsert_symbol_norm (symbol : Fin 256) :
    ‖recallRotaryInsert (recallSymbolCode symbol)‖ = 2 := by
  rw [recallRotaryInsert_norm]
  exact recallCode_norm _

/-- A derived compact copy tolerance survives both the actual matching matrix and original RoPE.
Source: their full real isometries, rather than a symbol-only norm calculation. -/
theorem recallRotaryInsert_error (position eta : ℝ) (x y : EucSpace 8) (herr : ‖x - y‖ ≤ eta) :
    ‖applyRope 16 10000 position (recallRotaryInsert x) -
      applyRope 16 10000 position (recallRotaryInsert y)‖ ≤ eta := by
  rw [recallRotaryInsert_rotated_dist]
  exact herr

example : ‖(0 : EucSpace 8) - 0‖ ≤ (0 : ℝ) := by rw [sub_self, norm_zero]

end Transformer.GPTMini.Semantics
