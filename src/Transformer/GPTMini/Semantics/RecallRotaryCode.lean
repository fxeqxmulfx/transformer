import Transformer.GPTMini.Semantics.RecallFrequencies

/-!
# Compact symbol codes in the actual sixteen-coordinate RoPE head

Source: apply_rope's contiguous half-pairing at f11b6e2 and the new
eight-coordinate recall codes. Pairs four through seven retain all
256 symbols; the other four pairs are zero. This module evaluates the
actual coordinate functions and actual rotation, with no hypothetical
position-independent attention. All codes have norm two before and
after RoPE. Score separation is established in the following module.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The compact categorical code placed in the four slow pairs of the original head.
Source: the actual 16-coordinate contiguous-half layout, retaining every input digit. -/
noncomputable def recallRotaryCode (digits : Fin 4 → Fin 4) : EucSpace 16 :=
  (EuclideanSpace.equiv (Fin 16) ℝ).symm fun i =>
    if h : 4 ≤ i.val ∧ i.val < 8 then quarterX (digits ⟨i.val - 4, by omega⟩)
    else if h : 12 ≤ i.val then quarterY (digits ⟨i.val - 12, by have hi := i.isLt; omega⟩)
    else 0

/-- First-half coordinates of every actual RoPE pair, including the unused fast pairs.
Source: original ropeFst and the concrete code table; no prepared Q/K feature is assumed. -/
theorem recallRotaryCode_fst (digits : Fin 4 → Fin 4) (p : Fin 8) :
    recallRotaryCode digits (ropeFst 16 p) =
      if h : 4 ≤ p.val then quarterX (digits ⟨p.val - 4, by omega⟩) else 0 := by
  have hf : ∀ p : Fin 8, (ropeFst 16 p).val = p.val := by decide
  have hp := p.isLt
  by_cases h : 4 ≤ p.val
  · simp [recallRotaryCode, hf, h, hp]
  · have hh : ¬12 ≤ p.val := by omega
    simp [recallRotaryCode, hf, h, hh]

/-- Second-half coordinates of every actual pair use the same categorical digit.
Source: original ropeSnd adds eight, so pair four's partner is coordinate twelve. -/
theorem recallRotaryCode_snd (digits : Fin 4 → Fin 4) (p : Fin 8) :
    recallRotaryCode digits (ropeSnd 16 p) =
      if h : 4 ≤ p.val then quarterY (digits ⟨p.val - 4, by omega⟩) else 0 := by
  have hs : ∀ p : Fin 8, (ropeSnd 16 p).val = p.val + 8 := by decide
  have hp := p.isLt
  have hfast : ¬p.val + 8 < 8 := by omega
  have hindex : p.val + 8 - 12 = p.val - 4 := by omega
  by_cases h : 4 ≤ p.val
  · have hh : 12 ≤ p.val + 8 := by omega
    simp [recallRotaryCode, hs, h, hfast, hh, hindex]
  · have hh : ¬12 ≤ p.val + 8 := by omega
    simp [recallRotaryCode, hs, h, hfast, hh]

/-- Placing the code in the original pair layout preserves its exact squared norm.
Source: the full sixteen-coordinate Euclidean sum and four actual unit digit pairs. -/
theorem recallRotaryCode_norm_sq (digits : Fin 4 → Fin 4) :
    ‖recallRotaryCode digits‖ ^ 2 = 4 := by
  rw [EuclideanSpace.norm_sq_eq, sum_split 16]
  change ((∑ p : Fin 8, ‖recallRotaryCode digits (ropeFst 16 p)‖ ^ 2) +
    ∑ p : Fin 8, ‖recallRotaryCode digits (ropeSnd 16 p)‖ ^ 2) +
      (∑ p : Fin 0, ‖recallRotaryCode digits ((ropeSplit 16).symm (Sum.inr p))‖ ^ 2) = 4
  simp only [Fin.sum_univ_eight, recallRotaryCode_fst, recallRotaryCode_snd,
    Real.norm_eq_abs, sq_abs]
  norm_num
  nlinarith [quarter_unit (digits 0), quarter_unit (digits 1),
    quarter_unit (digits 2), quarter_unit (digits 3)]

/-- The actual rotary-layout vector has norm two, independently of its symbol.
Source: the exact squared norm and nonnegativity of the genuine head vector. -/
theorem recallRotaryCode_norm (digits : Fin 4 → Fin 4) : ‖recallRotaryCode digits‖ = 2 := by
  nlinarith [recallRotaryCode_norm_sq digits, norm_nonneg (recallRotaryCode digits)]

/-- Actual RoPE preserves this norm at every real position.
Source: the original applyRope_isometry, so QKNorm has the same divisor at every position. -/
theorem recallRotaryCode_rotated_norm (digits : Fin 4 → Fin 4) (position : ℝ) :
    ‖applyRope 16 10000 position (recallRotaryCode digits)‖ = 2 := by
  rw [applyRope_isometry, recallRotaryCode_norm]

/-- First coordinates of the actual rotated pair are the original two-coordinate rotation formula.
Source: apply_rope at f11b6e2, with its unchanged exact angle at every pair. -/
theorem recallRotaryCode_rotated_fst (digits : Fin 4 → Fin 4) (position : ℝ) (p : Fin 8) :
    applyRope 16 10000 position (recallRotaryCode digits) (ropeFst 16 p) =
      recallRotaryCode digits (ropeFst 16 p) * Real.cos (ropeAngle 16 10000 position p) -
        recallRotaryCode digits (ropeSnd 16 p) * Real.sin (ropeAngle 16 10000 position p) := by
  have hp : ropeSplit 16 (ropeFst 16 p) = Sum.inl (Sum.inl p) := by simp [ropeFst]
  rw [applyRope_apply, hp]
  simp only [ropeCoord, Sum.elim_inl]

/-- Second coordinates are rotated using the same pair and exact frequency.
Source: original apply_rope; rotating the copied key code does not remove positional information. -/
theorem recallRotaryCode_rotated_snd (digits : Fin 4 → Fin 4) (position : ℝ) (p : Fin 8) :
    applyRope 16 10000 position (recallRotaryCode digits) (ropeSnd 16 p) =
      recallRotaryCode digits (ropeFst 16 p) * Real.sin (ropeAngle 16 10000 position p) +
        recallRotaryCode digits (ropeSnd 16 p) * Real.cos (ropeAngle 16 10000 position p) := by
  have hp : ropeSplit 16 (ropeSnd 16 p) = Sum.inl (Sum.inr p) := by simp [ropeSnd]
  rw [applyRope_apply, hp]
  simp only [ropeCoord, Sum.elim_inl, Sum.elim_inr]

/-- The actual two-coordinate contribution of a categorical pair after relative rotation.
Source: the dot product with the original real two-dimensional rotation matrix. -/
noncomputable def quarterRotaryInner (a b : Fin 4) (angle : ℝ) : ℝ :=
  (quarterX a * quarterX b + quarterY a * quarterY b) * Real.cos angle +
    (quarterY a * quarterX b - quarterX a * quarterY b) * Real.sin angle

/-- The actual full rotary-head inner product is exactly the sum of four categorical rotated pair scores.
Source: original applyRope_relative and all sixteen actual coordinates; every unequal frequency is retained. -/
theorem recallRotaryCode_inner (a b : Fin 4 → Fin 4) (s t : ℝ) :
    inner (𝕜 := ℝ) (applyRope 16 10000 s (recallRotaryCode a))
      (applyRope 16 10000 t (recallRotaryCode b)) =
        ∑ p : Fin 4, quarterRotaryInner (a p) (b p) ((t - s) * recallFrequency p) := by
  rw [applyRope_relative, PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial]
  rw [sum_split 16]
  change ((∑ p : Fin 8, applyRope 16 10000 (t - s) (recallRotaryCode b) (ropeFst 16 p) *
      recallRotaryCode a (ropeFst 16 p)) +
    ∑ p : Fin 8, applyRope 16 10000 (t - s) (recallRotaryCode b) (ropeSnd 16 p) *
      recallRotaryCode a (ropeSnd 16 p)) +
    (∑ p : Fin 0, applyRope 16 10000 (t - s) (recallRotaryCode b) ((ropeSplit 16).symm (Sum.inr p)) *
      recallRotaryCode a ((ropeSplit 16).symm (Sum.inr p))) = _
  simp only [Fin.sum_univ_eight, recallRotaryCode_rotated_fst, recallRotaryCode_rotated_snd,
    recallRotaryCode_fst, recallRotaryCode_snd, Fin.sum_univ_four, quarterRotaryInner,
    ropeAngle, recallFrequency]
  norm_num
  ring

/-- Actual QKNorm divides each code by two, producing the exact normalized four-pair score.
Source: original score/normL2, with the proved genuine post-RoPE norm and explicit clipping range. -/
theorem recallRotaryCode_score (alpha eps s t : ℝ) (heps : eps ≤ 2) (a b : Fin 4 → Fin 4) :
    score alpha eps (applyRope 16 10000 s (recallRotaryCode a))
      (applyRope 16 10000 t (recallRotaryCode b)) =
        Real.exp alpha * ((∑ p : Fin 4,
          quarterRotaryInner (a p) (b p) ((t - s) * recallFrequency p)) / 4) := by
  rw [score, normL2, normL2, recallRotaryCode_rotated_norm, recallRotaryCode_rotated_norm,
    max_eq_left heps, real_inner_smul_left, real_inner_smul_right, recallRotaryCode_inner]
  ring

example : (1 / 100000 : ℝ) ≤ 2 := by norm_num

/-- Actual clipped QKNorm multiplies the rotated code by one half on the stated epsilon range.
Source: the real normL2 operator and the proved exact norm; no postulated normalization divisor. -/
theorem recallRotaryCode_normL2 (eps : ℝ) (heps : eps ≤ 2)
    (digits : Fin 4 → Fin 4) (position : ℝ) :
    normL2 eps (applyRope 16 10000 position (recallRotaryCode digits)) =
      (1 / 2 : ℝ) • applyRope 16 10000 position (recallRotaryCode digits) := by
  rw [normL2, recallRotaryCode_rotated_norm, max_eq_left heps]

example : (1 / 100000 : ℝ) ≤ 2 := by norm_num

/-- The actual normalized compact matching code has exact unit norm.
Source: its verified clipped normalization and norm-two vector, useful for routing-error propagation. -/
theorem recallRotaryCode_normalized_norm (eps : ℝ) (heps : eps ≤ 2)
    (digits : Fin 4 → Fin 4) (position : ℝ) :
    ‖normL2 eps (applyRope 16 10000 position (recallRotaryCode digits))‖ = 1 := by
  rw [recallRotaryCode_normL2 eps heps, norm_smul, recallRotaryCode_rotated_norm]
  norm_num

example : (1 / 100000 : ℝ) ≤ 2 := by norm_num

end Transformer.GPTMini.Semantics
