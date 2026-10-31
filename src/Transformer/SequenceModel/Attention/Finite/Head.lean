module

public import Transformer.SequenceModel.Attention.Finite.Phases

/-!
# The actual normalized rotary XSA head produces the ordered code

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F5.
The causal denominator includes the readout token. Its contribution is then
removed by the source XSA operation, not by an attention mask. All earlier
tokens, including repeated readout labels, follow the same fixed matrices.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

variable {N p d : ℕ}

/-- Append one fixed vocabulary token as the readout position. -/
def withReadout (zero : Fin p) (x : Fin N → Fin p) : Fin (N + 1) → Fin p :=
  Fin.lastCases zero x

/-- The genuine softmax denominator, independent of the operand labels. -/
def codeDenominator (d : ℕ) (w : EucSpace N) : ℝ :=
  (∑ j, w j) + Real.exp (1 / Real.sqrt (d : ℝ))

/-- F5: strictly positive operand coefficients give a strictly positive
actual denominator, even for the zero-operand context. -/
theorem codeDenominator_pos (d : ℕ) (w : EucSpace N) (hw : ∀ j, 0 < w j) :
    0 < codeDenominator d w := by
  exact add_pos_of_nonneg_of_pos (Finset.sum_nonneg (fun j _ => (hw j).le)) (Real.exp_pos _)

/-- F5: exponentiated final-row scores are precisely the selected bounded
operand coefficients, plus the actual self-attention coefficient. -/
theorem last_exp_score (hd : 1 < d) {eps : ℝ} (heps : 0 < eps) (k : Fin (d / 2))
    (w : EucSpace N) (hw : ∀ j, 1 < w j ∧ w j < Real.exp (1 / Real.sqrt (d : ℝ)))
    (j : Fin (N + 1)) :
    Real.exp (headScore d eps (codePhases d w (Fin.last N)) (codePhases d w j)
      (unnormalize d eps (unit (ropeFst d k))) (unnormalize d eps (unit (ropeFst d k)))) =
      Fin.lastCases (Real.exp (1 / Real.sqrt (d : ℝ))) (fun j => w j) j := by
  refine Fin.lastCases ?_ (fun i => ?_) j
  · simp only [codePhases, Fin.lastCases_last, headScore_pair d hd heps k, Real.cos_zero]
  · simpa only [codePhases, Fin.lastCases_last, Fin.lastCases_castSucc] using
      exp_headScore_weight d hd heps k (w i) (hw i).1 (hw i).2

/-- F5: exact causal softmax weights, with the real self term retained. -/
theorem last_weight (hd : 1 < d) {eps : ℝ} (heps : 0 < eps) (k : Fin (d / 2))
    (w : EucSpace N) (hw : ∀ j, 1 < w j ∧ w j < Real.exp (1 / Real.sqrt (d : ℝ)))
    (j : Fin (N + 1)) :
    causalWeight (fun a b => headScore d eps (codePhases d w a) (codePhases d w b)
      (unnormalize d eps (unit (ropeFst d k))) (unnormalize d eps (unit (ropeFst d k))))
      (Fin.last N) j =
      Fin.lastCases (Real.exp (1 / Real.sqrt (d : ℝ))) (fun i => w i) j / codeDenominator d w := by
  have hI : Finset.Iic (Fin.last N) = Finset.univ := by ext i; simp [Fin.le_last]
  simp only [causalWeight, Fin.le_last, ↓reduceIte, hI, last_exp_score hd heps k w hw]
  rw [Fin.sum_univ_castSucc]
  simp only [Fin.lastCases_castSucc, Fin.lastCases_last, codeDenominator]

/-- Own-value subtraction followed by the fixed context-bank projection. -/
def removeOwn (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p) : EucSpace d →L[ℝ] EucSpace d :=
  (shift e).comp (ContinuousLinearMap.id ℝ _ -
    (innerSL ℝ (unit (e (.inl zero)))).smulRight (unit (e (.inl zero))))

/-- F5: this fixed linear map is exactly the actual unit-own-value XSA
subtraction followed by the source's output projection. -/
theorem removeOwn_apply (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p) (y : EucSpace d) :
    removeOwn e zero y = shift e (y - ⟪y, unit (e (.inl zero))⟫ • unit (e (.inl zero))) := by
  change shift e (y - ⟪unit (e (.inl zero)), y⟫ • unit (e (.inl zero))) = _
  rw [real_inner_comm y]

/-- F5: only the readout token's value is removed; every other token is copied. -/
theorem removeOwn_unit (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero a : Fin p) :
    removeOwn e zero (unit (e (.inl a))) = if a = zero then 0 else unit (e (.inr a)) := by
  rw [removeOwn_apply, inner_unit]
  simp only [e.injective.eq_iff, Sum.inl.injEq]
  by_cases h : a = zero
  · simp [h]
  · simp [h, shift_unit]

/-- F5: active nonzero rotary Q/K, causal softmax, clipped XSA and the
context projection produce the exact ordered-sequence code. The explicit
first-token zero is respected when the operand list is empty. -/
theorem shifted_head_code (hd : 1 < d) {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (k : Fin (d / 2)) (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p)
    (w : EucSpace N) (hw : ∀ j, 1 < w j ∧ w j < Real.exp (1 / Real.sqrt (d : ℝ)))
    (x : Fin N → Fin p) :
    shift e (headOutput eps xeps (codePhases d w)
      (fun _ => unnormalize d eps (unit (ropeFst d k)))
      (fun _ => unnormalize d eps (unit (ropeFst d k)))
      (fun j => unit (e (.inl (withReadout zero x j)))) (Fin.last N)) =
        (codeDenominator d w)⁻¹ • contextCode e zero w x := by
  by_cases hn : N = 0
  · subst N
    simp [headOutput, contextCode]
  · simp only [headOutput, withReadout, Fin.lastCases_last, Fin.val_last, hn,
      ↓reduceIte, normL2_unit _ xeps hx]
    rw [← removeOwn_apply, map_sum]
    simp only [map_smul, removeOwn_unit, last_weight hd heps k w hw]
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.lastCases_castSucc, Fin.lastCases_last, ↓reduceIte, smul_zero, add_zero]
    rw [contextCode, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    by_cases h : x j = zero <;> simp [h, smul_smul, div_eq_mul_inv, mul_comm]

example := codeDenominator_pos 4 (WithLp.toLp 2 ![1, 3, 9])
  (by intro j; fin_cases j <;> norm_num)

end Transformer.SequenceModel.Attention.Finite
