module

public import Transformer.SequenceModel.Attention.Finite.Separation
public import Transformer.SequenceModel.Attention.Expressivity.Layout

/-!
# Ordered finite-sequence codes after own-value subtraction

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F4.
One vocabulary token is a fixed readout token, appended after the operands.
XSA removes its value coordinate. The remaining weighted token counts still
recover every operand position when the positional subset sums are distinct.
The readout token is reused from the vocabulary, not an added label.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

variable {N p d : ℕ}

/-- Position-sensitive context in a bank disjoint from the token residual. -/
def contextCode (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p) (w : EucSpace N)
    (x : Fin N → Fin p) : EucSpace d :=
  ∑ j, if x j = zero then 0 else w j • unit (e (.inr (x j)))

/-- F4: the separating measurement is precisely the subset sum. -/
theorem inner_bits (w : EucSpace N) (b : Fin N → Bool) :
    ⟪w, bits N b⟫ = ∑ j, if b j then w j else 0 := by
  change (∑ j, (if b j then (1 : ℝ) else 0) * w j) = _
  apply Finset.sum_congr rfl
  intro j _
  cases b j <;> simp

/-- F4: every retained token coordinate is its weighted positional subset. -/
theorem context_measure (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero q : Fin p) (hq : q ≠ zero)
    (w : EucSpace N) (x : Fin N → Fin p) :
    ⟪unit (e (.inr q)), contextCode e zero w x⟫ =
      ⟪w, bits N (fun j => decide (x j = q))⟫ := by
  rw [contextCode, inner_sum, inner_bits]
  apply Finset.sum_congr rfl
  intro j _
  by_cases h : x j = zero
  · simp [h, Ne.symm hq]
  · rw [ite_eq_right h, real_inner_smul_right, inner_unit]
    simp only [e.injective.eq_iff, Sum.inr.injEq, decide_eq_true_eq]
    by_cases he : x j = q
    · simp only [he, ↓reduceIte, mul_one]
    · simp only [he, Ne.symm he, ↓reduceIte, mul_zero]

/-- F4: the context has zero coordinates in the entire residual token bank. -/
theorem context_orthogonal (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero q : Fin p)
    (w : EucSpace N) (x : Fin N → Fin p) :
    ⟪unit (e (.inl q)), contextCode e zero w x⟫ = 0 := by
  rw [contextCode, inner_sum]
  apply Finset.sum_eq_zero
  intro j _
  split_ifs <;> simp [real_inner_smul_right, inner_unit, e.injective.eq_iff]

/-- F4: removing the readout token's own value still leaves a lossless
ordered-sequence code, for every finite arity and vocabulary size. -/
theorem context_injective (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p) (w : EucSpace N)
    (hw : Function.Injective (fun b : Fin N → Bool => ⟪w, bits N b⟫)) :
    Function.Injective (contextCode e zero w) := by
  intro x y hxy
  have hb (q : Fin p) (hq : q ≠ zero) :
      (fun j => decide (x j = q)) = (fun j => decide (y j = q)) := by
    apply hw
    exact (context_measure e zero q hq w x).symm.trans
      ((congrArg (fun v => ⟪unit (e (.inr q)), v⟫) hxy).trans (context_measure e zero q hq w y))
  funext j
  by_cases hx : x j = zero
  · by_cases hy : y j = zero
    · exact hx.trans hy.symm
    · have h := congrFun (hb (y j) hy) j
      have hn : x j ≠ y j := by intro he; exact hy (he.symm.trans hx)
      simp [hn] at h
  · have h := congrFun (hb (x j) hx) j
    have he : y j = x j := by simpa using h.symm
    exact he.symm

/-- F4: strictly positive coefficients in any open interval can encode
every finite ordered token sequence after XSA's own-value deletion. -/
theorem exists_bounded_context (e : (Fin p ⊕ Fin p) ↪ Fin d) (zero : Fin p)
    (N : ℕ) {lo hi : ℝ} (h : lo < hi) :
    ∃ w : EucSpace N, (∀ j, lo < w j ∧ w j < hi) ∧
      Function.Injective (contextCode e zero w) := by
  obtain ⟨w, hbound, hinj⟩ := exists_bounded_subset_weights N h
  exact ⟨w, hbound, context_injective e zero w hinj⟩

private def banks : (Fin 2 ⊕ Fin 2) ↪ Fin 4 := finSumFinEquiv.toEmbedding
example := context_measure banks 0 1 (by decide) (WithLp.toLp 2 ![1, 3, 9]) ![1, 0, 1]
example := exists_bounded_context banks 0 7 (lo := 1) (hi := 1001 / 1000) (by norm_num)
example := context_injective banks 0 (WithLp.toLp 2 ![1, 3])
  (by
    intro a b h
    funext j
    simp only [inner_bits, Fin.sum_univ_two] at h
    fin_cases j <;> cases h₀ : a 0 <;> cases h₁ : a 1 <;>
      cases h₂ : b 0 <;> cases h₃ : b 1 <;> simp_all <;> norm_num at h)

end Transformer.SequenceModel.Attention.Finite
