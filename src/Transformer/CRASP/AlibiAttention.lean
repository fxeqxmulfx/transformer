/-
# Exact ALiBi attention from its integer coefficient sums

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`, and B.1,
Equation `eq:att`. The common mantissa scale cancels in weighted attention
and remains in the uniform denominator-zero fallback.
-/

import Transformer.CRASP.AlibiStateSums

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- Rounded attention expressed in integer units (Appendix F/B.1). -/
noncomputable def attentionValue (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (w : List σ) (i : ℕ) (q : Fin d → Fx p s) (c : Fin d) : Fx p s :=
  let D := integerSum T a ℓ q none w i
  let N := integerSum T a ℓ q (some c) w i
  let V := CountCells.sum (T.actAt [] ℓ 0) (fun q => T.WV ℓ q c) (T.actAt w ℓ) i
  if D = 0 then Fx.round p s (((V : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
  else Fx.round p s ((N : ℝ) / (D : ℝ))

/-- The integer-sum expression equals the actual rounded attention.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:alibi`, with the
zero-denominator fallback from Appendix B.1. -/
theorem attention_eq_value (T : PTfr (Option σ) p s d k) (a : ℝ) (hT : T.pe = .alibi a)
    (ℓ : ℕ) (w : List σ) (i : Fin (bos w).length) (q : Fin d → Fx p s)
    (hq : T.actAt w ℓ i.val = q) (c : Fin d) :
    PeriodicAttention.attention T ℓ (T.act (bos w) ℓ) i c =
      attentionValue T a ℓ w i.val q c := by
  let D := integerSum T a ℓ q none w i.val
  let N := integerSum T a ℓ q (some c) w i.val
  let V := CountCells.sum (T.actAt [] ℓ 0) (fun q => T.WV ℓ q c) (T.actAt w ℓ) i.val
  have hD := sum_coefficient T a hT ℓ w i q hq none
  have hN := sum_coefficient T a hT ℓ w i q hq (some c)
  have hV := sum_values T ℓ w i c
  simp only [Option.elim_none, mul_one] at hD
  simp only [Option.elim_some] at hN
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  simp only [PeriodicAttention.attention, hD, hN, hV, RTfr.card_masked]
  change (if (D : ℝ) / 2 ^ s = 0 then
      Fx.round p s (((V : ℝ) / 2 ^ s) / (i.val + 1 : ℕ))
    else Fx.round p s (((N : ℝ) / 2 ^ s) / ((D : ℝ) / 2 ^ s))) = _
  change _ = (if D = 0 then Fx.round p s (((V : ℝ) / (i.val + 1 : ℕ)) / 2 ^ s)
    else Fx.round p s ((N : ℝ) / (D : ℝ)))
  by_cases hz : D = 0
  · simp only [hz, Int.cast_zero, zero_div, ite_true]
    congr 1
    ring
  · have hzr : (D : ℝ) ≠ 0 := by exact_mod_cast hz
    simp only [hz, div_eq_zero_iff, hzr, hs, or_self, ite_false]
    congr 1
    field_simp

/-- Encoding and query-state hypotheses have a zero-dimensional witness (F). -/
example : ∃ T : PTfr (Option Bool) 2 0 0 0,
    T.pe = .alibi 1 ∧ T.actAt [true] 0 1 = (fun _ => 0) := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .alibi 1 }, rfl, ?_⟩
  funext c
  exact Fin.elim0 c

end Transformer.CRASP.AlibiTables
