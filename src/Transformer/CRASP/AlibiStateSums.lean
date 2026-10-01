/-
# Integer sums of the actual ALiBi coefficients

arXiv:2506.16055v3, Appendix F, Equation `eq:alibi` and
`thm:rtfr_to_TLCly`. The future mask makes the real distance equal to the
natural difference, and fixed-point mantissas remove the common scale.
-/

import Transformer.CRASP.AlibiTables
import Transformer.CRASP.PeriodicAttention

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The exact integer coefficient sum over the BOS-inclusive prefix (F/B.1). -/
noncomputable def integerSum (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (q : Fin d → Fx p s) (o : Option (Fin d)) (w : List σ) (i : ℕ) : ℤ :=
  ∑ j ∈ Finset.Icc 0 i, (coefficient T a ℓ (q, o, T.actAt w ℓ j) (i - j)).m

/-- The denominator mantissa sum is nonnegative (Appendix F/B.1). -/
theorem integerSum_nonneg (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (q : Fin d → Fx p s) (w : List σ) (i : ℕ) :
    0 ≤ integerSum T a ℓ q none w i := by
  apply Finset.sum_nonneg
  intro j hj
  exact Fx.m_round_nonneg _ (by simp; positivity)

/-- The actual masked coefficient sum equals its natural-indexed integer sum.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:alibi`. -/
theorem sum_coefficient (T : PTfr (Option σ) p s d k) (a : ℝ) (hT : T.pe = .alibi a)
    (ℓ : ℕ) (w : List σ) (i : Fin (bos w).length) (q : Fin d → Fx p s)
    (hq : T.actAt w ℓ i.val = q) (o : Option (Fin d)) :
    (∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp
      (T.pe.logit i.val j.val (T.WQ ℓ (T.act (bos w) ℓ i))
        (T.WK ℓ (T.act (bos w) ℓ j))) *
      o.elim 1 (fun c => (T.WV ℓ (T.act (bos w) ℓ j) c).val))).val) =
        (integerSum T a ℓ q o w i.val : ℝ) / 2 ^ s := by
  have hqi : T.act (bos w) ℓ i = q := by rw [← PTfr.actAt_valid]; exact hq
  have hs : (∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp
      (T.pe.logit i.val j.val (T.WQ ℓ (T.act (bos w) ℓ i))
        (T.WK ℓ (T.act (bos w) ℓ j))) *
      o.elim 1 (fun c => (T.WV ℓ (T.act (bos w) ℓ j) c).val))).val) =
      ∑ j ∈ RTfr.masked i,
        (coefficient T a ℓ (q, o, T.actAt w ℓ j.val) (i.val - j.val)).val := by
    apply Finset.sum_congr rfl
    intro j hj
    have hji : j.val ≤ i.val := by
      simpa only [RTfr.masked, Finset.mem_filter, Finset.mem_univ, true_and,
        Fin.le_def] using hj
    simp only [coefficient, hT, PosEnc.logit, hqi, PTfr.actAt_valid, Nat.cast_sub hji]
  rw [hs, RTfr.sum_masked_nat i (fun j =>
    (coefficient T a ℓ (q, o, T.actAt w ℓ j) (i.val - j)).val)]
  have hi : Finset.Icc 0 i.val = insert 0 (Finset.Icc 1 i.val) := by
    ext j
    simp only [Finset.mem_Icc, Finset.mem_insert]
    omega
  have hz : (0 : ℕ) ∉ Finset.Icc 1 i.val := by simp
  unfold integerSum
  rw [hi, Finset.sum_insert hz]
  simp only [Fx.val, Int.cast_add, Int.cast_sum]
  rw [← Finset.sum_div]
  ring

/-- The fallback's unweighted source sum is also an integer count sum (F/B.1). -/
theorem sum_values (T : PTfr (Option σ) p s d k) (ℓ : ℕ) (w : List σ)
    (i : Fin (bos w).length) (c : Fin d) :
    (∑ j ∈ RTfr.masked i, (T.WV ℓ (T.act (bos w) ℓ j) c).val) =
      (CountCells.sum (T.actAt [] ℓ 0) (fun q => T.WV ℓ q c)
        (T.actAt w ℓ) i.val : ℝ) / 2 ^ s := by
  rw [T.sum_masked_actAt w ℓ i (fun q => (T.WV ℓ q c).val)]
  simp only [CountCells.sum, Fx.val, Int.cast_add, Int.cast_sum]
  rw [← Finset.sum_div]
  ring

/-- The encoding and query-state hypotheses have a zero-model witness (F). -/
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
