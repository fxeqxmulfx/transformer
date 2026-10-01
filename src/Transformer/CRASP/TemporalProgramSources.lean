/-
# The exact numerator of a comparison average

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The BOS source contributes the constant and every ordinary source
contributes the truth values beneath the comparison's counts.
-/

import Transformer.CRASP.TemporalProgramCorrect

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- Summing the integer source values gives the difference of the compared terms.
Source: arXiv:2506.16055v3, Appendix B.2, uniform-attention construction. -/
theorem sum_sources (φ : Form σ) (k ℓ : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true)
    (hcorrect : Correct φ k ℓ) (t u : Term σ) (hm : Form.lt t u ∈ φ.subformulas)
    (hd : (Form.lt t u).depth ≤ ℓ + 1) (w : List σ) (i : ℕ) (hi : i ≤ w.length) :
    source φ (state φ k [] ℓ 0) ⟨.lt t u, hm⟩ +
      ∑ j ∈ Finset.Icc 1 i, source φ (state φ k w ℓ j) ⟨.lt t u, hm⟩ =
        (t.val (view w i) i : ℤ) - u.val (view w i) i := by
  have hptu : t.past = true ∧ u.past = true := by
    simpa only [Form.past, Bool.and_eq_true] using
      (φ.subformulas_properties hp hf _ hm).1
  have hbos : readBos φ (state φ k [] ℓ 0) = true := by
    rw [readBos, state_bos φ k [] ℓ 0 (by simp), Fx.read_ofBool]
    rfl
  have hordinary : ∀ j ∈ Finset.Icc 1 i, readBos φ (state φ k w ℓ j) = false := by
    intro j hj
    have hji := Finset.mem_Icc.mp hj
    rw [readBos, state_bos φ k w ℓ j (by omega), Fx.read_ofBool]
    simp [show j ≠ 0 by omega]
  have hbody : ∀ ψ ∈ t.countBodies ++ u.countBodies,
      ψ ∈ φ.subformulas ∧ ψ.depth ≤ ℓ := by
    intro ψ hψ
    rcases List.mem_append.mp hψ with hψ | hψ
    · have hmem := t.countBody_mem_subformulas ψ hψ
      refine ⟨φ.subformulas_trans _ _ hm
        (List.mem_cons_of_mem _ (List.mem_append_left _ hmem)), ?_⟩
      have hb := t.countBody_depth ψ hψ
      have ht : t.depth ≤ ℓ + 1 := (le_max_left _ _).trans hd
      omega
    · have hmem := u.countBody_mem_subformulas ψ hψ
      refine ⟨φ.subformulas_trans _ _ hm
        (List.mem_cons_of_mem _ (List.mem_append_right _ hmem)), ?_⟩
      have hb := u.countBody_depth ψ hψ
      have hu : u.depth ≤ ℓ + 1 := (le_max_right _ _).trans hd
      omega
  have hbits : ∀ ψ ∈ t.countBodies ++ u.countBodies, ∀ j ∈ Finset.Icc 1 i,
      read φ (state φ k w ℓ j) ψ = ψ.sat w j := by
    intro ψ hψ j hj
    have hjb := Finset.mem_Icc.mp hj
    obtain ⟨hmψ, hdψ⟩ := hbody ψ hψ
    have h := hcorrect w j (by omega) ⟨ψ, hmψ⟩ hdψ
    simpa only [view, show j ≠ 0 by omega, ite_false] using h
  have ht := sum_countP_eq_counts t.countBodies w i
    (fun j => read φ (state φ k w ℓ j))
    (fun ψ hψ => hbits ψ (List.mem_append_left _ hψ))
  have hu := sum_countP_eq_counts u.countBodies w i
    (fun j => read φ (state φ k w ℓ j))
    (fun ψ hψ => hbits ψ (List.mem_append_right _ hψ))
  have hs : ∀ j ∈ Finset.Icc 1 i, source φ (state φ k w ℓ j) ⟨.lt t u, hm⟩ =
      (t.countBodies.countP (read φ (state φ k w ℓ j)) : ℤ) -
        u.countBodies.countP (read φ (state φ k w ℓ j)) := by
    intro j hj
    change (if readBos φ (state φ k w ℓ j) then _ else _) = _
    simp only [hordinary j hj, Bool.false_eq_true, ite_false]
  have hb : source φ (state φ k [] ℓ 0) ⟨.lt t u, hm⟩ =
      (t.constant : ℤ) - u.constant := by
    change (if readBos φ (state φ k [] ℓ 0) then _ else _) = _
    simp only [hbos, ite_true]
  rw [hb, Finset.sum_congr rfl hs, Finset.sum_sub_distrib, ht, hu,
    ← term_val_view t hptu.1, ← term_val_view u hptu.2]
  have htv := t.val_countBodies hptu.1 w i
  have huv := u.val_countBodies hptu.2 w i
  omega

/-- The masked value-projection sum is exactly the compared term difference (B.2). -/
theorem sum_values (φ : Form σ) (k ℓ : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true)
    (hcorrect : Correct φ k ℓ) (t u : Term σ) (hm : Form.lt t u ∈ φ.subformulas)
    (hd : (Form.lt t u).depth ≤ ℓ + 1) (w : List σ) (i : Fin (bos w).length) :
    (∑ j ∈ RTfr.masked i,
      ((model φ k).WV ℓ ((model φ k).act (bos w) ℓ j)
        (coordinate φ (some (.inr ⟨.lt t u, hm⟩)))).val) =
      (t.val (view w i.val) i.val : ℝ) - u.val (view w i.val) i.val := by
  let ψ : Node φ := ⟨.lt t u, hm⟩
  have hv (H : Fin (dimension φ) → Fx (φ.capacity + 2) 0) :
      ((model φ k).WV ℓ H (coordinate φ (some (.inr ψ)))).val =
        (source φ (decode φ H) ψ : ℝ) := by
    change (encode φ (values φ (decode φ H)) (coordinate φ (some (.inr ψ)))).val = _
    simp only [encode, Equiv.symm_apply_apply]
    rw [Fx.val, m_values_scratch, pow_zero, div_one]
  have hsum : (∑ j ∈ RTfr.masked i,
      ((model φ k).WV ℓ ((model φ k).act (bos w) ℓ j)
        (coordinate φ (some (.inr ψ)))).val) =
      ∑ j ∈ RTfr.masked i, (source φ (decode φ ((model φ k).act (bos w) ℓ j)) ψ : ℝ) :=
    Finset.sum_congr rfl (fun j _ => hv ((model φ k).act (bos w) ℓ j))
  rw [hsum]
  rw [(model φ k).sum_masked_actAt w ℓ i (fun H => (source φ (decode φ H) ψ : ℝ))]
  change (source φ (state φ k [] ℓ 0) ψ : ℝ) +
    ∑ j ∈ Finset.Icc 1 i.val, (source φ (state φ k w ℓ j) ψ : ℝ) = _
  have hi : i.val ≤ w.length := by have := i.isLt; simp only [length_bos] at this; omega
  exact_mod_cast sum_sources φ k ℓ hp hf hcorrect t u hm hd w i.val hi

/-- The comparison and previous-layer hypotheses have a nonconstant witness (B.2). -/
example : Correct (Form.lt (.countL (.sym true)) .one) 1 0 ∧
    Form.lt (.countL (.sym true)) .one ∈
      (Form.lt (.countL (.sym true)) .one).subformulas ∧
    (Form.lt (.countL (.sym true)) .one).depth ≤ 0 + 1 ∧ 1 ≤ [true].length :=
  ⟨initial_correct _ _ rfl rfl, Form.mem_subformulas _, le_rfl, le_rfl⟩

end Transformer.CRASP.TemporalProgram
