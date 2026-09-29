/-
# Sequence-axis butterfly pairing as masked cyclic shifts

Arora et al., arXiv:2312.04927v1, Appendix `eq: butterfly-split` and
`prop: butterfly-hyena`. Toggling binary digit t adds 2^t in the upper
half of its block and subtracts 2^t in the lower half. These are exact
cyclic index identities, not an assumption about the convolution.
-/

import Transformer.Zoology.Appendix_ButterflyToggle

namespace Transformer.Zoology

/-- Whether the paired coordinate is in the upper half at digit t.
Source: Appendix `def: butterfly`, two-half block partition. -/
def butterflyToggleUpper : (k t : ℕ) → Fin (butterflyWidth k) → Bool
  | 0, _, _ => false
  | k + 1, t, i =>
      let p := finProdFinEquiv.symm i
      if t = k then decide (p.1 = 0) else butterflyToggleUpper k t p.2

/-- The partner differs by precisely one signed half-block offset.
Source: Appendix `eq: butterfly-split`, retained shifts. -/
theorem butterflyToggleIndex_val_delta (k t : ℕ) (ht : t < k)
    (i : Fin (butterflyWidth k)) :
    ((butterflyToggleIndex k t i).val : ℤ) = (i.val : ℤ) +
      if butterflyToggleUpper k t i then (butterflyWidth t : ℤ)
      else -(butterflyWidth t : ℤ) := by
  induction k generalizing t with
  | zero => omega
  | succ k ih =>
      have hj := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = p at hj
      rcases p with ⟨half, j⟩
      rw [← hj]
      by_cases htk : t = k
      · subst t
        fin_cases half <;>
          simp [butterflyToggleIndex, butterflyToggleUpper,
            finProdFinEquiv_apply_val]
      · have hsmall : t < k := by omega
        have hdelta := ih t hsmall j
        simp only [butterflyToggleIndex, butterflyToggleUpper,
          Equiv.symm_apply_apply, htk, ite_false]
        change ((finProdFinEquiv (half, butterflyToggleIndex k t j)).val : ℤ) =
          ((finProdFinEquiv (half, j)).val : ℤ) +
            if butterflyToggleUpper k t j then (butterflyWidth t : ℤ)
            else -(butterflyWidth t : ℤ)
        simp only [finProdFinEquiv_apply_val]
        push_cast
        rw [hdelta]
        ring

/-- The half-block offset fits in the full sequence.
Source: Appendix `eq: butterfly-split`, block size at most matrix size. -/
def butterflyToggleOffset (k : ℕ) (t : Fin k) : Fin (butterflyWidth k) :=
  ⟨butterflyWidth t.val, by
    rw [butterflyWidth_eq_pow, butterflyWidth_eq_pow]
    exact Nat.pow_lt_pow_right (by decide) t.isLt⟩

/-- Upper-half partners are read by a positive cyclic offset, and lower
half partners by the negative offset. Source: Appendix `eq: butterfly-split`. -/
theorem butterflyToggleIndex_shift (k : ℕ) (t : Fin k)
    (i : Fin (butterflyWidth k)) :
    butterflyToggleIndex k t.val i =
      if butterflyToggleUpper k t.val i then i + butterflyToggleOffset k t
      else i - butterflyToggleOffset k t := by
  have hdelta := butterflyToggleIndex_val_delta k t.val t.isLt i
  split_ifs with hu
  · simp [hu] at hdelta
    have hval : (butterflyToggleIndex k t.val i).val =
        i.val + butterflyWidth t.val := by exact_mod_cast hdelta
    have hlt : i.val + (butterflyToggleOffset k t).val < butterflyWidth k := by
      change i.val + butterflyWidth t.val < butterflyWidth k
      rw [← hval]
      exact (butterflyToggleIndex k t.val i).isLt
    apply Fin.ext
    rw [Fin.val_add_eq_of_add_lt hlt]
    exact hval
  · simp [hu] at hdelta
    have hle : butterflyToggleOffset k t ≤ i := by
      change butterflyWidth t.val ≤ i.val
      omega
    apply Fin.ext
    rw [Fin.sub_val_of_le hle]
    change (butterflyToggleIndex k t.val i).val = i.val - butterflyWidth t.val
    omega

/-- The shift lemma's nonempty-digit hypothesis has concrete instances.
Source: Appendix `def: butterfly`, a size-two factor. -/
example : (0 : ℕ) < 1 := by decide

end Transformer.Zoology
