/-
# Sequence buffers for a sum of three shifted diagonal operators

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`
and `lmm: kaleido-coyote`. The printed butterfly proof drops shifts in
one display. A corrected construction keeps three shifted copies in
disjoint sequence buffers, preserving the feature dimension.
-/

import Transformer.Zoology.Appendix_CyclicCircuitNetwork

open scoped BigOperators

namespace Transformer.Zoology

/-- Zero-pad a sequence into a workspace nine times as long, preserving
its feature dimension. Source: Appendix `prop: butterfly-hyena`,
constant-factor inner-sequence expansion. -/
def shiftSumPad {m d : ℕ} (u : RealSequence m d) : RealSequence (9 * m) d :=
  fun j q => if h : j.val < m then u ⟨j.val, h⟩ q else 0

/-- The first-block position of an original token in the workspace.
Source: Appendix `prop: butterfly-hyena`, zero-padded layout. -/
def shiftSumInputSlot {m : ℕ} (i : Fin m) : Fin (9 * m) :=
  ⟨i.val, by omega⟩

/-- The shift that duplicates the input into the immediately following
block. Source: Appendix `prop: prim-add`, the filter `1+X^m`. -/
def shiftSumCopyOffset (m : ℕ) [NeZero m] : Fin (9 * m) :=
  ⟨m, by have hm := Nat.pos_of_ne_zero (NeZero.ne m); omega⟩

/-- The beginning of one of three nonoverlapping output buffers.
Source: Appendix `prop: butterfly-hyena`, corrected parallel branches. -/
def shiftSumTermBase {m : ℕ} [NeZero m] (a : Fin 3) : Fin (9 * m) :=
  ⟨(3 * a.val + 1) * m, by
    have hm := Nat.pos_of_ne_zero (NeZero.ne m)
    fin_cases a <;> norm_num <;> omega⟩

/-- A token in one of the three output buffers.
Source: Appendix `prop: butterfly-hyena`, corrected parallel branches. -/
def shiftSumTermSlot {m : ℕ} (a : Fin 3) (i : Fin m) : Fin (9 * m) :=
  ⟨(3 * a.val + 1) * m + i.val, by fin_cases a <;> norm_num <;> omega⟩

/-- The convolution tap that places a shifted duplicate into its own
buffer. Source: Appendix `prop: prim-shift` and `prop: butterfly-hyena`. -/
def shiftSumTermTap {m : ℕ} (a : Fin 3) (s : Fin m) : Fin (9 * m) :=
  ⟨3 * a.val * m + s.val, by fin_cases a <;> norm_num <;> omega⟩

/-- Distinct term/token pairs have distinct workspace positions.
Source: Appendix `prop: butterfly-hyena`, independent branches. -/
theorem shiftSumTermSlot_eq_iff {m : ℕ}
    (a b : Fin 3) (i j : Fin m) :
    shiftSumTermSlot a i = shiftSumTermSlot b j ↔ a = b ∧ i = j := by
  constructor
  · intro h
    have hv := congrArg Fin.val h
    dsimp [shiftSumTermSlot] at hv
    have hab : a = b := by
      fin_cases a <;> fin_cases b <;> norm_num at hv ⊢ <;>
        (first | rfl | omega)
    subst b
    exact ⟨rfl, Fin.ext (by omega)⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- A tap reads the appropriate entry of the duplicated input when
observed in its own output buffer. Source: Appendix `prop: prim-shift`. -/
theorem shiftSumTermTap_self_val {m : ℕ}
    (a : Fin 3) (i s : Fin m) :
    (shiftSumTermSlot a i - shiftSumTermTap a s).val =
      m + i.val - s.val := by
  have hle : shiftSumTermTap a s ≤ shiftSumTermSlot a i := by
    apply Fin.le_def.mpr
    fin_cases a <;> dsimp [shiftSumTermTap, shiftSumTermSlot] <;> omega
  rw [Fin.sub_val_of_le hle]
  fin_cases a <;> dsimp [shiftSumTermTap, shiftSumTermSlot] <;> omega

/-- A tap belonging to a different branch reads outside the two-block
duplicated input. Source: Appendix `prop: butterfly-hyena`, corrected
separation of the three shifted terms. -/
theorem shiftSumTermTap_other_val {m : ℕ}
    (a b : Fin 3) (i s : Fin m) (hab : a ≠ b) :
    2 * m ≤ (shiftSumTermSlot a i - shiftSumTermTap b s).val := by
  have heq := Fin.intCast_val_sub_eq_sub_add_ite
    (shiftSumTermSlot a i) (shiftSumTermTap b s)
  fin_cases a <;> fin_cases b <;>
    norm_num [shiftSumTermSlot, shiftSumTermTap, Fin.le_def] at heq ⊢
  all_goals first | exact (hab rfl).elim | (split_ifs at heq <;> omega)

/-- Shifting the output buffer back to the first block reads its matching
token. Source: Appendix `prop: prim-shift`, cyclic option. -/
theorem shiftSumTermSlot_read {m : ℕ} [NeZero m]
    (a : Fin 3) (i : Fin m) :
    shiftSumInputSlot i - (-shiftSumTermBase a) = shiftSumTermSlot a i := by
  simp only [Fin.sub_eq_add_neg]
  have hneg : -(-shiftSumTermBase (m := m) a) = shiftSumTermBase a := by simp
  rw [hneg]
  apply Fin.ext
  rw [Fin.val_add]
  have hlt : i.val + (3 * a.val + 1) * m < 9 * m := by
    fin_cases a <;> norm_num <;> omega
  simp only [shiftSumInputSlot, shiftSumTermBase, shiftSumTermSlot]
  rw [Nat.mod_eq_of_lt hlt]
  omega

/-- Store each term's coefficient at its own output-buffer coordinate.
Source: Appendix `prop: butterfly-hyena`, three diagonal multipliers. -/
def shiftSumCoefficient {m d : ℕ}
    (coeff : Fin 3 → RealSequence m d) : RealSequence (9 * m) d :=
  fun j q => ∑ a : Fin 3, ∑ i : Fin m,
    if j = shiftSumTermSlot a i then coeff a i q else 0

/-- At a branch's token, the position gate equals exactly that branch's
coefficient. Source: Appendix `prop: butterfly-hyena`, diagonal gating. -/
theorem shiftSumCoefficient_slot {m d : ℕ}
    (coeff : Fin 3 → RealSequence m d) (a : Fin 3) (i : Fin m) (q : Fin d) :
    shiftSumCoefficient coeff (shiftSumTermSlot a i) q = coeff a i q := by
  unfold shiftSumCoefficient
  rw [Finset.sum_eq_single a]
  · simp [shiftSumTermSlot_eq_iff]
  · intro b _ hb
    have hab : a ≠ b := Ne.symm hb
    simp [shiftSumTermSlot_eq_iff, hab]
  · intro h
    exact (h (Finset.mem_univ a)).elim

/-- Nonempty sequence and workspace hypotheses have a concrete instance.
Source: Appendix `prop: butterfly-hyena`, finite dimensions. -/
example : shiftSumTermSlot (m := 1) 0 0 ≠ shiftSumTermSlot 1 0 := by
  decide

end Transformer.Zoology
