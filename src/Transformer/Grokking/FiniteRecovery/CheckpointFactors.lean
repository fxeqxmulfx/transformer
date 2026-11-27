module

public import Transformer.Grokking.FiniteRecovery.CheckpointCertificate
public import Transformer.Grokking.FiniteRecovery.ArgandModularSum

/-! Completeness of the rank-one sign certificate for the additive binary
class. A successful model admits unit factors regardless of the singular
values of its phase table. The leading SVD provides one numerical proposal. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

private noncomputable def bitUnit (a : ZMod 2) : ℝ := if a = 0 then 1 else -1

private theorem bitUnit_add (a b : ZMod 2) : bitUnit (a+b) = bitUnit a * bitUnit b := by
  have cases' : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
  rcases cases' a with rfl | rfl <;> rcases cases' b with rfl | rfl <;> norm_num [bitUnit]
  decide

private theorem unit_abs (a : ZMod 2) : |bitUnit a| = 1 := by
  unfold bitUnit
  split <;> norm_num

private theorem unit_bit (a : ZMod 2) : negativeBit (bitUnit a) = a := by
  have cases' : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
  rcases cases' a with rfl | rfl <;> norm_num [bitUnit, negativeBit]

private theorem phase_near_label {T H d L : ℕ} (m : Model (ZMod 2) H d L) (r : ℕ)
    (target : Word (ZMod 2) (T+1) → ZMod 2) (h : m.realizes r target) (w : Word (ZMod 2) (T+1)) :
    |m.binaryPhase r w - bitUnit (target w)| < 1 := by
  let p := m.probability r w (Fin.last T)
  have pos (a : ZMod 2) : 0 < p a := m.probability_pos _ _ _ _
  have cap (a : ZMod 2) : p a ≤ 1 := by
    rw [← m.sum_probability r w (Fin.last T)]
    exact Finset.single_le_sum (fun t _ => (pos t).le) (Finset.mem_univ a)
  have hsum : 0 < ∑ a, Real.exp (m.logits r w (Fin.last T) a) :=
    Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨0, Finset.mem_univ _, Real.exp_pos _⟩
  have better (a : ZMod 2) (ha : a ≠ target w) : p a < p (target w) :=
    (div_lt_div_iff_of_pos_right hsum).mpr (Real.exp_lt_exp.mpr (h w a ha))
  have cases' : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  change |p 0 - p 1 - bitUnit (target w)| < 1
  rcases cases' (target w) with ht | ht
  · have hb := better 1 (by rw [ht]; decide)
    rw [ht] at hb ⊢
    rw [show bitUnit 0 = 1 by simp [bitUnit], abs_lt]
    constructor <;> linarith only [hb, pos 1, cap 0]
  · have hb := better 0 (by rw [ht]; decide)
    rw [ht] at hb ⊢
    rw [show bitUnit 1 = -1 by norm_num [bitUnit], abs_lt]
    constructor <;> linarith only [hb, pos 0, cap 1]

/-- Existence of factors passing the observed-sign and unlabeled-residual
tests is equivalent to strict classification on the entire domain. This
applies after any finite training procedure. The model can have full rank. -/
theorem realizes_iff_rank_one_checkpoint {s T H d L : ℕ}
    (cut : Fin (s+1) × Fin (s+1) ≃ Word (ZMod 2) (T+1)) (m : Model (ZMod 2) H d L) (r : ℕ)
    (row col : Fin (s+1) → ZMod 2) :
    m.realizes r (fun w => row (cut.symm w).1 + col (cut.symm w).2) ↔
    ∃ u v : Fin (s+1) → ℝ,
      (∀ i, negativeBit (u i * v i) = row i + col i) ∧
      (∀ i : Fin s, negativeBit (u i.castSucc * v i.succ) = row i.castSucc + col i.succ) ∧
      ∀ i j, |m.binaryPhase r (cut (i,j)) - u i * v j| < |u i * v j| := by
  constructor
  · intro h
    refine ⟨fun i => bitUnit (row i), fun j => bitUnit (col j), ?_, ?_, ?_⟩
    · intro i
      rw [← bitUnit_add, unit_bit]
    · intro i
      rw [← bitUnit_add, unit_bit]
    · intro i j
      rw [← bitUnit_add, unit_abs]
      simpa only [Equiv.symm_apply_apply] using phase_near_label m r _ h (cut (i,j))
  · rintro ⟨u,v,hd,he,hr⟩
    exact realizes_of_rank_one_checkpoint cut m r row col u v hd he hr

/-- The certificate has an actual six-layer Argand witness on the 31-cell
mask. These explicitly constructed weights are independent of the experiment. -/
example :=
  let e := wordEnumeration 2 4
  (realizes_iff_rank_one_checkpoint (s := 15) (T := 7) (enumeratedCut e)
    (Model.fromWeights (Parity.model 8).toWeights) 1 (fun u => ∑ i, e u i) (fun v => ∑ i, e v i)).mp
    (modularSum_native_realizes e (by decide))

end Transformer.Grokking.FiniteRecovery.Argand
