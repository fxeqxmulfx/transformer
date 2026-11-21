module

public import Mathlib.Data.Nat.Bitwise
public import Mathlib.Data.Set.Function
public import Mathlib.Order.Interval.Set.Defs
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Shuffle

/-!
# PT6: marking the first entries of the shuffle

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. To read the training
half of E6 as a set, the kernel walks over the first entries of the shuffled array and collects them
in a bit mask (`mark`). An entry that is out of range or seen before stops the count. So if the
count reaches `n`, the first `n` entries are different cells and the mask holds exactly them
(`mark_spec`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

/-- One step on `(entries counted, mask of their values, array)`. -/
def mark (s : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  if 9409 ≤ get s.2.2 s.1 ∨ s.2.1.testBit (get s.2.2 s.1) = true then s
  else (s.1 + 1, s.2.1 ||| 1 <<< get s.2.2 s.1, s.2.2)

lemma mark_array (s : ℕ × ℕ × ℕ) : (mark s).2.2 = s.2.2 := by
  unfold mark
  split_ifs <;> rfl

lemma mark_count_le (s : ℕ × ℕ × ℕ) : (mark s).1 ≤ s.1 + 1 := by
  unfold mark
  split_ifs <;> simp

/-- A bit of a mask with one more value. -/
lemma testBit_or_shift (T c d : ℕ) :
    (T ||| 1 <<< c).testBit d = true ↔ T.testBit d = true ∨ c = d := by
  simp [Nat.testBit_or, Nat.one_shiftLeft, Nat.testBit_two_pow]

/-- **What the count says**: if it reaches `n`, the first `n` entries are different numbers below
9409, and the mask holds exactly them. -/
theorem mark_spec (A n : ℕ) (hn : (mark^[n] (0, 0, A)).1 = n) :
    Set.InjOn (get A) (Set.Iio n) ∧ (∀ i < n, get A i < 9409) ∧
      ∀ c, (mark^[n] (0, 0, A)).2.1.testBit c = true ↔ ∃ i < n, get A i = c := by
  have hA (k : ℕ) : (mark^[k] (0, 0, A)).2.2 = A := by
    induction k with
    | zero => rfl
    | succ k ih => rw [Function.iterate_succ_apply', mark_array, ih]
  have hle (k : ℕ) : (mark^[k] (0, 0, A)).1 ≤ k := by
    induction k with
    | zero => exact le_rfl
    | succ k ih =>
      rw [Function.iterate_succ_apply']
      exact (mark_count_le _).trans (Nat.succ_le_succ ih)
  induction n with
  | zero =>
    refine ⟨fun i hi => absurd hi (Nat.not_lt_zero i), fun i hi => absurd hi (Nat.not_lt_zero i),
      fun c => ?_⟩
    simp
  | succ n ih =>
    rw [Function.iterate_succ_apply'] at hn ⊢
    have hsA := hA n
    have hsle := hle n
    generalize mark^[n] (0, 0, A) = s at hn hsA hsle ih ⊢
    obtain ⟨i₀, T, A'⟩ := s
    simp only at hsA hsle ih
    subst hsA
    by_cases hb : 9409 ≤ get A' i₀ ∨ T.testBit (get A' i₀) = true
    · have hm : mark (i₀, T, A') = (i₀, T, A') := ite_eq_left hb
      rw [hm] at hn
      simp only at hn
      omega
    · have hm : mark (i₀, T, A') = (i₀ + 1, T ||| 1 <<< get A' i₀, A') := ite_eq_right hb
      rw [hm] at hn ⊢
      simp only at hn ⊢
      obtain rfl : i₀ = n := by omega
      obtain ⟨hinj, hlt, hT⟩ := ih rfl
      obtain ⟨hc, hbit⟩ := not_or.1 hb
      have hfresh (i : ℕ) (hi : i < i₀) : get A' i ≠ get A' i₀ :=
        fun h => hbit ((hT _).2 ⟨i, hi, h⟩)
      refine ⟨?_, ?_, fun c => ?_⟩
      · intro i hi j hj hij
        have hi' : i < i₀ + 1 := hi
        have hj' : j < i₀ + 1 := hj
        rcases Nat.lt_succ_iff_lt_or_eq.1 hi' with hi'' | rfl
        · rcases Nat.lt_succ_iff_lt_or_eq.1 hj' with hj'' | rfl
          · exact hinj hi'' hj'' hij
          · exact absurd hij (hfresh i hi'')
        · rcases Nat.lt_succ_iff_lt_or_eq.1 hj' with hj'' | rfl
          · exact absurd hij.symm (hfresh j hj'')
          · rfl
      · intro i hi
        rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi' | rfl
        · exact hlt i hi'
        · omega
      · rw [testBit_or_shift, hT]
        constructor
        · rintro (⟨i, hi, rfl⟩ | rfl)
          · exact ⟨i, by omega, rfl⟩
          · exact ⟨i₀, by omega, rfl⟩
        · rintro ⟨i, hi, rfl⟩
          rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi' | rfl
          · exact Or.inl ⟨i, hi', rfl⟩
          · exact Or.inr rfl

example : (mark^[0] (0, 0, 0)).1 = 0 ∧ Set.InjOn (get 0) (Set.Iio 0) :=
  ⟨rfl, (mark_spec 0 0 rfl).1⟩

end Transformer.Grokking.FlatSpectrum.TablePart.E6
