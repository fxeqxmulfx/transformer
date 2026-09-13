/-
# The sorted key list, derived rather than assumed

`Transformer.ALM.Hull` answers a lookup only for keys that arrive strictly
increasing: `hull_isGreatest` and `hull_bsearch_isGreatest` both take
`hstep : K j < K (j + 1)` as a hypothesis.  The machine gets that ordering
from its container — `_HullCHT` stores lines in a `std::set` ordered by slope
(`transformer_vm/attention/hull2d_cht.h`, lines 95-98) — and duplicate keys
collapse there, since two equal slopes keep one line (lines 132-139).

Taking the ordering as a hypothesis leaves a gap: nothing said that an
arbitrary family of keys *can* be presented that way.  This module closes it.
From any finite family `K : Fin n → ℝ` it builds `sortedKey K`, the distinct
values in increasing order, continued past the last one by unit steps so that
the sequence is strictly increasing on all of `ℕ` and the hull results apply
verbatim.  `exists_sortedKey_eq` says nothing is lost: every original key is
one of the first `keyCard K` entries.

What the container does in C++ — order and deduplicate — is here a function
with a proof, so `hstep` is no longer an assumption about the input but a
fact about the build.  `Transformer.ALM.HullIndex` uses it to make the hull
an index.
-/

import Transformer.ALM.BinSearch

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The distinct keys, in order -/

/-- The set of key values: what the `std::set` of lines holds after the
duplicates have collapsed. -/
noncomputable def keySet (K : Fin n → ℝ) : Finset ℝ := Finset.image K Finset.univ

/-- How many distinct keys there are. -/
noncomputable def keyCard (K : Fin n → ℝ) : ℕ := (keySet K).card

lemma mem_keySet (K : Fin n → ℝ) (i : Fin n) : K i ∈ keySet K :=
  Finset.mem_image.mpr ⟨i, Finset.mem_univ i, rfl⟩

lemma keyCard_pos [Nonempty (Fin n)] (K : Fin n → ℝ) : 0 < keyCard K :=
  Finset.card_pos.mpr ⟨K (Classical.arbitrary (Fin n)), mem_keySet K _⟩

lemma keyCard_le (K : Fin n → ℝ) : keyCard K ≤ n := by
  refine le_trans (Finset.card_image_le) ?_
  simp

/-- The distinct keys listed in increasing order. -/
noncomputable def keyEmb (K : Fin n → ℝ) : Fin (keyCard K) ↪o ℝ :=
  (keySet K).orderEmbOfFin rfl

/-- **The sorted key sequence.**  The distinct keys in increasing order, then
continued by unit steps, so that it is strictly increasing at every index and
the hull theorems apply without a bound on the range. -/
noncomputable def sortedKey [Nonempty (Fin n)] (K : Fin n → ℝ) (j : ℕ) : ℝ :=
  keyEmb K ⟨min j (keyCard K - 1),
      lt_of_le_of_lt (min_le_right _ _) (Nat.sub_lt (keyCard_pos K) one_pos)⟩
    + ((j - (keyCard K - 1) : ℕ) : ℝ)

/-- On the distinct keys themselves the continuation is inert. -/
lemma sortedKey_of_le [Nonempty (Fin n)] (K : Fin n → ℝ) {j : ℕ}
    (hj : j ≤ keyCard K - 1) :
    sortedKey K j = keyEmb K ⟨j, lt_of_le_of_lt hj (Nat.sub_lt (keyCard_pos K) one_pos)⟩ := by
  unfold sortedKey
  rw [Nat.sub_eq_zero_of_le hj]
  simp only [Nat.cast_zero, add_zero]
  exact congrArg _ (Fin.ext (min_eq_left hj))

/-- Past the last distinct key the sequence is that key plus the offset. -/
lemma sortedKey_of_ge [Nonempty (Fin n)] (K : Fin n → ℝ) {j : ℕ}
    (hj : keyCard K - 1 ≤ j) :
    sortedKey K j
      = keyEmb K ⟨keyCard K - 1, Nat.sub_lt (keyCard_pos K) one_pos⟩
        + ((j - (keyCard K - 1) : ℕ) : ℝ) := by
  unfold sortedKey
  exact congrArg (· + ((j - (keyCard K - 1) : ℕ) : ℝ))
    (congrArg (keyEmb K) (Fin.ext (min_eq_right hj)))

/-- **The build produces a strictly increasing sequence.**  This is the
hypothesis `hull_bsearch_isGreatest` asks for, now a theorem about the sorted
list rather than an assumption about the input. -/
theorem sortedKey_lt_succ [Nonempty (Fin n)] (K : Fin n → ℝ) (j : ℕ) :
    sortedKey K j < sortedKey K (j + 1) := by
  by_cases hj : j + 1 ≤ keyCard K - 1
  · rw [sortedKey_of_le K (le_trans (Nat.le_succ j) hj), sortedKey_of_le K hj]
    exact (keyEmb K).strictMono (by simp)
  · have hge : keyCard K - 1 ≤ j := by omega
    rw [sortedKey_of_ge K hge, sortedKey_of_ge K (le_trans hge (Nat.le_succ j))]
    have hstep : j + 1 - (keyCard K - 1) = (j - (keyCard K - 1)) + 1 := by omega
    rw [hstep]
    push_cast
    linarith

/-- **Nothing is lost.**  Every key of the original family appears among the
first `keyCard K` entries of the sorted sequence. -/
theorem exists_sortedKey_eq [Nonempty (Fin n)] (K : Fin n → ℝ) (i : Fin n) :
    ∃ j ≤ keyCard K - 1, sortedKey K j = K i := by
  have hmem : K i ∈ keySet K := mem_keySet K i
  obtain ⟨a, ha⟩ : ∃ a : Fin (keyCard K), keyEmb K a = K i := by
    have := (Finset.range_orderEmbOfFin (keySet K) (k := keyCard K) rfl)
    have hx : K i ∈ Set.range ((keySet K).orderEmbOfFin (k := keyCard K) rfl) := by
      rw [this]; exact hmem
    obtain ⟨a, ha⟩ := hx
    exact ⟨a, ha⟩
  refine ⟨a.1, by omega, ?_⟩
  rw [sortedKey_of_le K (by omega : a.1 ≤ keyCard K - 1)]
  simpa using ha

/-- And the sorted sequence introduces nothing new: its first `keyCard K`
entries are keys of the family. -/
theorem exists_eq_sortedKey [Nonempty (Fin n)] (K : Fin n → ℝ) {j : ℕ}
    (hj : j ≤ keyCard K - 1) : ∃ i : Fin n, K i = sortedKey K j := by
  rw [sortedKey_of_le K hj]
  have hmem := (keySet K).orderEmbOfFin_mem (k := keyCard K) rfl
    ⟨j, lt_of_le_of_lt hj (Nat.sub_lt (keyCard_pos K) one_pos)⟩
  obtain ⟨i, -, hi⟩ := Finset.mem_image.mp hmem
  exact ⟨i, hi⟩

/-- **The keys the search runs over are bounded.**  Finitely many sorted keys
have a largest absolute value, so the bound every floating-point statement in
this development assumes — `|sortedKey K j| ≤ B` on the searched window — is
never a restriction on the data: it holds for some `B` for every family.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 203-215 (the window the
search runs over). -/
theorem exists_bound_sortedKey [Nonempty (Fin n)] (K : Fin n → ℝ) :
    ∃ B : ℝ, ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ B := by
  obtain ⟨b, -, hb⟩ := Finset.exists_max_image (Finset.range (keyCard K))
    (fun j => |sortedKey K j|) ⟨0, Finset.mem_range.mpr (keyCard_pos K)⟩
  refine ⟨|sortedKey K b|, fun j hj => hb j (Finset.mem_range.mpr ?_)⟩
  have hpos := keyCard_pos K
  omega

/-- The construction is not vacuous: two keys, one of them repeated, sort to
the two distinct values. -/
example : keyCard (fun j : Fin 3 => (![1, 1, 2] : Fin 3 → ℝ) j) = 2 := by
  have h : keySet (fun j : Fin 3 => (![1, 1, 2] : Fin 3 → ℝ) j) = {1, 2} := by
    ext x
    constructor
    · intro hx
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hx
      fin_cases i <;> simp
    · intro hx
      rcases Finset.mem_insert.mp hx with rfl | hx
      · exact Finset.mem_image.mpr ⟨0, Finset.mem_univ _, by simp⟩
      · rw [Finset.mem_singleton.mp hx]
        exact Finset.mem_image.mpr ⟨2, Finset.mem_univ _, by simp⟩
  rw [keyCard, h]
  norm_num

end ALM
end Transformer
