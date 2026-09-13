/-
# RASP: the sorting program

Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like Transformers",
Figure 12 (Appendix, `sort`):

    def sort(vals, keys) {
        smaller = select(keys,keys,<) or
            (select(keys,keys,==) and select(indices,indices,<));
        num_smaller = selector_width(smaller);
        target_pos  = num_smaller;
        sel_new     = select(target_pos, indices, ==);
        sort        = aggregate(sel_new, vals);
    }

This is the program §4 leans on: sorting is realizable in RASP for arbitrary
length and alphabet, at a fixed number of heads and layers, hence "a standard
transformer can take full advantage of `Ω(n log n)` of the `n²` operations it
performs in every attention head".  The combinatorial core is that
`num_smaller` — the rank in the tie-broken order, index breaking ties — is a
*bijection* of the positions, so `sel_new` selects exactly one position in
every row and `aggregate` is a permutation of `vals`.

The lower bound §4 draws from this is a statement about restricted-attention
architectures, which are not modelled here; what is formalized is the program
and its correctness.
-/

import Transformer.RASP.Basic
import Mathlib.Data.Prod.Lex

namespace Transformer
namespace RASP

variable {n : ℕ} {α : Type*}

/-! ### Rank in a linear order

`selector_width(smaller)` is the rank of a position in a strict linear order
on the positions.  The two facts that make the program work — the rank is a
position, and ranking is injective — hold for any injective key map. -/

section Rank

variable {γ : Type*} [LinearOrder γ]

/-- The number of positions whose key is strictly smaller than `i`'s. -/
def rankOf (κ : Fin n → γ) (i : Fin n) : ℕ := (Finset.univ.filter fun j => κ j < κ i).card

lemma rankOf_lt_rankOf {κ : Fin n → γ} {i j : Fin n} (h : κ i < κ j) :
    rankOf κ i < rankOf κ j := by
  have hsub : (Finset.univ.filter fun l => κ l < κ i) ⊆ Finset.univ.filter fun l => κ l < κ j := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl ⊢
    exact hl.trans h
  refine Finset.card_lt_card ((Finset.ssubset_iff_of_subset hsub).2 ⟨i, ?_, ?_⟩) <;> simp [h]

lemma rankOf_lt (κ : Fin n → γ) (i : Fin n) : rankOf κ i < n := by
  have hsub : (Finset.univ.filter fun j => κ j < κ i) ⊆ Finset.univ.erase i := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
    simp only [Finset.mem_erase, Finset.mem_univ, and_true]
    rintro rfl
    exact absurd hl (lt_irrefl _)
  calc rankOf κ i ≤ (Finset.univ.erase i).card := Finset.card_le_card hsub
    _ = n - 1 := by
        rw [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin]
    _ < n := by have := i.isLt; omega

/-- The rank, as a position of the output sequence. -/
def rank (κ : Fin n → γ) (i : Fin n) : Fin n := ⟨rankOf κ i, rankOf_lt κ i⟩

lemma rank_lt_iff {κ : Fin n → γ} (hκ : Function.Injective κ) (i j : Fin n) :
    rank κ i < rank κ j ↔ κ i < κ j := by
  constructor
  · intro h
    rcases lt_trichotomy (κ i) (κ j) with hlt | heq | hgt
    · exact hlt
    · exact absurd (congrArg (fun z => rank κ z) (hκ heq)) h.ne
    · exact absurd (rankOf_lt_rankOf hgt) (by simpa [Fin.lt_def, rank] using h.le.not_gt)
  · intro h
    exact rankOf_lt_rankOf h

lemma rank_injective {κ : Fin n → γ} (hκ : Function.Injective κ) :
    Function.Injective (rank κ) := by
  intro i j hij
  rcases lt_trichotomy (κ i) (κ j) with h | h | h
  · exact absurd hij ((rank_lt_iff hκ i j).2 h).ne
  · exact hκ h
  · exact absurd hij.symm ((rank_lt_iff hκ j i).2 h).ne

/-- Ranking is a permutation of the positions: this is what makes `sel_new`
select exactly one position in each row. -/
noncomputable def rankEquiv (κ : Fin n → γ) (hκ : Function.Injective κ) : Fin n ≃ Fin n :=
  Equiv.ofBijective (rank κ) (Finite.injective_iff_bijective.mp (rank_injective hκ))

@[simp] lemma rank_rankEquiv_symm (κ : Fin n → γ) (hκ : Function.Injective κ) (i : Fin n) :
    rank κ ((rankEquiv κ hκ).symm i) = i :=
  Equiv.ofBijective_apply_symm_apply _ _ i

/-- Reading the input in rank order reads it in key order. -/
theorem strictMono_key_rankEquiv_symm (κ : Fin n → γ) (hκ : Function.Injective κ) :
    StrictMono fun i => κ ((rankEquiv κ hκ).symm i) := by
  intro i j hij
  refine (rank_lt_iff hκ _ _).1 ?_
  rwa [rank_rankEquiv_symm, rank_rankEquiv_symm]

end Rank

/-! ### The program -/

/-- The tie-broken key of a position: its key, with its index breaking ties,
i.e. the order `select(keys,keys,<) or (select(keys,keys,==) and
select(indices,indices,<))` compares by. -/
def lexKey (keys : Seq n ℝ) (i : Fin n) : ℝ ×ₗ Fin n := toLex (keys i, i)

lemma lexKey_injective (keys : Seq n ℝ) : Function.Injective (lexKey keys) := by
  intro i j hij
  have h : (keys i, i) = (keys j, j) := by simpa [lexKey] using hij
  exact (Prod.ext_iff.1 h).2

/-- `smaller = select(keys,keys,<) or (select(keys,keys,==) and
select(indices,indices,<))` (Figure 12). -/
noncomputable def smaller (keys : Seq n ℝ) : Selector n :=
  Selector.or' (sel keys keys (fun a b => decide (a < b)))
    (Selector.and' (sel keys keys (fun a b => decide (a = b)))
      (sel (indices n) (indices n) (fun a b => decide (a < b))))

lemma mem_selected_smaller (keys : Seq n ℝ) (i j : Fin n) :
    j ∈ selected (smaller keys) i ↔ lexKey keys j < lexKey keys i := by
  have hidx : (indices n j < indices n i) ↔ j < i := by
    rw [indices, indices, Nat.cast_lt, Fin.lt_def]
  simp [smaller, Selector.or', Selector.and', sel, lexKey, Prod.Lex.lt_iff, hidx]

/-- **`num_smaller` is the rank.**  `selector_width(smaller)` counts the
positions below `i` in the tie-broken order. -/
theorem selectorWidth_smaller (keys : Seq n ℝ) (i : Fin n) :
    selectorWidth (smaller keys) i = (rankOf (lexKey keys) i : ℕ) := by
  have hset : selected (smaller keys) i
      = Finset.univ.filter fun j => lexKey keys j < lexKey keys i := by
    ext j
    rw [mem_selected_smaller]
    simp
  rw [selectorWidth, hset, rankOf]

/-- `sel_new = select(target_pos, indices, ==)` with `target_pos =
num_smaller` (Figure 12, the `assume_bos = False` branch). -/
noncomputable def selNew (keys : Seq n ℝ) : Selector n :=
  sel (selectorWidth (smaller keys)) (indices n) (fun a b => decide (a = b))

lemma selected_selNew (keys : Seq n ℝ) (i : Fin n) :
    selected (selNew keys) i = {(rankEquiv (lexKey keys) (lexKey_injective keys)).symm i} := by
  ext j
  have hrank : (selectorWidth (smaller keys) j = indices n i)
      ↔ rank (lexKey keys) j = i := by
    rw [selectorWidth_smaller, indices]
    refine ⟨fun h => Fin.ext (Nat.cast_injective h), fun h => ?_⟩
    rw [← h]
    rfl
  simp only [mem_selected, selNew, sel, decide_eq_true_eq, Finset.mem_singleton, hrank]
  constructor
  · rintro rfl
    simp [rankEquiv]
  · rintro rfl
    exact rank_rankEquiv_symm _ _ i

/-- `sort = aggregate(sel_new, vals)` (Figure 12).  Each row of `sel_new`
selects exactly one position, so this is the token-transporting aggregation
of the footnote to §3. -/
noncomputable def sortProg (keys : Seq n ℝ) (vals : Seq n α) (d : α) : Seq n α :=
  aggregateOne (selNew keys) vals d

/-- **The program permutes the input.**  Position `i` of the output carries
the value of the position whose rank is `i`. -/
theorem sortProg_apply (keys : Seq n ℝ) (vals : Seq n α) (d : α) (i : Fin n) :
    sortProg keys vals d i
      = vals ((rankEquiv (lexKey keys) (lexKey_injective keys)).symm i) :=
  aggregateOne_of_selected_eq_singleton (selected_selNew keys i) vals d

/-- **And it sorts.**  The keys, read along the output, are non-decreasing:
strictly increasing in the tie-broken order, hence monotone in the keys
themselves. -/
theorem sortProg_keys_monotone (keys : Seq n ℝ) :
    Monotone fun i => keys ((rankEquiv (lexKey keys) (lexKey_injective keys)).symm i) := by
  intro i j hij
  rcases eq_or_lt_of_le hij with rfl | hlt
  · exact le_rfl
  · have := strictMono_key_rankEquiv_symm (lexKey keys) (lexKey_injective keys) hlt
    rcases Prod.Lex.lt_iff.1 this with h | h
    · exact h.le
    · exact h.1.le

/-- The hypotheses are satisfiable: on two positions with keys `[1, 0]`, the
program moves the value at position `1` to position `0`. -/
example :
    let keys : Seq 2 ℝ := ![1, 0]
    sortProg keys keys 0 0 = 0 ∧ sortProg keys keys 0 1 = 1 := by
  intro keys
  have h0 : rank (lexKey keys) 0 = 1 := by
    have : rankOf (lexKey keys) 0 = 1 := by
      have : (Finset.univ.filter fun j => lexKey keys j < lexKey keys 0) = {1} := by
        ext j
        fin_cases j <;> simp [lexKey, keys, Prod.Lex.lt_iff]
      rw [rankOf, this]
      simp
    exact Fin.ext this
  have h1 : rank (lexKey keys) 1 = 0 := by
    have : rankOf (lexKey keys) 1 = 0 := by
      have : (Finset.univ.filter fun j => lexKey keys j < lexKey keys 1) = ∅ := by
        ext j
        fin_cases j <;> simp [lexKey, keys, Prod.Lex.lt_iff]
      rw [rankOf, this]
      simp
    exact Fin.ext this
  have e0 : (rankEquiv (lexKey keys) (lexKey_injective keys)).symm 0 = 1 := by
    rw [← h1]
    simp [rankEquiv, Equiv.ofBijective_symm_apply_apply]
  have e1 : (rankEquiv (lexKey keys) (lexKey_injective keys)).symm 1 = 0 := by
    rw [← h0]
    simp [rankEquiv, Equiv.ofBijective_symm_apply_apply]
  refine ⟨?_, ?_⟩
  · rw [sortProg_apply, e0]; simp [keys]
  · rw [sortProg_apply, e1]; simp [keys]

end RASP
end Transformer
