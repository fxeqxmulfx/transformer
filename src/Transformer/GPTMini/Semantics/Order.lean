import Transformer.GPTMini.Semantics.Presence
import Transformer.Basis.Depth
import Mathlib.Data.List.NodupEquivFin

/-!
# Ordered subsequences from internal prefix features

Source: arXiv:2506.16055v3, §2.4, the alternating-subsequence description
of E_k, and CausalMHA.forward at f11b6e2. A pattern followed by a letter
occurs exactly when some position holds that letter and its earlier
prefix contains the pattern. Thus faithful prefix-feature values let an
actual softmax head detect ordered subsequences, rather than token bags.

The query's final token differs from the detected letter, making its
self-value zero and preserving the probe through XSA. Prefix-feature
encoding remains a local representation obligation; no full GPTMini
parameter construction or assumed final logit correctness is used.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical

/-- An ordered pattern ends at a position whose strictly earlier prefix contains the remaining pattern.
Source: the new indexed form of §2.4's subsequence semantics, for any token alphabet. -/
theorem sublist_snoc_iff {α : Type*} (pattern word : List α) (letter : α) :
    (pattern ++ [letter]).Sublist word ↔
      ∃ j : Fin word.length, pattern.Sublist (word.take j.val) ∧ word.get j = letter := by
  constructor
  · intro h
    obtain ⟨f, hf⟩ := List.sublist_iff_exists_fin_orderEmbedding_get_eq.mp h
    let last : Fin (pattern ++ [letter]).length := ⟨pattern.length, by simp⟩
    let extend (t : Fin pattern.length) : Fin (pattern ++ [letter]).length :=
      ⟨t.val, by simp⟩
    let selected := f last
    have hearlier (t : Fin pattern.length) : (f (extend t)).val < selected.val := by
      exact f.strictMono (show extend t < last from t.isLt)
    let g : Fin pattern.length ↪o Fin (word.take selected.val).length :=
      OrderEmbedding.ofMapLEIff
        (fun t => ⟨(f (extend t)).val, by
          rw [List.length_take_of_le selected.isLt.le]
          exact hearlier t⟩)
        (by
          intro t u
          change (f (extend t)).val ≤ (f (extend u)).val ↔ t.val ≤ u.val
          exact f.le_iff_le)
    refine ⟨selected, List.sublist_iff_exists_fin_orderEmbedding_get_eq.mpr ⟨g, ?_⟩, ?_⟩
    · intro t
      have hprefix : (word.take selected.val).get (g t) = word.get (f (extend t)) := by
        have hbound : (f (extend t)).val < (word.take selected.val).length := by
          rw [List.length_take_of_le selected.isLt.le]
          exact hearlier t
        change (word.take selected.val)[(f (extend t)).val]'hbound =
          word[(f (extend t)).val]'(f (extend t)).isLt
        rw [List.getElem_take]
      rw [hprefix]
      have ht := hf (extend t)
      simpa only [extend, List.get_eq_getElem, List.getElem_append_left t.isLt] using ht
    · have ht := hf last
      have hletter : (pattern ++ [letter]).get last = letter := by
        simp only [last, List.get_eq_getElem, List.getElem_append_right (le_refl pattern.length),
          Nat.sub_self, List.getElem_cons_zero]
      rw [hletter] at ht
      exact ht.symm
  · rintro ⟨j, hprefix, hletter⟩
    have hstep : word.take (j.val + 1) = word.take j.val ++ [letter] := by
      rw [List.take_succ_eq_append_getElem j.isLt]
      change word.take j.val ++ [word.get j] = _
      rw [hletter]
    have h := hprefix.append (List.Sublist.refl [letter])
    rw [← hstep] at h
    exact h.trans (List.take_sublist _ _)

/-- A faithful prefix-feature value lets the actual complete head detect an ordered pattern.
Source: §2.4's subsequences and the original causal softmax/QKNorm/RoPE/XSA head at f11b6e2. -/
theorem head_ordered_pattern (cfg : Config) {α : Type*} [DecidableEq α] (word pattern : List α) (letter : α)
    (alpha eps : ℝ) (q k v : Fin word.length → EucSpace cfg.head_dim)
    (i : Fin word.length) (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hfinal : ∀ r : Fin word.length, r.val ≤ i.val) (hlast : word.get i ≠ letter)
    (hvalues : ∀ j, v j = if pattern.Sublist (word.take j.val) ∧ word.get j = letter then
      direction else 0) :
    0 < inner (𝕜 := ℝ)
        (attentionHead cfg alpha eps q k v (fun j => (j.val : ℝ)) i) direction ↔
      (pattern ++ [letter]).Sublist word := by
  let P := fun j : Fin word.length => pattern.Sublist (word.take j.val) ∧ word.get j = letter
  have hself : v i = 0 := by
    rw [hvalues i, ite_eq_right (fun h => hlast h.2)]
  have hvaluesP : ∀ j, v j = if P j then direction else 0 := by
    intro j
    by_cases hp : pattern.Sublist (word.take j.val) ∧ word.get j = letter
    · simpa only [P, ite_eq_left hp] using hvalues j
    · simpa only [P, ite_eq_right hp] using hvalues j
  rw [head_presence cfg alpha eps q k v _ i P direction hunit hvaluesP hself,
    sublist_snoc_iff]
  constructor
  · rintro ⟨j, hji, hp⟩
    exact ⟨j, hp⟩
  · rintro ⟨j, hp⟩
    exact ⟨j, hfinal j, hp⟩

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 5, j.val ≤ (4 : Fin 5).val) ∧
    ([1, 9, 10, 9, 10] : List ℤ).get ⟨4, by decide⟩ ≠ (9 : ℤ) ∧
    ∀ j : Fin 5,
      (if ([9, 10] : List ℤ).Sublist ([1, 9, 10, 9, 10].take j.val) ∧
          ([1, 9, 10, 9, 10] : List ℤ).get j = 9 then
        EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if ([9, 10] : List ℤ).Sublist ([1, 9, 10, 9, 10].take j.val) ∧
          ([1, 9, 10, 9, 10] : List ℤ).get j = 9 then
        EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0 := by
  refine ⟨by simp [PiLp.norm_single], ?_, by decide, fun _ => rfl⟩
  intro j
  have hj := j.isLt
  change j.val ≤ 4
  omega

/-- The real same-bag Basis depth pair gives different head outputs under faithful ordered-prefix features.
Source: Basis.depth_order_pair at cbafbe9; the extra aba pattern rejects abab but is absent in aabb.
The theorem applies to arbitrary Q/K arrays, not only an existential attention construction. -/
theorem depth_order_head_separation (cfg : Config) (alpha eps : ℝ)
    (qLeft kLeft vLeft qRight kRight vRight : Fin 5 → EucSpace cfg.head_dim)
    (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hleft : ∀ j : Fin 5, vLeft j =
      if ([9, 10] : List ℤ).Sublist ([1, 9, 9, 10, 10].take j.val) ∧
        ([1, 9, 9, 10, 10] : List ℤ).get j = 9 then direction else 0)
    (hright : ∀ j : Fin 5, vRight j =
      if ([9, 10] : List ℤ).Sublist ([1, 9, 10, 9, 10].take j.val) ∧
        ([1, 9, 10, 9, 10] : List ℤ).get j = 9 then direction else 0) :
    attentionHead cfg alpha eps qLeft kLeft vLeft (fun j => (j.val : ℝ)) 4 ≠
      attentionHead cfg alpha eps qRight kRight vRight (fun j => (j.val : ℝ)) 4 := by
  have hfinal (r : Fin 5) : r.val ≤ (4 : Fin 5).val := by
    have hr := r.isLt
    change r.val ≤ 4
    omega
  have hl := head_ordered_pattern cfg ([1, 9, 9, 10, 10] : List ℤ) [9, 10] 9
    alpha eps qLeft kLeft vLeft 4 direction hunit hfinal (by decide) hleft
  have hr := head_ordered_pattern cfg ([1, 9, 10, 9, 10] : List ℤ) [9, 10] 9
    alpha eps qRight kRight vRight 4 direction hunit hfinal (by decide) hright
  have hpositive : 0 < inner (𝕜 := ℝ)
      (attentionHead cfg alpha eps qRight kRight vRight (fun j => (j.val : ℝ)) (4 : Fin 5))
      direction := by simpa only [List.length_cons, List.length_nil] using hr.mpr (by decide)
  intro heq
  have hleftPositive : 0 < inner (𝕜 := ℝ)
      (attentionHead cfg alpha eps qLeft kLeft vLeft (fun j => (j.val : ℝ)) (4 : Fin 5))
      direction := by simpa only [heq] using hpositive
  have hsub := hl.mp hleftPositive
  exact (by decide : ¬([9, 10, 9] : List ℤ).Sublist [1, 9, 9, 10, 10]) hsub

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 5, (if ([9, 10] : List ℤ).Sublist ([1, 9, 9, 10, 10].take j.val) ∧
        ([1, 9, 9, 10, 10] : List ℤ).get j = 9 then
      EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if ([9, 10] : List ℤ).Sublist ([1, 9, 9, 10, 10].take j.val) ∧
        ([1, 9, 9, 10, 10] : List ℤ).get j = 9 then
      EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) ∧
    ∀ j : Fin 5, (if ([9, 10] : List ℤ).Sublist ([1, 9, 10, 9, 10].take j.val) ∧
        ([1, 9, 10, 9, 10] : List ℤ).get j = 9 then
      EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if ([9, 10] : List ℤ).Sublist ([1, 9, 10, 9, 10].take j.val) ∧
        ([1, 9, 10, 9, 10] : List ℤ).get j = 9 then
      EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0 := by
  exact ⟨by simp [PiLp.norm_single], fun _ => rfl, fun _ => rfl⟩

end Transformer.GPTMini.Semantics
