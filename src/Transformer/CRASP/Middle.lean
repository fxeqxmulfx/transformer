/-
# Depth-0 formulas in the middle of an affix restriction

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.2 and Appendix C.1, the proof of `lem:TLCP_commutative`.

In a string `λ(n⃗) u ϱ(n⃗)` with Parikh vector `n⃗`, the prefixes that end inside
`u` have their Parikh vectors in the middle `[ℙ(λ(n⃗)), n⃗ − ℙ(ϱ(n⃗))]`
(`prefixVec_mem_middle`).  A depth-0 formula whose PNPs are constant on the
middle therefore reads there the letter alone: at a position of `u` it holds
exactly where it holds at the first occurrence of that letter in `u`
(`Form.sat_middle`).  Its count over the positions of `u` is then a count over
the letters of `u` (`countP_range'_eq_countP`), and two middles with the same
Parikh vector are permutations of each other (`perm_of_parikh_eq`).
-/

import Transformer.CRASP.BoundedExists
import Transformer.CRASP.DepthZero

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A count over the positions `[1, |u|]` of a predicate that reads only the
letter at each position is a count over the letters of `u` (Appendix C.1,
proof of `lem:TLCP_commutative`, where the paper permutes the positions
instead). -/
theorem countP_range'_eq_countP {α : Type*} (G : α → Bool) :
    ∀ (u : List α) (F : ℕ → Bool), (∀ j a, 1 ≤ j → u[j - 1]? = some a → F j = G a) →
      (List.range' 1 u.length).countP F = u.countP G
  | [], _, _ => rfl
  | a :: l, F, h => by
      rw [List.length_cons, Nat.add_comm, countP_range'_add F 1 l.length,
        countP_range'_eq_countP G l (fun j => F (1 + j)) fun j b hj hb => ?_]
      · rw [List.range'_one, List.countP_singleton, h 1 a le_rfl rfl, Nat.add_comm, List.countP_cons]
      · obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
        exact h (1 + (j + 1)) b (by omega) (by simpa [Nat.add_comm 1] using hb)

/-- The hypothesis of `countP_range'_eq_countP` is satisfiable: the letter
read at each position. -/
example (u : List Bool) :
    ∀ j a, 1 ≤ j → u[j - 1]? = some a → (fun j => u[j - 1]?.getD false) j = id a :=
  fun _ _ _ h => by simp [h]

/-- The last letter of `l ++ s` is the last letter of `s` when `s` is not
empty (Appendix C.1, proof of `lem:TLCP_commutative`: with `s = ϱ(n⃗)`, the last
position lies in the suffix, "and `π(n) = n`"). -/
theorem getElem?_length_sub_one_append {α : Type*} (l : List α) {s : List α}
    (hs : 1 ≤ s.length) : (l ++ s)[(l ++ s).length - 1]? = s[s.length - 1]? := by
  rw [List.getElem?_append_right (by simp only [List.length_append]; omega), List.length_append,
    show l.length + s.length - 1 - l.length = s.length - 1 by omega]

/-- The hypothesis of `getElem?_length_sub_one_append` is satisfiable: a
one-letter suffix. -/
example (a : σ) : 1 ≤ [a].length := le_rfl

variable [DecidableEq σ]

/-- The Parikh vector of a concatenation is the sum of the Parikh vectors
(§2.3, Definition `def:Parikh_map`). -/
theorem parikh_append (u v : List σ) : parikh (u ++ v) = parikh u + parikh v := by
  funext a
  simp only [parikh, List.filter_append, List.length_append, Pi.add_apply]

/-- Strings with the same Parikh vector are permutations of each other
(§2.3, Definition `def:Parikh_map`). -/
theorem perm_of_parikh_eq {u u' : List σ} (h : parikh u = parikh u') : u.Perm u' :=
  List.perm_iff_count.mpr fun a => by
    rw [List.count_eq_countP, List.count_eq_countP, List.countP_eq_length_filter,
      List.countP_eq_length_filter]
    exact congrFun h a

/-- The hypothesis of `perm_of_parikh_eq` is satisfiable: `ab` and `ba`. -/
example : parikh [true, false] = parikh [false, true] := by
  funext a
  cases a <;> rfl

/-- Two strings `λ u ϱ` and `λ u' ϱ` with the same Parikh vector have middles
with the same Parikh vector (§2.3, Definition `def:Parikh_map`). -/
theorem parikh_eq_of_parikh_append_eq {p u u' s : List σ}
    (h : parikh (p ++ u ++ s) = parikh (p ++ u' ++ s)) : parikh u = parikh u' := by
  funext a
  have := congrFun h a
  simp only [parikh_append, Pi.add_apply] at this
  omega

/-- The hypothesis of `parikh_eq_of_parikh_append_eq` is satisfiable:
`a·ab·b` and `a·ba·b`. -/
example : parikh ([false] ++ [false, true] ++ [true]) = parikh ([false] ++ [true, false] ++ [true]) := by
  funext a
  cases a <;> rfl

/-- **The middle is where it should be.**  In `λ(n⃗) u ϱ(n⃗)` with Parikh
vector `n⃗`, the prefixes ending between `|λ(n⃗)|` and `|λ(n⃗)| + |u|` have
their Parikh vectors in the middle `[ℙ(λ(n⃗)), n⃗ − ℙ(ϱ(n⃗))]` (§4.2). -/
theorem prefixVec_mem_middle (A : Affix σ) {n : PVec σ} {u : List σ}
    (hn : parikh (A.pre n ++ u ++ A.suf n) = n) {j : ℕ} (hj : j ≤ u.length) :
    prefixVec (A.pre n ++ u ++ A.suf n) ((A.pre n).length + j) ∈ A.middle n := by
  have hv : ∀ a, parikh (A.pre n) a + parikh u a + parikh (A.suf n) a = n a := fun a => by
    have := congrFun hn a
    rwa [parikh_append, parikh_append] at this
  have ht : ∀ a, parikh (u.take j) a ≤ parikh u a := fun a =>
    ((List.take_sublist j u).filter _).length_le
  refine ⟨fun a => ?_, fun a => ?_⟩ <;>
    simp only [prefixVec, Affix.middle, List.append_assoc, List.take_length_add_append,
      List.take_append_of_le_length hj, parikh_append, Pi.add_apply]
  · omega
  · have := hv a
    have := ht a
    omega

/-- The hypotheses of `prefixVec_mem_middle` are satisfiable: the trivial
affix restriction, `u = ε` and `j = 0`. -/
example : parikh (((⟨fun _ => [], fun _ => []⟩ : Affix σ).pre (parikh [])) ++ [] ++
    (⟨fun _ => [], fun _ => []⟩ : Affix σ).suf (parikh [])) = parikh [] ∧ 0 ≤ ([] : List σ).length :=
  ⟨rfl, le_rfl⟩

/-- **A depth-0 formula in the middle reads the letter alone** (Appendix C.1,
the depth-0 cases of the proof of `lem:TLCP_commutative`).  Let `λ(n⃗) u ϱ(n⃗)`
and `λ(n⃗) v ϱ(n⃗)` both have Parikh vector `n⃗`, and let the PNPs of `χ` be
constant on the middle.  At a position of `v` that carries a letter `a` of
`u`, `χ` holds exactly where it holds at the first occurrence of `a` in `u`. -/
theorem Form.sat_middle (A : Affix σ) {n : PVec σ} {u v : List σ} {χ : Form σ}
    (hχ : χ.depth = 0) (hpnp : PnpsConstantOn χ A.middle)
    (hu : parikh (A.pre n ++ u ++ A.suf n) = n) (hv : parikh (A.pre n ++ v ++ A.suf n) = n)
    {j : ℕ} {a : σ} (hj : 1 ≤ j) (hja : v[j - 1]? = some a) (ha : a ∈ u) :
    χ.sat (A.pre n ++ v ++ A.suf n) ((A.pre n).length + j) =
      χ.sat (A.pre n ++ u ++ A.suf n) ((A.pre n).length + (u.idxOf a + 1)) := by
  have hjv : j ≤ v.length := by
    have := (List.getElem?_eq_some_iff.mp hja).1
    omega
  have hidx : u.idxOf a < u.length := List.idxOf_lt_length_of_mem ha
  refine Form.sat_eq_of_depth_eq_zero ?_ χ hχ fun ψ hψ =>
    hpnp ψ hψ n _ _ _ _ hv hu (by omega) (by simp only [List.length_append]; omega) (by omega)
      (by simp only [List.length_append]; omega) (prefixVec_mem_middle A hv hjv)
      (prefixVec_mem_middle A hu hidx)
  rw [List.append_assoc, List.append_assoc,
    show (A.pre n).length + j - 1 = (A.pre n).length + (j - 1) by omega,
    show (A.pre n).length + (u.idxOf a + 1) - 1 = (A.pre n).length + u.idxOf a by omega,
    List.getElem?_append_right (l₂ := v ++ A.suf n) (by omega),
    List.getElem?_append_right (l₂ := u ++ A.suf n) (by omega), Nat.add_sub_cancel_left,
    Nat.add_sub_cancel_left, List.getElem?_append_left (l₂ := A.suf n) (by omega),
    List.getElem?_append_left (l₂ := A.suf n) hidx, hja, List.getElem?_eq_getElem hidx,
    List.getElem_idxOf]

/-- The hypotheses of `Form.sat_middle` are satisfiable: `Q_a` under the
trivial affix restriction, at the only position of `a`. -/
example (a : σ) :
    (Form.sym a : Form σ).depth = 0 ∧
      PnpsConstantOn (Form.sym a : Form σ) (⟨fun _ => [], fun _ => []⟩ : Affix σ).middle ∧
      parikh ((⟨fun _ => [], fun _ => []⟩ : Affix σ).pre (parikh [a]) ++ [a] ++
        (⟨fun _ => [], fun _ => []⟩ : Affix σ).suf (parikh [a])) = parikh [a] ∧
      1 ≤ 1 ∧ [a][1 - 1]? = some a ∧ a ∈ [a] := by
  refine ⟨rfl, fun ψ hψ => ?_, rfl, le_rfl, rfl, List.mem_singleton_self a⟩
  rw [Form.pnps] at hψ
  exact absurd hψ List.not_mem_nil

end CRASP
end Transformer
