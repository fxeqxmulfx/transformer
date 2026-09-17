/-
# The lower bound of the two-sided depth hierarchy

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D: the lower bound of `thm:TLC_depth`, proved without
`lem:reduction`.

The paper derives that `L_{2k+3}` needs depth `k + 2` in `TL[◁#,▷#]` from
`lem:cropping` and `lem:reduction`, both false as stated.  The argument here
keeps the cropping, with the positions around the interval fixed to a frame,
and adds one block on each side per level of depth in place of the reduction.

Formulas of `TL[◁#,▷#]_d` are all constant on the middles with `V` letters `a`
and `V` letters `b` of a frame `(u, v)` with `u` and `v` words of `L_{d+1}`
(`exists_constOnMiddle_altPlus`).  By induction, the formulas under their
counts are constant on the middles of a larger frame; shrinking gives a box on
which the formulas themselves are constant (`exists_constOnBox_list`); and
putting its corner `μ₁` after `u` and the rest `μ₂` of the Parikh vector before
`v` lengthens a block of each and adds one, and turns the box into the middles
of the new frame (`ConstOnBox.constOnMiddle`).  A formula of depth `d + 1`
then gives the same value at the last position of `u·s s̄ s̄ s·v`, a word of
`L_{2d+3}`, and of `u·s̄ s s s̄·v`, a word of `L_{2d+5}`
(`exists_models_iff_altPlusDouble`).
-/

import Transformer.CRASP.Blocks
import Transformer.CRASP.FrameShrink

namespace Transformer
namespace CRASP

/-- **A box of the middles of a frame is the set of middles of a smaller
frame.**  Putting the corner `μ₁` of the box after `u` and the rest `μ₂` of the
Parikh vector before `v`, a formula constant on the box, whose counted formulas
are constant on the middles, is constant on the middles of `(u μ₁, μ₂ v)`
(Appendix D, proof of `lem:cropping`). -/
theorem ConstOnBox.constOnMiddle {ψ : Form Bool} {u v μ₁ μ₂ : List Bool} {X Y V : ℕ}
    (hf : ψ.pnpFree = true) (hS : ∀ χ ∈ ψ.countSubs, ConstOnMiddle χ u v X Y)
    (h : ConstOnBox ψ u v X Y (μ₁.count false) (μ₁.count true) V)
    (hX : μ₁.count false + V + μ₂.count false = X) (hY : μ₁.count true + V + μ₂.count true = Y) :
    ConstOnMiddle ψ (u ++ μ₁) (μ₂ ++ v) V V := by
  obtain ⟨g, hg⟩ := h
  have hM : ∀ m : List Bool, m.count false = V → m.count true = V →
      (μ₁ ++ m ++ μ₂).count false = X ∧ (μ₁ ++ m ++ μ₂).count true = Y := fun m h₁ h₂ => by
    simp only [List.count_append]
    omega
  have hw : ∀ m : List Bool, u ++ μ₁ ++ m ++ (μ₂ ++ v) = u ++ (μ₁ ++ m ++ μ₂) ++ v := by simp
  refine ⟨⟨g, fun m h₁ h₂ i hi => ?_⟩, fun m m' h₁ h₂ h₁' h₂' j hj hj' => ?_⟩
  · have hl : μ₁.length + i < (μ₁ ++ m ++ μ₂).length := by
      simp only [List.length_append]
      omega
    have ht : (μ₁ ++ m ++ μ₂).take (μ₁.length + i) = μ₁ ++ m.take i := by
      rw [List.append_assoc, List.take_length_add_append, List.take_append_of_le_length hi.le]
    have hx := ((List.take_sublist i m).count_le false).trans h₁.le
    have hy := ((List.take_sublist i m).count_le true).trans h₂.le
    have e := hg _ (hM m h₁ h₂).1 (hM m h₁ h₂).2 (μ₁.length + i) hl
      (by rw [ht, List.count_append]; omega) (by rw [ht, List.count_append]; omega)
      (by rw [ht, List.count_append]; omega) (by rw [ht, List.count_append]; omega)
    have hc := getElem?_frame μ₁ m μ₂ hi
    rw [Nat.add_sub_cancel, List.getElem?_eq_getElem hl, Option.some.injEq] at hc
    rw [hw, List.length_append, Nat.add_assoc u.length, e, hc]
  · have hm := length_eq_of_count h₁ h₂
    have hm' := length_eq_of_count h₁' h₂'
    obtain ⟨hMX, hMY⟩ := hM m h₁ h₂
    obtain ⟨hMX', hMY'⟩ := hM m' h₁' h₂'
    rw [hw, hw]
    rw [List.length_append] at hj'
    by_cases hin : j ≤ u.length ∨ u.length + X + Y < j
    · exact Form.sat_eq_of_constOnMiddle hMX hMY hMX' hMY' hj hin ψ hf hS
    · have hl₁ := List.count_false_add_count_true μ₁
      obtain ⟨i, rfl⟩ : ∃ i, j = u.length + i + 1 := ⟨j - u.length - 1, by omega⟩
      have hi : i < (μ₁ ++ m ++ μ₂).length := by rw [length_eq_of_count hMX hMY]; omega
      have hi' : i < (μ₁ ++ m' ++ μ₂).length := by rw [length_eq_of_count hMX' hMY']; omega
      have hcount : ∀ c, ((μ₁ ++ m ++ μ₂).take i).count c = ((μ₁ ++ m' ++ μ₂).take i).count c := by
        intro c
        rcases Nat.lt_or_ge i μ₁.length with hi₁ | hi₁
        · rw [List.append_assoc, List.append_assoc, List.take_append_of_le_length hi₁.le,
            List.take_append_of_le_length hi₁.le]
        · obtain ⟨k, rfl⟩ : ∃ k, i = (μ₁ ++ m).length + k :=
            ⟨i - (μ₁ ++ m).length, by rw [List.length_append]; omega⟩
          rw [List.take_length_add_append,
            show (μ₁ ++ m).length + k = (μ₁ ++ m').length + k by simp only [List.length_append]; omega,
            List.take_length_add_append]
          simp only [List.count_append]
          cases c <;> omega
      have hc := getElem?_frame_outside μ₁ μ₂ (hm.trans hm'.symm) (j := i + 1) (by omega)
        (by omega)
      rw [Nat.add_sub_cancel, List.getElem?_eq_getElem hi, List.getElem?_eq_getElem hi',
        Option.some.injEq] at hc
      exact Form.sat_eq_of_take_count_eq hMX hMY hMX' hMY' hi hi' (hcount false) (hcount true) hc
        ψ hf hS

/-- The hypotheses of `ConstOnBox.constOnMiddle` are satisfiable: `Q_a` on the
frame `(ε, ε, 1, 1)`, with the box at `μ₁ = μ₂ = ε` of side `1`. -/
example : ConstOnBox (.sym false) [] [] 1 1 (([] : List Bool).count false)
      (([] : List Bool).count true) 1 ∧
    ([] : List Bool).count false + 1 + ([] : List Bool).count false = 1 := by
  obtain ⟨⟨g, hg⟩, -⟩ := Form.constOnMiddle_of_depth_eq_zero [] [] 1 1 (.sym false) rfl rfl
  exact ⟨⟨g, fun m hX hY i hi _ _ _ _ => hg m hX hY i hi⟩, rfl⟩

/-- Blocks of the first letter lengthen the first block (§2.4, Equation
`eq:altplus`). -/
theorem replicate_append_mem_altPlus {s : Bool} {k : ℕ} {v : List Bool}
    (h : v ∈ altPlus s (k + 1)) : ∀ b, List.replicate b s ++ v ∈ altPlus s (k + 1)
  | 0 => h
  | b + 1 => cons_mem_altPlus_same (replicate_append_mem_altPlus h b)

/-- The hypothesis of `replicate_append_mem_altPlus` is satisfiable: `a ∈ L_1`. -/
example : [false] ∈ altPlus false (0 + 1) := ⟨1, Nat.one_pos, [], rfl, rfl⟩

/-- **Formulas of `TL[◁#,▷#]_d` are constant on the middles of a frame of words
of `L_{d+1}`.**  For finitely many of them and every `V` there are
`u ∈ L_{d+1}` and `v`, of `d + 1` blocks ending with `a`, such that each is
constant on the middles of `(u, v, V, V)` (Appendix D, the induction behind
`lem:cropping`, with one new block on each side per level of depth in place of
`lem:reduction`). -/
theorem exists_constOnMiddle_altPlus :
    ∀ d (S : List (Form Bool)), (∀ ψ ∈ S, ψ ∈ TLC Bool d) → ∀ V,
      ∃ u ∈ altPlus false (d + 1), ∃ v ∈ altPlus (false ^^ decide (d % 2 = 1)) (d + 1),
        ∀ ψ ∈ S, ConstOnMiddle ψ u v V V
  | 0, _, hS, V => ⟨[false], ⟨1, Nat.one_pos, [], rfl, rfl⟩, [false], ⟨1, Nat.one_pos, [], rfl, rfl⟩,
      fun ψ hψ => Form.constOnMiddle_of_depth_eq_zero [false] [false] V V ψ (hS ψ hψ).1
        (Nat.le_zero.1 (hS ψ hψ).2)⟩
  | d + 1, S, hS, V => by
      obtain ⟨N, hN⟩ := exists_constOnBox_list S (fun ψ hψ => (hS ψ hψ).1) V
      obtain ⟨u, hu, v, hv, hS'⟩ := exists_constOnMiddle_altPlus d (S.flatMap Form.countSubs)
        (fun χ hχ => by
          obtain ⟨ψ, hψ, hχ⟩ := List.mem_flatMap.1 hχ
          exact Form.mem_TLC_of_mem_countSubs ψ (hS ψ hψ) hχ) (N + 2)
      obtain ⟨p, q, hp, hpV, hq, hqV, h⟩ := hN u v (N + 2) (N + 2) 1 1 fun ψ hψ χ hχ =>
        hS' χ (List.mem_flatMap.2 ⟨ψ, hψ, hχ⟩)
      have hmem := append_mem_altPlus d false u hu
      have e : (false ^^ decide ((d + 1) % 2 = 1)) = !(false ^^ decide (d % 2 = 1)) := by
        rcases Nat.mod_two_eq_zero_or_one d with h | h <;> simp [h, Nat.add_mod]
      rw [e]
      generalize (false ^^ decide (d % 2 = 1)) = s at hmem hv
      have hc : ∀ ψ ∈ S, ∀ μ₁ μ₂ : List Bool, μ₁.count false = p → μ₁.count true = q →
          μ₂.count false = N + 2 - p - V → μ₂.count true = N + 2 - q - V →
            ConstOnMiddle ψ (u ++ μ₁) (μ₂ ++ v) V V := fun ψ hψ μ₁ μ₂ h₁ h₂ h₁' h₂' =>
        ConstOnBox.constOnMiddle (hS ψ hψ).1 (fun χ hχ => hS' χ (List.mem_flatMap.2 ⟨ψ, hψ, hχ⟩))
          (by rw [h₁, h₂]; exact h ψ hψ) (by omega) (by omega)
      cases s
      · refine ⟨u ++ (List.replicate p false ++ List.replicate q true), ?_,
          List.replicate (N + 2 - q - V) true ++ List.replicate (N + 2 - p - V) false ++ v, ?_,
          fun ψ hψ => hc ψ hψ _ _ (by simp [List.count_replicate]) (by simp [List.count_replicate])
            (by simp [List.count_replicate]) (by simp [List.count_replicate])⟩
        · rw [← List.append_assoc]
          exact hmem p 1 _ ⟨q, by omega, [], rfl, (List.append_nil _).symm⟩
        · rw [List.append_assoc]
          exact ⟨N + 2 - q - V, by omega, _, replicate_append_mem_altPlus hv _, rfl⟩
      · refine ⟨u ++ (List.replicate q true ++ List.replicate p false), ?_,
          List.replicate (N + 2 - p - V) false ++ List.replicate (N + 2 - q - V) true ++ v, ?_,
          fun ψ hψ => hc ψ hψ _ _ (by simp [List.count_replicate]) (by simp [List.count_replicate])
            (by simp [List.count_replicate]) (by simp [List.count_replicate])⟩
        · rw [← List.append_assoc]
          exact hmem q 1 _ ⟨p, by omega, [], rfl, (List.append_nil _).symm⟩
        · rw [List.append_assoc]
          exact ⟨N + 2 - p - V, by omega, _, replicate_append_mem_altPlus hv _, rfl⟩

/-- The hypothesis of `exists_constOnMiddle_altPlus` is satisfiable: `Q_a` has
depth `0`. -/
example : ∀ ψ ∈ [(Form.sym false : Form Bool)], ψ ∈ TLC Bool 0 := by
  intro ψ hψ
  rw [List.mem_singleton] at hψ
  exact hψ ▸ ⟨rfl, le_rfl⟩

/-- **A formula of `TL[◁#,▷#]` of depth `k + 1` confuses `L_{2k+3}` with
`L_{2k+5}`.**  There are `w₁ ∈ L_{2k+3}` and `w₂ ∈ L_{2k+5}` on which it agrees
(Appendix D: what `lem:cropping` and `lem:reduction` are used for in the proof
of `thm:TLC_depth`). -/
theorem exists_models_iff_altPlusDouble (k : ℕ) (φ : Form Bool) (hφ : φ ∈ TLC Bool (k + 1)) :
    ∃ w₁ w₂ : List Bool, w₁ ∈ altPlus false (2 * k + 3) ∧ w₂ ∈ altPlus false (2 * k + 5) ∧
      (φ.models w₁ ↔ φ.models w₂) := by
  obtain ⟨u, hu, v, hv, hS⟩ := exists_constOnMiddle_altPlus k φ.countSubs
    (fun ψ hψ => Form.mem_TLC_of_mem_countSubs φ hφ hψ) 2
  have hmem := append_mem_altPlus k false u hu
  generalize (false ^^ decide (k % 2 = 1)) = s at hmem hv
  have hv₁ : 1 ≤ v.length := by
    obtain ⟨m, hm, v₀, -, rfl⟩ := hv
    simp only [List.length_append, List.length_replicate]
    omega
  have e := Form.sat_eq_of_constOnMiddle (u := u) (v := v) (m := [s, !s, !s, s])
    (m' := [!s, s, s, !s]) (X := 2) (Y := 2) (by cases s <;> rfl) (by cases s <;> rfl)
    (by cases s <;> rfl) (by cases s <;> rfl) (j := (u ++ [s, !s, !s, s] ++ v).length)
    (by simp only [List.length_append, List.length_cons, List.length_nil]; omega) (Or.inr (by simp only [List.length_append, List.length_cons, List.length_nil]; omega))
    φ hφ.1 hS
  refine ⟨u ++ [s, !s, !s, s] ++ v, u ++ [!s, s, s, !s] ++ v, ?_, ?_, ?_⟩
  · have h₁ := hmem 1 (k + 2) (List.replicate 2 (!s) ++ (List.replicate 1 s ++ v))
      ⟨2, by omega, _, by rw [Bool.not_not]; exact replicate_append_mem_altPlus hv 1, rfl⟩
    rw [show k + 1 + (k + 2) = 2 * k + 3 by omega] at h₁
    simpa using h₁
  · have a₁ : (!s) :: v ∈ altPlus (!s) (k + 2) := cons_mem_altPlus_flip (by rwa [Bool.not_not])
    have a₂ : s :: s :: (!s) :: v ∈ altPlus s (k + 3) :=
      cons_mem_altPlus_same (cons_mem_altPlus_flip a₁)
    have h₂ := hmem 0 (k + 4) ([!s, s, s, !s] ++ v)
      (cons_mem_altPlus_flip (by rwa [Bool.not_not]))
    rw [show k + 1 + (k + 4) = 2 * k + 5 by omega] at h₂
    simpa using h₂
  · have hl : (u ++ [!s, s, s, !s] ++ v).length = (u ++ [s, !s, !s, s] ++ v).length := by simp
    rw [Form.models, Form.models, hl, e]

/-- The hypothesis of `exists_models_iff_altPlusDouble` is satisfiable:
`▷#[Q_a] < 1` has depth `1`. -/
example : (Form.lt (.countR (.sym false)) .one : Form Bool) ∈ TLC Bool (0 + 1) :=
  ⟨rfl, by simp [Form.depth, Term.depth]⟩

/-- **`L_{2k+3}` is not definable in `TL[◁#,▷#]` at depth `k + 1`**
(arXiv:2506.16055v3, Theorem `thm:TLC_depth`, the lower bound, proved without
`lem:reduction`). -/
theorem not_definable_altPlus_double (k : ℕ) : ¬ Definable (altPlus false (2 * k + 3)) (k + 1) := by
  rintro ⟨φ, hφ, hL⟩
  obtain ⟨w₁, w₂, h₁, h₂, h⟩ := exists_models_iff_altPlusDouble k φ hφ
  have := eq_of_mem_altPlus ((Set.ext_iff.1 hL w₂).1 (h.1 ((Set.ext_iff.1 hL w₁).2 h₁))) h₂
  omega

end CRASP
end Transformer
