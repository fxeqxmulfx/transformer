/-
# The lower bound of the depth hierarchy

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4 and Appendix C.2–C.4: the lower bounds of `thm:TLCl_depth`
and `cor:prediction_task_depth`, proved without `lem:reduction`.

The paper derives that `L_{k+2}` needs depth `k + 2` from
`lem:cropping_oneway` and `lem:reduction`, both false as stated
(`Transformer.CRASP.CroppingUnsound`, `Transformer.CRASP.ReductionUnsound`).
The argument here keeps the cropping, with the positions before the interval
fixed to a prefix, and adds one block per level of depth in place of the
reduction.

Formulas of `TL[◁#]_d` are all constant, letter by letter, on the quadrant
`x ≤ V` after some word `u` of `L_{d+1}` (`exists_constOnStrip_altPlus`).  By
induction, the formulas under their counts are constant on a larger quadrant
after some `u'` of `L_d`; shrinking gives a strip `p ≤ x ≤ p + V`, `q ≤ y` on
which the formulas themselves are constant (`exists_constOnStrip_list`); and
`u = u' aᵖ bᑫ`, or `u' bᑫ aᵖ`, lengthens the last block of `u'`, adds one, and
turns that strip into the quadrant after `u`.  A formula of depth `d + 1` then
reads, after `u`, only the letter and the numbers of `a`s and `b`s
(`Form.sat_eq_of_count_eq`), so it cannot tell `u·ss·s̄s̄`, a word of `L_{d+2}`,
from `u·s·s̄·s·s̄`, a word of `L_{d+4}`, where `s` is the last letter of `u`
(`exists_models_iff_altPlus`).
-/

import Transformer.CRASP.Blocks
import Transformer.CRASP.Shrink

namespace Transformer
namespace CRASP

/-- **Formulas of `TL[◁#]_d` are constant after a word of `L_{d+1}`.**  For
finitely many of them and every `V` there is `u ∈ L_{d+1}` after which each is
constant, letter by letter, on the quadrant `x ≤ V` (§4.4 and Appendix C.2, the
induction behind `lem:cropping_oneway`, with one new block per level of depth
in place of `lem:reduction`). -/
theorem exists_constOnStrip_altPlus :
    ∀ d (S : List (Form Bool)), (∀ ψ ∈ S, ψ ∈ TLCl Bool d) → ∀ V,
      ∃ u ∈ altPlus false (d + 1), ∀ ψ ∈ S, ConstOnStrip ψ u 0 V 0
  | 0, _, hS, V => ⟨[false], ⟨1, Nat.one_pos, [], rfl, rfl⟩, fun ψ hψ => by
      obtain ⟨-, hf, hd⟩ := hS ψ hψ
      exact Form.constOnStrip_of_depth_eq_zero [false] 0 V 0 ψ hf (Nat.le_zero.1 hd)⟩
  | d + 1, S, hS, V => by
      obtain ⟨N, hN⟩ := exists_constOnStrip_list S (fun ψ hψ =>
        let ⟨hp, hf, _⟩ := hS ψ hψ; ⟨hp, hf⟩) V
      obtain ⟨u, hu, hS'⟩ := exists_constOnStrip_altPlus d (S.flatMap Form.countSubs)
        (fun χ hχ => by
          obtain ⟨ψ, hψ, hχ⟩ := List.mem_flatMap.1 hχ
          exact Form.mem_TLCl_of_mem_countSubs ψ (hS ψ hψ) hχ) (1 + N)
      obtain ⟨p, q, hp, -, hq, h⟩ := hN u (1 + N) 1 1 le_rfl fun ψ hψ χ hχ =>
        hS' χ (List.mem_flatMap.2 ⟨ψ, hψ, hχ⟩)
      have hmem := append_mem_altPlus d false u hu
      generalize (false ^^ decide (d % 2 = 1)) = c at hmem
      cases c
      · refine ⟨u ++ List.replicate p false ++ List.replicate q true,
          hmem p 1 (List.replicate q true) ⟨q, by omega, [], rfl, (List.append_nil _).symm⟩,
          fun ψ hψ => ?_⟩
        rw [List.append_assoc]
        exact (h ψ hψ).append (by simp [List.count_replicate]) (by simp [List.count_replicate])
      · refine ⟨u ++ List.replicate q true ++ List.replicate p false,
          hmem q 1 (List.replicate p false) ⟨p, by omega, [], rfl, (List.append_nil _).symm⟩,
          fun ψ hψ => ?_⟩
        rw [List.append_assoc]
        exact (h ψ hψ).append (by simp [List.count_replicate]) (by simp [List.count_replicate])

/-- The hypothesis of `exists_constOnStrip_altPlus` is satisfiable: `Q_a` has
depth `0`. -/
example : ∀ ψ ∈ [(Form.sym false : Form Bool)], ψ ∈ TLCl Bool 0 := by
  intro ψ hψ
  rw [List.mem_singleton] at hψ
  exact hψ ▸ ⟨rfl, rfl, le_rfl⟩

/-- **A formula of depth `k + 1` confuses `L_{k+2}` with `L_{k+4}`.**  There
are `w₁ ∈ L_{k+2}` and `w₂ ∈ L_{k+4}` on which it agrees, and `w₁` is a prefix of
a word of `L_{k+4}` (§4.4 and Appendix C.2–C.4: what `lem:cropping_oneway` and
`lem:reduction` are used for in the proofs of `thm:TLCl_depth` and
`cor:prediction_task_depth`). -/
theorem exists_models_iff_altPlus (k : ℕ) (φ : Form Bool) (hφ : φ ∈ TLCl Bool (k + 1)) :
    ∃ w₁ w₂ v : List Bool, w₁ ∈ altPlus false (k + 2) ∧ w₂ ∈ altPlus false (k + 4) ∧
      w₁ ++ v ∈ altPlus false (k + 4) ∧ (φ.models w₁ ↔ φ.models w₂) := by
  obtain ⟨u, hu, hS⟩ := exists_constOnStrip_altPlus k φ.countSubs
    (fun ψ hψ => Form.mem_TLCl_of_mem_countSubs φ hφ hψ) 2
  obtain ⟨hp, hf, -⟩ := hφ
  have hmem := append_mem_altPlus k false u hu
  generalize (false ^^ decide (k % 2 = 1)) = s at hmem
  have e := Form.sat_eq_of_count_eq u 2 (m := [s, s, !s, !s]) (m' := [s, !s, s, !s]) (i := 3)
    (i' := 3) (by cases s <;> decide) (by cases s <;> decide) (by simp) (by simp)
    (by cases s <;> decide) (by cases s <;> decide) rfl φ hp hf hS
  refine ⟨u ++ [s, s, !s, !s], u ++ [s, !s, s, !s], [s, !s], ?_, ?_, ?_, ?_⟩
  · have h₁ := hmem 2 1 [!s, !s] ⟨2, by decide, [], rfl, rfl⟩
    rw [List.append_assoc] at h₁
    exact h₁
  · have h₂ := hmem 1 3 _ (mem_altPlus_three s).1
    rw [List.append_assoc] at h₂
    exact h₂
  · have h₃ := hmem 2 3 _ (mem_altPlus_three s).2
    rw [List.append_assoc] at h₃ ⊢
    exact h₃
  · have e' : φ.sat (u ++ [s, s, !s, !s]) (u ++ [s, s, !s, !s]).length =
        φ.sat (u ++ [s, !s, s, !s]) (u ++ [s, !s, s, !s]).length := by
      rw [List.length_append, List.length_append]
      exact e
    rw [Form.models, Form.models, e']

/-- The hypothesis of `exists_models_iff_altPlus` is satisfiable:
`◁#[Q_a] < 1` has depth `1`. -/
example : (Form.lt (.countL (.sym false)) .one : Form Bool) ∈ TLCl Bool (0 + 1) :=
  ⟨rfl, rfl, by simp [Form.depth, Term.depth]⟩

/-- **`L_{k+2}` is not definable at depth `k + 1`** (arXiv:2506.16055v3,
Theorem `thm:TLCl_depth`, the lower bound, proved without `lem:reduction`). -/
theorem not_definableL_altPlus (k : ℕ) : ¬ DefinableL (altPlus false (k + 2)) (k + 1) := by
  rintro ⟨φ, hφ, hL⟩
  obtain ⟨w₁, w₂, -, h₁, h₂, -, h⟩ := exists_models_iff_altPlus k φ hφ
  have := eq_of_mem_altPlus ((Set.ext_iff.1 hL w₂).1 (h.1 ((Set.ext_iff.1 hL w₁).2 h₁))) h₂
  omega

end CRASP
end Transformer
