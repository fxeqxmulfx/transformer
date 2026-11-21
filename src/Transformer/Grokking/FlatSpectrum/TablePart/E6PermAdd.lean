module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6SumWitnesses
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Sums

/-!
# PT6: the table of a permutation of `a + b`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. In the task `perm-add`
the label is `π(a + b)` for a fixed permutation `π` of `ℤ/97`. Every row of its kernel runs over all
the values of `ψ`, so the rows sum to zero and the kernel has no inverse. But `K + J` has one, so
`not_isMinOn_of_balanced` applies: for any two anti-diagonals, one row has an unseen cell on both.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The permutation of the task `perm-add` on `ZMod 97`. -/
def permMap (s : ZMod 97) : ZMod 97 := (permTable.getD s.val 0 : ℕ)

/-- `permTable` takes every value below 97. -/
theorem perm_values : ∀ y, y < 97 → ∃ s, s < 97 ∧ permTable.getD s 0 = y := by
  decide +kernel

lemma permMap_surjective : Function.Surjective permMap := fun y => by
  obtain ⟨s, hs, h⟩ := perm_values y.val y.val_lt
  refine ⟨(s : ZMod 97), ?_⟩
  rw [permMap, ZMod.val_natCast_of_lt hs, h, ZMod.natCast_zmod_val]

/-- The values of `ψ` along the permutation sum to zero. -/
lemma sum_perm : ∑ s : ZMod 97, (ZMod.stdAddChar (permMap s) : ℂ) = 0 := by
  have hbij : Function.Bijective permMap :=
    ⟨Finite.injective_iff_surjective.2 permMap_surjective, permMap_surjective⟩
  rw [hbij.sum_comp fun y => (ZMod.stdAddChar y : ℂ)]
  have h := sum_stdAddChar_mul (p := 97) 1
  simpa using h

/-- **For any two anti-diagonals, one row has a cell outside the half on each.** -/
theorem pair_cover : ∀ x, x < 97 → ∀ y, y < 97 → ∃ r, r < 97 ∧
    halfMask.testBit (97 * r + (x + 97 - r) % 97) = false ∧
    halfMask.testBit (97 * r + (y + 97 - r) % 97) = false := by
  decide +kernel

/-- **M does not return the task `perm-add` from the training half.** -/
theorem not_returned_permAdd : NotReturned fun a b => permMap (a + b) := by
  have hK0 : tableKernel (fun a b => permMap (a + b)) ≠ 0 := fun h =>
    tableKernel_ne_zero (fun a b => permMap (a + b)) 0 0 (by rw [h]; rfl)
  have hrow (u : Fin 1 → ZMod 97) :
      ∑ v, tableKernel (fun a b => permMap (a + b)) u v = 0 := by
    have hline : ∑ v : Fin 1 → ZMod 97, (ZMod.stdAddChar (permMap (u 0 + v 0)) : ℂ) =
        ∑ x, (ZMod.stdAddChar (permMap (u 0 + x)) : ℂ) :=
      sum_line_apply fun x => (ZMod.stdAddChar (permMap (u 0 + x)) : ℂ)
    simp only [tableKernel, of_apply]
    have hs := Equiv.sum_comp (Equiv.addLeft (u 0)) fun s => (ZMod.stdAddChar (permMap s) : ℂ)
    simp only [Equiv.coe_addLeft] at hs
    rw [hline, hs, sum_perm]
  have hdet : (tableKernel (fun a b => permMap (a + b)) +
      Matrix.of fun _ _ => (1 : ℂ)).det ≠ 0 := by
    have h0 := det_sum_ne_zero (e := fun s => permTable.getD s 0) (c := 1) (W := wPerm)
      (by decide +kernel)
    have hmat : tableKernel (fun a b => permMap (a + b)) + Matrix.of (fun _ _ => (1 : ℂ)) =
        Matrix.of fun u v : Fin 1 → ZMod 97 =>
          (ZMod.stdAddChar ((permTable.getD (u 0 + v 0).val 0 : ℕ) : ZMod 97) : ℂ) +
            ((1 : ℕ) : ℂ) := by
      ext u v
      simp [tableKernel, permMap]
    rw [hmat]
    exact h0
  refine not_isMinOn_of_balanced hK0 hrow hdet (fun _ => 0) shiftRow shiftCol (fun _ _ => 1)
    (fun _ _ => 1) (by simp) (by simp) (fun t u v => ?_) fun u v => ?_
  · simp [tableKernel, shiftRow, shiftCol]
  · obtain ⟨r, hr, h1, h2⟩ := pair_cover (u 0 + v 0).val (ZMod.val_lt _)
      (u 0 + (fun _ => (0 : ZMod 97)) (0 : Fin 1)).val (ZMod.val_lt _)
    refine ⟨(r : ZMod 97) - u 0, ?_, ?_⟩
    · rw [shiftRow_eq, shiftCol_eq u v hr]
      intro hmem
      have hb := (mem_half_iff hr (Nat.mod_lt _ (by norm_num))).1 (mem_coe.1 hmem)
      rw [h1] at hb
      exact Bool.false_ne_true hb
    · rw [shiftRow_eq, shiftCol_eq u (fun _ => 0) hr]
      intro hmem
      have hb := (mem_half_iff hr (Nat.mod_lt _ (by norm_num))).1 (mem_coe.1 hmem)
      rw [h2] at hb
      exact Bool.false_ne_true hb

end Transformer.Grokking.FlatSpectrum.TablePart.E6
