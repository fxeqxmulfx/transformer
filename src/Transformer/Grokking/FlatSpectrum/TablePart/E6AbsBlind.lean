module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6BlindBlock

/-!
# PT6: M misses the table `⌊|a − b| / 2⌋`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The kernel of this
table is invertible, so its one certificate `Y` is symmetric and keeps its form under the flip
`x ↦ 96 − x`. That leaves two live rows in each of the columns 38, 43, 53 and 58. Eight entries of
`Kᴴ Y = Yᴴ K` close on their four unknowns, and the system is invertible, so column 38 of `Y`
vanishes. Then row 38 of `K = Y (Yᴴ K)` vanishes, which is impossible.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- **M does not return the table `⌊|a − b| / 2⌋` from the training half.** -/
theorem not_returned_absDiffBlind : NotReturned blindTable := by
  intro hmin
  obtain ⟨Y, hYΩ, hY, htr⟩ := exists_isContraction_of_isMinOn hmin
  obtain ⟨hsymm, hflip0⟩ := certificate_symmetries det_blind_ne_zero blindTable_comm
    (α := fun _ => 1) (β := fun _ => 1) (by simp) (by simp)
    (fun a b => by simp [blindTable_flip]) hY htr
  have hflip (a b : ℕ) (ha : a < 97) (hb : b < 97) :
      Y (line 97 a) (line 97 b) = Y (line 97 (96 - a)) (line 97 (96 - b)) := by
    simpa using hflip0 a b ha hb
  have hV : Vanishes Y halfMask := fun a b ha hb hbit => hYΩ _ _ fun hmem => by
    have h1 := (mem_half_iff ha hb).1 (mem_coe.1 hmem)
    rw [hbit] at h1
    exact Bool.false_ne_true h1
  have hz (a b : ℕ) (ha : a < 97) (hb : b < 97) (h : deadCell a b = true) :
      Y (line 97 a) (line 97 b) = 0 := zero_of_deadCell hV hsymm hflip ha hb h
  have c38 : TwoRows Y 38 42 47 := twoRows_of_dead (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hz fun l hl => (block_dead l hl).1
  have c58 : TwoRows Y 58 54 49 := twoRows_of_dead (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hz fun l hl => (block_dead l hl).2.1
  have c43 : TwoRows Y 43 32 59 := twoRows_of_dead (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hz fun l hl => (block_dead l hl).2.2.1
  have c53 : TwoRows Y 53 64 37 := twoRows_of_dead (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hz fun l hl => (block_dead l hl).2.2.2
  have hH : (tableKernel blindTable)ᴴ * Y = Yᴴ * tableKernel blindTable := by
    have h := conjTranspose_mul_of_certificate hY htr
    have h2 := congrArg conjTranspose h
    rw [conjTranspose_mul, conjTranspose_conjTranspose, gramSqrt_conjTranspose, ← h] at h2
    exact h2
  have e1 := block_eq hH (by norm_num) (by norm_num) c38 c38 2 4 2 4 rfl rfl rfl rfl
  have e2 := block_eq hH (by norm_num) (by norm_num) c38 c58 8 5 8 5 rfl rfl rfl rfl
  have e3 := block_eq hH (by norm_num) (by norm_num) c43 c43 5 8 5 8 rfl rfl rfl rfl
  have e4 := block_eq hH (by norm_num) (by norm_num) c43 c53 10 3 10 3 rfl rfl rfl rfl
  have e5 := block_eq hH (by norm_num) (by norm_num) c38 c43 3 10 0 2 rfl rfl rfl rfl
  have e6 := block_eq hH (by norm_num) (by norm_num) c43 c38 0 2 3 10 rfl rfl rfl rfl
  have e7 := block_eq hH (by norm_num) (by norm_num) c38 c53 13 0 5 3 rfl rfl rfl rfl
  have e8 := block_eq hH (by norm_num) (by norm_num) c53 c38 5 3 13 0 rfl rfl rfl rfl
  have f1 : Y (line 97 54) (line 97 58) = Y (line 97 42) (line 97 38) :=
    hflip 54 58 (by norm_num) (by norm_num)
  have f2 : Y (line 97 49) (line 97 58) = Y (line 97 47) (line 97 38) :=
    hflip 49 58 (by norm_num) (by norm_num)
  have f3 : Y (line 97 64) (line 97 53) = Y (line 97 32) (line 97 43) :=
    hflip 64 53 (by norm_num) (by norm_num)
  have f4 : Y (line 97 37) (line 97 53) = Y (line 97 59) (line 97 43) :=
    hflip 37 53 (by norm_num) (by norm_num)
  rw [f1, f2] at e2
  rw [f3, f4] at e4 e7 e8
  obtain ⟨h0, h1⟩ := blindSys_solve det_blindSys_ne_zero e1 e2 e3 e4 e5 e6 e7 e8
  have hcol (u : Fin 1 → ZMod 97) : Y u (line 97 38) = 0 := by
    rw [← line_val u]
    rcases c38.2.2.2 _ (ZMod.val_lt (u 0)) with h | h | h
    · rw [h]
      exact h0
    · rw [h]
      exact h1
    · exact h
  have hK := mul_gramSqrt_of_certificate hY htr
  apply tableKernel_ne_zero blindTable (line 97 38) 0
  rw [← hK, Matrix.mul_apply]
  exact Finset.sum_eq_zero fun v _ => by rw [hsymm v (line 97 38), hcol v, zero_mul]

end Transformer.Grokking.FlatSpectrum.TablePart.E6
