module

public import Transformer.Grokking.FlatSpectrum.TablePart.ReadoutCodes

/-!
# The energy a balanced code keeps on the seen cells

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT8. The table of a code
holds at most `2p − 1` copies of the energy of the code (`frobeniusSq_codeTable_le`). So if M
returns the table of a balanced code from `Ω`, the seen cells carry at least the share
`(p − 1)² / ((2p − 1)(1 + √κ) p)` of its energy (`energy_share_of_isMinOn`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- The table holds at most `2p − 1` copies of the energy of the code: the rows `a ≠ 0` are
permutations of the code, and the row `a = 0` is constant. -/
lemma frobeniusSq_codeTable_le (P : (ZMod p)[X]) (ρ : ZMod p → ℂ) :
    frobeniusSq (codeTable P ρ) ≤ (2 * p - 1) * ∑ c, ‖ρ c‖ ^ 2 := by
  set E := ∑ c, ‖ρ c‖ ^ 2 with hE
  have hrow (a : ZMod p) : ∑ b, ‖codeTable P ρ a b‖ ^ 2 ≤ if a = 0 then (p : ℝ) * E else E := by
    by_cases ha : a = 0
    · simp only [ha, ↓reduceIte, taskKernel, of_apply, zero_mul, add_zero, Finset.sum_const,
        Finset.card_univ, ZMod.card, nsmul_eq_mul]
      exact mul_le_mul_of_nonneg_left
        (Finset.single_le_sum (f := fun c => ‖ρ c‖ ^ 2) (fun _ _ => by positivity)
          (Finset.mem_univ _)) (Nat.cast_nonneg _)
    · simp only [ha, ↓reduceIte, taskKernel, of_apply]
      exact (sum_row_eq P ha fun c => ‖ρ c‖ ^ 2).le
  calc frobeniusSq (codeTable P ρ) = ∑ a, ∑ b, ‖codeTable P ρ a b‖ ^ 2 := rfl
    _ ≤ ∑ a, if a = 0 then (p : ℝ) * E else E := Finset.sum_le_sum fun a _ => hrow a
    _ = (2 * p - 1) * E := by
      have e (a : ZMod p) : (if a = 0 then (p : ℝ) * E else E) =
          E + if a = 0 then ((p : ℝ) - 1) * E else 0 := by
        split_ifs <;> ring
      simp only [e, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, ZMod.card,
        nsmul_eq_mul, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
      ring

/-- **A balanced code of any shape keeps a share of its energy on the seen cells.** -/
theorem energy_share_of_isMinOn {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) {ρ : ZMod p → ℂ}
    (hbal : ∑ c, ρ c = 0) {Ω : Set (ZMod p × ZMod p)} [DecidablePred (· ∈ Ω)]
    (hK : IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = codeTable P ρ ij.1 ij.2}
      (codeTable P ρ)) :
    ((p : ℝ) - 1) ^ 2 * frobeniusSq (codeTable P ρ) ≤
      (2 * p - 1) * ((1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) *
        (p * ∑ a, ∑ b, if (a, b) ∈ Ω then ‖codeTable P ρ a b‖ ^ 2 else 0)) := by
  have hcenter : ∑ c, ‖centerCode ρ c‖ ^ 2 = ∑ c, ‖ρ c‖ ^ 2 := by
    have h := sum_norm_sq_centerCode ρ
    rw [hbal, norm_zero, zero_pow two_ne_zero, zero_div, add_zero] at h
    exact h.symm
  have h1 := centered_energy_le_of_isMinOn hm ρ hK
  rw [hcenter] at h1
  have hp : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_lt.le
  calc ((p : ℝ) - 1) ^ 2 * frobeniusSq (codeTable P ρ)
      ≤ ((p : ℝ) - 1) ^ 2 * ((2 * p - 1) * ∑ c, ‖ρ c‖ ^ 2) :=
        mul_le_mul_of_nonneg_left (frobeniusSq_codeTable_le P ρ) (sq_nonneg _)
    _ = (2 * p - 1) * (((p : ℝ) - 1) ^ 2 * ∑ c, ‖ρ c‖ ^ 2) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_left h1 (by linarith)

example := energy_share_of_isMinOn (P := (X ^ 2 : (ZMod 3)[X]))
  (ρ := fun c => (ZMod.stdAddChar c : ℂ)) (Ω := Set.univ) (by simp)
  (by simpa using sum_stdAddChar_mul (p := 3) 1) (isMinOn_univ _)

end Transformer.Grokking.FlatSpectrum.TablePart
