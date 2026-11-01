module

public import Transformer.SequenceModel.Attention.Backend.Tables

/-!
# Exact geometric phases and stored-table boxes for head width four

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K5--K6;
diff_gpt b98c665b, `rope.py`. At source base 10000 and position one the
two exact phases are 1 and 1/100. The captured source computes both tables
in float32; the checks include frequency construction and table rounding.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open GPTMini

/-- K6: the two rational fixture phases are the source geometric frequencies. -/
theorem head4_phases (k : Fin 2) : invFreq 4 10000 k = ![1, 1 / 100] k := by
  fin_cases k
  · norm_num [invFreq]
  · have h : (10000 : ℝ) ^ (-(1 / 2 : ℝ)) = 1 / 100 := by
      rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow]
      norm_num
    convert h using 1 <;> norm_num [invFreq]

/-- K6: every independently captured cosine and sine within its rational
polynomial box satisfies the stored-table contract with tau=2e-6. -/
theorem rope_fixture (c s : Fin 2 → ℝ)
    (hc : ∀ k, |c k - (rotationPolynomial (![1, 1 / 100] k) 12).re| ≤ 1 / 1000000)
    (hs : ∀ k, |s k - (rotationPolynomial (![1, 1 / 100] k) 12).im| ≤ 1 / 1000000) :
    ∀ k, |c k - Real.cos (invFreq 4 10000 k)| ≤ 2 / 1000000 ∧
      |s k - Real.sin (invFreq 4 10000 k)| ≤ 2 / 1000000 := by
  intro k
  rw [head4_phases]
  have h := stored_rotation_error (![1, 1 / 100] k) (c k) (s k) (1 / 1000000) 12
    (by fin_cases k <;> norm_num) (hc k) (hs k)
  have hb : (1 / 1000000 : ℝ) + |![1, 1 / 100] k| ^ 12 / (Nat.factorial 12) * 2 ≤
      2 / 1000000 := by fin_cases k <;> norm_num
  exact ⟨h.1.trans hb, h.2.trans hb⟩

example := rope_fixture (fun k => (rotationPolynomial (![1, 1 / 100] k) 12).re)
  (fun k => (rotationPolynomial (![1, 1 / 100] k) 12).im)
  (by intro k; simp) (by intro k; simp)

end Transformer.SequenceModel.Attention.Backend
