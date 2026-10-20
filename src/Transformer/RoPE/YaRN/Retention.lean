module

public import Transformer.RoPE.YaRN.Section3_Temperature
public import Transformer.ALM.Softmax
public import Transformer.Imbalance.Section3_SimpleModel

/-!
# Deriving temperature from an explicit attention-retention objective

These are derived results, not claims of YaRN. They investigate whether the
empirical temperature of arXiv:2309.00071v3, §3.3, can follow from softmax.
We reuse the exact two-level softmax calculation from
`Transformer.Imbalance.softmax_twoLevel` (arXiv:2402.19449v2, Appendix I,
Lemma 4) and the arbitrary-score gap bound in `Transformer.ALM.Softmax`.
Competitors are counted separately from the one desired token. All gaps
are measured in logits after the ordinary attention normalization.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE.YaRN

/-- Exact winner probability with `competitors` identical lower logits.
The formula is also a worst-case bound for scores with a common gap. -/
noncomputable def twoLevelMass (competitors gap gain : ℝ) : ℝ :=
  1 / (1 + competitors * Real.exp (-(gain * gap)))

/-- The mass formula is the actual finite softmax, with one winner at `gap`
and `n` competitors at zero. It is not an assumed probability model. -/
theorem two_level_mass_eq_softmax (n : ℕ) (target : Fin (n + 1)) (gap gain : ℝ) :
    Precision.softmax (fun j => gain * (if j = target then gap else 0)) target =
      twoLevelMass n gap gain := by
  have hf : (fun j => gain * (if j = target then gap else 0)) =
      Imbalance.twoLevel (gain * gap + 0) 0 target := by
    funext j
    by_cases hj : j = target <;> simp [Imbalance.twoLevel, hj]
  unfold Precision.softmax
  rw [hf, Imbalance.sum_exp_twoLevel]
  simp only [Imbalance.twoLevel, ite_true, add_zero, Real.exp_zero, mul_one,
    Nat.cast_add, Nat.cast_one, add_sub_cancel_right]
  unfold twoLevelMass
  rw [Real.exp_neg]
  field_simp

/-- Equal scores give uniform attention at every finite temperature. -/
theorem uniform_mass (competitors gain : ℝ) :
    twoLevelMass competitors 0 gain = 1 / (1 + competitors) := by
  simp [twoLevelMass]

/-- A concrete obstruction with genuine softmax rows: appending a third
tied key reduces the first key's mass, whatever the two temperatures. -/
theorem tied_context_counterexample (oldGain newGain : ℝ) :
    Precision.softmax (fun _ : Fin 3 => newGain * 0) 0 <
      Precision.softmax (fun _ : Fin 2 => oldGain * 0) 0 := by
  norm_num [Precision.softmax, Fin.sum_univ_succ]

/-- Existing ALM concentration gives a nonvacuous bound for arbitrary
competitor logits separated from the desired token by at least `gap`.
Unlike the exact two-level identity, equal losing scores are unnecessary. -/
theorem mass_lower_bound {n : ℕ} (scores : Fin (n + 1) → ℝ)
    (target : Fin (n + 1)) (gap gain : ℝ) (hg : 0 ≤ gain)
    (hgap : ∀ j, j ≠ target → scores j + gap ≤ scores target) :
    twoLevelMass n gap gain ≤ Precision.softmax (fun j => gain * scores j) target := by
  have htail : ∑ j ∈ Finset.univ.erase target,
      Real.exp (gain * (scores j - scores target)) ≤
        (n : ℝ) * Real.exp (-(gain * gap)) := by
    calc
      _ ≤ ∑ _j ∈ Finset.univ.erase target, Real.exp (-(gain * gap)) := by
        apply Finset.sum_le_sum
        intro j hj
        apply Real.exp_le_exp.2
        have h := hgap j (Finset.ne_of_mem_erase hj)
        nlinarith
      _ = _ := by simp
  exact ALM.softmax_winner_sharp gain scores target _ htail

example : twoLevelMass 1 1 2 ≤
    Precision.softmax (fun j : Fin 2 => 2 * (if j = 0 then 1 else 0)) 0 := by
  simpa only [Nat.cast_one] using mass_lower_bound (n := 1)
    (fun j : Fin 2 => if j = 0 then 1 else 0) 0 1 2 (by norm_num) (by
    intro j hj
    fin_cases j <;> simp_all)

/-- Keeping the winner's probability fixed determines the scaled gap
exactly. The logarithm appears from counting the competing exponentials. -/
theorem mass_preservation_changed_gap_iff {original extended : ℝ}
    (ho : 0 < original) (he : 0 < extended) (oldGap newGap gain : ℝ) :
    twoLevelMass extended newGap gain = twoLevelMass original oldGap 1 ↔
      gain * newGap = oldGap + Real.log (extended / original) := by
  have hmass : twoLevelMass extended newGap gain = twoLevelMass original oldGap 1 ↔
      extended * Real.exp (-(gain * newGap)) = original * Real.exp (-oldGap) := by
    unfold twoLevelMass
    simp only [one_mul, one_div]
    constructor
    · intro h
      have hi := congrArg (fun x : ℝ => x⁻¹) h
      simp only [inv_inv] at hi
      linarith
    · intro h
      rw [h]
  rw [hmass]
  constructor
  · intro h
    have hl := congrArg Real.log h
    rw [Real.log_mul (ne_of_gt he) (Real.exp_ne_zero _),
      Real.log_mul (ne_of_gt ho) (Real.exp_ne_zero _), Real.log_exp, Real.log_exp] at hl
    rw [Real.log_div (ne_of_gt he) (ne_of_gt ho)]
    linarith
  · intro h
    have hx : -(gain * newGap) = -oldGap - Real.log (extended / original) := by linarith
    rw [hx, Real.exp_sub, Real.exp_log (div_pos he ho)]
    field_simp

example : twoLevelMass 2 1 (2 + Real.log 2) = twoLevelMass 1 2 1 := by
  apply (mass_preservation_changed_gap_iff (by norm_num : (0 : ℝ) < 1)
    (by norm_num : (0 : ℝ) < 2) _ _ _).2
  simp

/-- If the gap is unchanged, retention adds the log competitor growth to
that same reference gap. The unchanged-gap assumption is explicit. -/
theorem mass_preservation_iff {original extended : ℝ}
    (ho : 0 < original) (he : 0 < extended) (gap gain : ℝ) :
    twoLevelMass extended gap gain = twoLevelMass original gap 1 ↔
      gain * gap = gap + Real.log (extended / original) :=
  mass_preservation_changed_gap_iff ho he gap gap gain

example : twoLevelMass 255 5 (1 + Real.log (255 / 63 : ℝ) / 5) =
    twoLevelMass 63 5 1 := by
  apply (mass_preservation_iff (by norm_num : (0 : ℝ) < 63)
    (by norm_num : (0 : ℝ) < 255) _ _).2
  ring

end Transformer.RoPE.YaRN
