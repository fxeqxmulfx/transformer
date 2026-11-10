module

public import Transformer.Grokking.GradientEvidence.MainAdamMotion
public import Mathlib.Analysis.SpecialFunctions.Exponential

/-!
# What the decay of a recipe removes within its run

Source: `review/flat_spectrum.md`, *The rate of a recipe* (added), items RR1 and RR2.

The path of a coordinate of the main model's AdamW is its initial value and its updates, each
times the product of the decay factors `1 - η r * ρ` of the later steps (`mainAdamPath_eq_sum`).
A coordinate whose gradient is zero at every step keeps zero moments, and its path is the
initial value times that product (`mainAdamPath_zero`). A product of such factors is at least
one minus their sum (`one_sub_sum_le_prod`).

* The recipe of the selected main has rates at most `0.001` and the decay `0.1`: over any of
  its 5000 updates the product is between one half and one (`half_le_prod_main`,
  `half_le_prod_Ico_main`).
* A constant rate `0.03` with the decay `1`, the numbers of the historical recipe, leaves at
  most `exp (-150)` of a coordinate without gradient (`historical_remainder`).

These are statements about the decay term alone. They do not say which coordinates of a
trained model receive no gradient.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset

/-- **RR1**, `review/flat_spectrum.md`: the path is the initial value and the updates, each
times the product of the decay factors of the steps after it. -/
theorem mainAdamPath_eq_sum (p0 ρ eps : ℝ) (η g : ℕ → ℝ) (t : ℕ) :
    mainAdamPath p0 ρ eps η g t = (∏ s ∈ range t, (1 - η s * ρ)) * p0 -
      ∑ s ∈ range t, (∏ r ∈ Ico (s + 1) t, (1 - η r * ρ)) * (η s * mainDirection g eps s) := by
  induction t with
  | zero => simp [mainAdamPath]
  | succ t ih =>
    have h : ∑ s ∈ range t, (∏ r ∈ Ico (s + 1) (t + 1), (1 - η r * ρ)) *
        (η s * mainDirection g eps s) = (1 - η t * ρ) * ∑ s ∈ range t,
          (∏ r ∈ Ico (s + 1) t, (1 - η r * ρ)) * (η s * mainDirection g eps s) := by
      rw [mul_sum]
      refine sum_congr rfl fun s hs => ?_
      rw [prod_Ico_succ_top (Nat.succ_le_of_lt (mem_range.mp hs))]
      ring
    rw [mainAdamPath, ih, prod_range_succ, sum_range_succ, h, Ico_self, prod_empty]
    ring

/-- RR1: with no gradient the two moments stay zero. -/
theorem mainMoments_zero (t : ℕ) : mainMoments (fun _ => 0) t = (0, 0) := by
  induction t with
  | zero => rfl
  | succ t ih => simp [mainMoments, ih, mainFirst, mainSecond]

/-- RR1: with no gradient the adaptive direction is zero, for every epsilon. -/
theorem mainDirection_zero (eps : ℝ) (t : ℕ) : mainDirection (fun _ => 0) eps t = 0 := by
  simp [mainDirection, mainMoments_zero]

/-- **RR1**, `review/flat_spectrum.md`: a coordinate without gradient is its initial value
times the product of the decay factors of the schedule. -/
theorem mainAdamPath_zero (p0 ρ eps : ℝ) (η : ℕ → ℝ) (t : ℕ) :
    mainAdamPath p0 ρ eps η (fun _ => 0) t = (∏ s ∈ range t, (1 - η s * ρ)) * p0 := by
  rw [mainAdamPath_eq_sum]
  simp [mainDirection_zero]

/-- **RR2**, `review/flat_spectrum.md`: a product of factors `1 - a s` with `0 ≤ a s ≤ 1` is
at least one minus the sum. -/
theorem one_sub_sum_le_prod (S : Finset ℕ) (a : ℕ → ℝ) (h0 : ∀ s, 0 ≤ a s)
    (h1 : ∀ s, a s ≤ 1) : 1 - ∑ s ∈ S, a s ≤ ∏ s ∈ S, (1 - a s) := by
  induction S using Finset.induction_on with
  | empty => simp
  | insert x S hx ih =>
    rw [sum_insert hx, prod_insert hx]
    have hs : 0 ≤ ∑ s ∈ S, a s := sum_nonneg fun s _ => h0 s
    have hp := mul_le_mul_of_nonneg_left ih (sub_nonneg.mpr (h1 x))
    nlinarith [mul_nonneg hs (h0 x)]

example : 1 - ∑ s ∈ range 3, 1 / ((s : ℝ) + 2) ≤ ∏ s ∈ range 3, (1 - 1 / ((s : ℝ) + 2)) :=
  one_sub_sum_le_prod (range 3) (fun s => 1 / ((s : ℝ) + 2)) (fun s => by positivity)
    (fun s => by rw [div_le_one (by positivity)]; linarith [Nat.cast_nonneg (α := ℝ) s])

/-- **RR2 for the recipe of the selected main**, `review/flat_spectrum.md`: with rates in
`[0, 0.001]` and the decay `0.1`, the product of the decay factors of at most 5000 updates is
between one half and one. -/
theorem half_le_prod_main (η : ℕ → ℝ) (h0 : ∀ s, 0 ≤ η s) (h1 : ∀ s, η s ≤ 1 / 1000)
    (S : Finset ℕ) (hS : S.card ≤ 5000) :
    1 / 2 ≤ ∏ s ∈ S, (1 - η s * (1 / 10)) ∧ ∏ s ∈ S, (1 - η s * (1 / 10)) ≤ 1 := by
  refine ⟨?_, prod_le_one₀ (fun s _ => by nlinarith [h1 s]) fun s _ => by nlinarith [h0 s]⟩
  have hsum : ∑ s ∈ S, η s * (1 / 10) ≤ 1 / 2 := by
    have h := sum_le_card_nsmul S (fun s => η s * (1 / 10)) (1 / 10000)
      fun s _ => by nlinarith [h1 s]
    rw [nsmul_eq_mul] at h
    have hc : (S.card : ℝ) ≤ 5000 := by exact_mod_cast hS
    linarith
  have h := one_sub_sum_le_prod S (fun s => η s * (1 / 10))
    (fun s => by nlinarith [h0 s]) (fun s => by nlinarith [h1 s])
  linarith

/-- The schedule of the selected main between its warm-up and its cooldown. -/
example :=
  half_le_prod_main (fun _ => 1 / 1000) (fun _ => by norm_num) (fun _ => le_rfl) (range 5000)
    (by simp)

/-- **RR2**, `review/flat_spectrum.md`: in the recipe of the selected main every update
reaches the end of a run of at most 5000 updates with a weight between one half and one. -/
theorem half_le_prod_Ico_main (η : ℕ → ℝ) (h0 : ∀ s, 0 ≤ η s) (h1 : ∀ s, η s ≤ 1 / 1000)
    (s : ℕ) {t : ℕ} (ht : t ≤ 5000) :
    1 / 2 ≤ ∏ r ∈ Ico (s + 1) t, (1 - η r * (1 / 10)) ∧
      ∏ r ∈ Ico (s + 1) t, (1 - η r * (1 / 10)) ≤ 1 :=
  half_le_prod_main η h0 h1 _ (by rw [Nat.card_Ico]; omega)

example := half_le_prod_Ico_main (fun _ => 1 / 1000) (fun _ => by norm_num) (fun _ => le_rfl) 7
  (le_refl 5000)

/-- RR2: a power of a decay factor is at most the exponential of minus the sum. -/
theorem one_sub_pow_le_exp {x : ℝ} (hx : x ≤ 1) (n : ℕ) :
    (1 - x) ^ n ≤ Real.exp (-(n * x)) := by
  rw [← mul_neg, Real.exp_nat_mul]
  exact pow_le_pow_left₀ (sub_nonneg.mpr hx) (Real.one_sub_le_exp_neg x) n

example : (1 - 1 / 2 : ℝ) ^ 4 ≤ Real.exp (-((4 : ℕ) * (1 / 2))) :=
  one_sub_pow_le_exp (by norm_num) 4

/-- **RR2 for a constant rate `0.03` and the decay `1`**, `review/flat_spectrum.md`: after
5000 updates at most `exp (-150)` of a coordinate without gradient remains. -/
theorem historical_remainder (p0 eps : ℝ) :
    |mainAdamPath p0 1 eps (fun _ => 3 / 100) (fun _ => 0) 5000| ≤ Real.exp (-150) * |p0| := by
  rw [mainAdamPath_zero, prod_const, card_range, abs_mul]
  refine mul_le_mul_of_nonneg_right ?_ (abs_nonneg _)
  have h := one_sub_pow_le_exp (x := 3 / 100 * 1) (by norm_num) 5000
  rw [abs_of_nonneg (pow_nonneg (by norm_num) _)]
  refine h.trans (le_of_eq ?_)
  norm_num

end Transformer.Grokking.GradientEvidence
