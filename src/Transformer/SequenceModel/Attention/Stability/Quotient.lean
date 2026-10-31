module

public import Transformer.CRASP.FixedError
public import Transformer.SequenceModel.Attention.Stability.Softmax

/-!
# Rounding positive weighted averages

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A6;
the coefficient rounding convention in `CRASP.Rotary.Source`, extending
arXiv:2506.16055v3, Appendix B.1--B.2. Sums here are exact, as in that
counting model. No IEEE summation or hardware exponential accuracy is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP
open scoped BigOperators

/-- A6: numerator and denominator error propagate through division when
the denominator error is smaller than a certified positive lower bound. -/
theorem quotient_error (N D N' D' lower eN eD B : ℝ)
    (hD : lower ≤ D) (hsmall : eD < lower)
    (hN : |N' - N| ≤ eN) (hD' : |D' - D| ≤ eD) (hB : |N / D| ≤ B) :
    |N' / D' - N / D| ≤ (eN + B * eD) / (lower - eD) := by
  have heD : 0 ≤ eD := (abs_nonneg _).trans hD'
  have heN : 0 ≤ eN := (abs_nonneg _).trans hN
  have hB0 : 0 ≤ B := (abs_nonneg _).trans hB
  have hpos : 0 < D := (heD.trans_lt hsmall).trans_le hD
  have hlower : lower - eD ≤ D' := by linarith [(abs_le.mp hD').1]
  have hpos' : 0 < D' := (sub_pos.mpr hsmall).trans_le hlower
  have heq : N' / D' - N / D = ((N' - N) - (N / D) * (D' - D)) / D' := by
    field_simp
    ring
  have hn : |(N' - N) - (N / D) * (D' - D)| ≤ eN + B * eD := by
    refine (abs_sub _ _).trans ?_
    rw [abs_mul]
    exact add_le_add hN (mul_le_mul hB hD' (abs_nonneg _) hB0)
  rw [heq, abs_div, abs_of_pos hpos']
  exact div_le_div₀ (by positivity) hn (sub_pos.mpr hsmall) hlower

/-- The real ratio before the final rounding of an attention coordinate.
Both coefficient tables use the actual `Fx.round`; sums have no rounding. -/
def roundedRatio {A : Type*} (S : Finset A) (p s : ℕ) (w v : A → ℝ) : ℝ :=
  (∑ j ∈ S, (Fx.round p s (w j * v j)).val) /
    ∑ j ∈ S, (Fx.round p s (w j)).val

/-- A6: exact summation of n independently rounded coefficients has error
at most n quanta, with a separate range condition at every rounding site. -/
theorem rounded_sum_error {A : Type*} (S : Finset A) (p s : ℕ) (f : A → ℝ)
    (h : ∀ j ∈ S, Fx.NoSaturation p s (f j)) :
    |(∑ j ∈ S, (Fx.round p s (f j)).val) - ∑ j ∈ S, f j| ≤
      (S.card : ℝ) * (1 / (2 : ℝ) ^ s) := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ S).trans ?_
  calc
    _ ≤ ∑ j ∈ S, (1 / (2 : ℝ) ^ s) :=
      Finset.sum_le_sum (fun j hj => (Fx.round_error_lt p s (f j) (h j hj)).le)
    _ = _ := by simp

/-- A6: coefficient rounding has a length-independent ratio error when
each weight is at least a > quantum. Both numerator and denominator must
avoid saturation. The common factor n cancels; final quotient rounding is
handled separately, and the bound is for the same input value stream. -/
theorem roundedRatio_error {A : Type*} (S : Finset A) (hne : S.Nonempty)
    (p s : ℕ) (w v : A → ℝ) (a B : ℝ)
    (ha : 1 / (2 : ℝ) ^ s < a) (hw : ∀ j ∈ S, a ≤ w j)
    (hv : ∀ j ∈ S, |v j| ≤ B)
    (hwRange : ∀ j ∈ S, Fx.NoSaturation p s (w j))
    (hnRange : ∀ j ∈ S, Fx.NoSaturation p s (w j * v j)) :
    |roundedRatio S p s w v - (∑ j ∈ S, w j * v j) / (∑ j ∈ S, w j)| ≤
      (1 / (2 : ℝ) ^ s) * (1 + B) / (a - 1 / (2 : ℝ) ^ s) := by
  let u := 1 / (2 : ℝ) ^ s
  let n := (S.card : ℝ)
  have hn : 0 < n := Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  have hu : 0 < u := by positivity
  have ha0 : 0 < a := hu.trans ha
  have hD : n * a ≤ ∑ j ∈ S, w j := by
    simpa only [nsmul_eq_mul] using Finset.card_nsmul_le_sum S w a hw
  have hD0 : 0 < ∑ j ∈ S, w j := (mul_pos hn ha0).trans_le hD
  have hB0 : 0 ≤ B := by
    obtain ⟨j, hj⟩ := hne
    exact (abs_nonneg _).trans (hv j hj)
  have hB : |(∑ j ∈ S, w j * v j) / (∑ j ∈ S, w j)| ≤ B := by
    rw [abs_div, abs_of_pos hD0, div_le_iff₀ hD0]
    refine (Finset.abs_sum_le_sum_abs _ S).trans ?_
    calc
      _ ≤ ∑ j ∈ S, B * w j := by
        apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul, abs_of_pos (ha0.trans_le (hw j hj)), mul_comm]
        exact mul_le_mul_of_nonneg_right (hv j hj) (ha0.trans_le (hw j hj)).le
      _ = _ := by rw [Finset.mul_sum]
  have he := quotient_error (∑ j ∈ S, w j * v j) (∑ j ∈ S, w j)
    (∑ j ∈ S, (Fx.round p s (w j * v j)).val) (∑ j ∈ S, (Fx.round p s (w j)).val)
    (n * a) (n * u) (n * u) B hD (mul_lt_mul_of_pos_left ha hn)
    (rounded_sum_error S p s _ hnRange) (rounded_sum_error S p s _ hwRange) hB
  have heq : (n * u + B * (n * u)) / (n * a - n * u) = u * (1 + B) / (a - u) := by
    rw [← mul_sub]
    field_simp
  exact he.trans_eq heq

example := quotient_error 2 3 (21 / 10) (29 / 10) 3 (1 / 10) (1 / 10) 1
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := rounded_sum_error (Finset.univ : Finset (Fin 3)) 8 4
  (fun _ => (1 : ℝ) / 3) (by intro j hj; norm_num [Fx.NoSaturation])
example := roundedRatio_error (Finset.univ : Finset (Fin 3)) (by simp) 8 4
  (fun _ => (1 : ℝ)) (fun _ => (1 : ℝ) / 3) 1 1 (by norm_num)
  (by intro j hj; rfl) (by intro j hj; norm_num)
  (by intro j hj; norm_num [Fx.NoSaturation]) (by intro j hj; norm_num [Fx.NoSaturation])

end Transformer.SequenceModel.Attention.Stability
