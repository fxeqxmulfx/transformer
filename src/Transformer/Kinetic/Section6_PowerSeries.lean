module

public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Analysis.Calculus.SmoothSeries
public import Mathlib.Analysis.SpecificLimits.Normed

/-!
# Kinetic theory for Transformers — entire power series

The proof of `thm:U-shape` of arXiv:2605.09213v1 differentiates termwise the series
`ψ_c(y) = Σ_k c^{k+1} y^k / (k!(k+1)!)` of the profile at the output layer, which converges on
the whole line. This is that calculus: coefficients summable against every power (`IsEntire`),
the sum `powerSum a y = Σ_k a_k y^k`, its derivative, the sum of the coefficients
`(k+1) a_{k+1}` (`hasDerivAt_powerSum`), linearity, and comparison of sums.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer
namespace Kinetic

/-- The coefficients of an entire power series: `Σ_k |a_k| R^k` converges for every `R ≥ 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
def IsEntire (a : ℕ → ℝ) : Prop := ∀ R : ℝ, 0 ≤ R → Summable fun k => |a k| * R ^ k

/-- The sum `Σ_k a_k y^k`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def powerSum (a : ℕ → ℝ) (y : ℝ) : ℝ := ∑' k, a k * y ^ k

/-- The coefficients `(k+1) a_{k+1}` of the derivative of `Σ_k a_k y^k`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
def derivCoeffs (a : ℕ → ℝ) (k : ℕ) : ℝ := (k + 1) * a (k + 1)

/-- The coefficients `1/k!` of `exp` are entire: the witness of the examples below.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem isEntire_inv_factorial : IsEntire fun k => 1 / k.factorial := fun R _ =>
  (Real.summable_pow_div_factorial R).congr fun k => by
    rw [abs_of_pos (by positivity)]
    ring

variable {a b : ℕ → ℝ}

/-- An entire coefficient sequence converges at each actual argument.
Source: arXiv:2605.09213v1, §6, the series in `thm:U-shape`'s proof. -/
theorem IsEntire.summable (ha : IsEntire a) (y : ℝ) : Summable fun k => a k * y ^ k :=
  (ha |y| (abs_nonneg y)).of_norm_bounded fun k => by
    rw [Real.norm_eq_abs, abs_mul, abs_pow]

example := isEntire_inv_factorial.summable 1

/-- Sums of entire sequences remain entire, by absolute comparison.
Source: arXiv:2605.09213v1, §6, the series in `thm:U-shape`'s proof. -/
theorem IsEntire.add (ha : IsEntire a) (hb : IsEntire b) : IsEntire fun k => a k + b k :=
  fun R hR => Summable.of_nonneg_of_le (fun _ => by positivity)
    (fun k => by
      rw [← add_mul]
      exact mul_le_mul_of_nonneg_right (abs_add_le _ _) (by positivity))
    ((ha R hR).add (hb R hR))

example := isEntire_inv_factorial.add isEntire_inv_factorial

/-- Real scaling preserves entire coefficient sequences.
Source: arXiv:2605.09213v1, §6, the series in `thm:U-shape`'s proof. -/
theorem IsEntire.const_mul (ha : IsEntire a) (c : ℝ) : IsEntire fun k => c * a k :=
  fun R hR => ((ha R hR).mul_left |c|).congr fun k => by rw [abs_mul, mul_assoc]

example := isEntire_inv_factorial.const_mul 2

/-- `(k+1) R^k ≤ (R+1)^{k+1}`, the term `k` of the binomial expansion.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem succ_mul_pow_le {R : ℝ} (hR : 0 ≤ R) (k : ℕ) : (k + 1) * R ^ k ≤ (R + 1) ^ (k + 1) := by
  induction k with
  | zero => simpa using hR
  | succ k ih =>
    have h := pow_le_pow_left₀ hR (by linarith : R ≤ R + 1) (k + 1)
    push_cast
    calc ((k : ℝ) + 1 + 1) * R ^ (k + 1) = R * ((k + 1) * R ^ k) + R ^ (k + 1) := by ring
      _ ≤ R * (R + 1) ^ (k + 1) + (R + 1) ^ (k + 1) := by gcongr
      _ = (R + 1) ^ (k + 1 + 1) := by ring

example := succ_mul_pow_le zero_le_one 2

/-- Differentiated coefficients remain entire, by the original
binomial bound. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem IsEntire.derivCoeffs (ha : IsEntire a) : IsEntire (derivCoeffs a) := by
  intro R hR
  refine Summable.of_nonneg_of_le (fun k => by positivity) (fun k => ?_)
    ((summable_nat_add_iff 1).2 (ha (R + 1) (by linarith)))
  rw [Kinetic.derivCoeffs, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ k + 1)]
  calc (k + 1 : ℝ) * |a (k + 1)| * R ^ k = |a (k + 1)| * ((k + 1) * R ^ k) := by ring
    _ ≤ |a (k + 1)| * (R + 1) ^ (k + 1) := by gcongr; exact succ_mul_pow_le hR k

example := isEntire_inv_factorial.derivCoeffs

/-- **Termwise differentiation**: `(Σ_k a_k y^k)' = Σ_k (k+1) a_{k+1} y^k`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_powerSum (ha : IsEntire a) (y : ℝ) :
    HasDerivAt (powerSum a) (powerSum (derivCoeffs a) y) y := by
  set R := |y| + 1
  have hR : 0 ≤ R := by positivity
  have hyR : y ∈ Set.Ioo (-R) R := abs_lt.1 (by simp [R])
  have hb : ∀ (k : ℕ) z, z ∈ Set.Ioo (-R) R →
      ‖a k * (k * z ^ (k - 1))‖ ≤ |a k| * (k * R ^ (k - 1)) := fun k z hz => by
    rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_pow, Nat.abs_cast]
    gcongr
    exact (abs_lt.2 hz).le
  have hs : Summable fun k : ℕ => |a k| * (k * R ^ (k - 1)) := by
    refine (summable_nat_add_iff 1).1 ((ha.derivCoeffs R hR).congr fun k => ?_)
    rw [Kinetic.derivCoeffs, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ k + 1)]
    push_cast
    ring
  have h := hasDerivAt_tsum_of_isPreconnected (g := fun (k : ℕ) (z : ℝ) => a k * z ^ k)
    (g' := fun (k : ℕ) (z : ℝ) => a k * (k * z ^ (k - 1))) hs isOpen_Ioo isPreconnected_Ioo
    (fun k z _ => (hasDerivAt_pow k z).const_mul (a k)) hb hyR (ha.summable y) hyR
  have h0 : Summable fun k : ℕ => a k * (k * y ^ (k - 1)) :=
    hs.of_norm_bounded fun k => hb k y hyR
  rw [h0.tsum_eq_zero_add] at h
  refine h.congr_deriv ?_
  simp only [powerSum, Kinetic.derivCoeffs, CharP.cast_eq_zero, zero_mul, mul_zero, zero_add,
    Nat.cast_add, Nat.cast_one, add_tsub_cancel_right]
  congr 1
  ext k
  ring

example := hasDerivAt_powerSum isEntire_inv_factorial 0

/-- The series extension at zero is its constant coefficient.
Source: arXiv:2605.09213v1, §6, `ψ_c(0)=c` in the proof of `thm:U-shape`. -/
theorem powerSum_zero (a : ℕ → ℝ) : powerSum a 0 = a 0 := by
  rw [powerSum, tsum_eq_single 0 fun k hk => by simp [hk]]
  simp

/-- Entire sums distribute over addition of coefficient sequences.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem powerSum_add (ha : IsEntire a) (hb : IsEntire b) (y : ℝ) :
    powerSum (fun k => a k + b k) y = powerSum a y + powerSum b y := by
  simp only [powerSum, add_mul]
  exact (ha.summable y).tsum_add (hb.summable y)

example := powerSum_add isEntire_inv_factorial isEntire_inv_factorial 0

/-- Real scaling distributes over the entire sum.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem powerSum_const_mul (c : ℝ) (a : ℕ → ℝ) (y : ℝ) :
    powerSum (fun k => c * a k) y = c * powerSum a y := by
  simp only [powerSum, mul_assoc]
  exact tsum_mul_left

/-- `|Σ_k a_k y^k| ≤ Σ_k b_k R^k` when `|a_k| ≤ b_k` and `|y| ≤ R`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem abs_powerSum_le (hb : IsEntire b) (hab : ∀ k, |a k| ≤ b k) {y R : ℝ} (hy : |y| ≤ R) :
    |powerSum a y| ≤ powerSum b R := by
  have hR : 0 ≤ R := (abs_nonneg y).trans hy
  have hbR : Summable fun k => b k * R ^ k :=
    (hb R hR).congr fun k => by rw [abs_of_nonneg ((abs_nonneg _).trans (hab k))]
  have hle : ∀ k, ‖a k * y ^ k‖ ≤ b k * R ^ k := fun k => by
    rw [Real.norm_eq_abs, abs_mul, abs_pow]
    exact mul_le_mul (hab k) (pow_le_pow_left₀ (abs_nonneg y) hy k) (by positivity)
      ((abs_nonneg _).trans (hab k))
  calc |powerSum a y| = ‖∑' k, a k * y ^ k‖ := rfl
    _ ≤ ∑' k, ‖a k * y ^ k‖ := norm_tsum_le_tsum_norm (hbR.of_nonneg_of_le (fun _ => by
        positivity) hle)
    _ ≤ ∑' k, b k * R ^ k := (hbR.of_nonneg_of_le (fun _ => by positivity) hle).tsum_le_tsum
        hle hbR

example := abs_powerSum_le isEntire_inv_factorial (a := fun k => 1 / k.factorial)
  (fun _ => (abs_of_pos (by positivity)).le) (le_refl |(0 : ℝ)|)

/-- `Σ_k a_k y^k ≤ Σ_k b_k y^k` when `a_k ≤ b_k` and `y ≥ 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem powerSum_le_powerSum (ha : IsEntire a) (hb : IsEntire b) (hab : ∀ k, a k ≤ b k)
    {y : ℝ} (hy : 0 ≤ y) : powerSum a y ≤ powerSum b y :=
  (ha.summable y).tsum_le_tsum (fun k => by gcongr; exact hab k) (hb.summable y)

example := powerSum_le_powerSum isEntire_inv_factorial isEntire_inv_factorial
  (fun _ => le_rfl) zero_le_one

/-- A term `a_j y^j` of a series of nonnegative coefficients is at most its sum, for `y ≥ 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem le_powerSum (ha : IsEntire a) (h0 : ∀ k, 0 ≤ a k) {y : ℝ} (hy : 0 ≤ y) (j : ℕ) :
    a j * y ^ j ≤ powerSum a y :=
  le_hasSum (ha.summable y).hasSum j fun k _ => mul_nonneg (h0 k) (pow_nonneg hy k)

example := le_powerSum isEntire_inv_factorial (fun _ => by positivity) zero_le_one 0

/-- A series of nonnegative coefficients has a nonnegative sum, for `y ≥ 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem powerSum_nonneg (h0 : ∀ k, 0 ≤ a k) {y : ℝ} (hy : 0 ≤ y) : 0 ≤ powerSum a y :=
  tsum_nonneg fun k => mul_nonneg (h0 k) (pow_nonneg hy k)

example := powerSum_nonneg (a := fun k => 1 / k.factorial) (fun _ => by positivity)
  zero_le_one

end Kinetic
end Transformer
