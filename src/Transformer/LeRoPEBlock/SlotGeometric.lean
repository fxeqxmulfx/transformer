module

public import Transformer.LeRoPEBlock.SlotOneSidedHead
public import Mathlib.Analysis.Complex.Trigonometric

/-!
# Geometric weights on keys to one side

Source: `review/lerope_block.md`, *Reading a slot* (added), RS3.

`SlotOneSidedHead` caps the weight on a key through weights `w` on keys to one side of it: their
entropy `H` and their defect, a bound `V` on `|1 − ∑ w e^{i e φ}|` at every frequency, `e` the
offsets. Geometric weights make both explicit. Put `(1 − r) r^j` at the offsets `j + 1 ≤ T` and
the rest, `r^T`, at `T + 1` (`geomWeight`); they sum to one (`sum_geomWeight`). The defect is a
norm on the unit circle (`defect_eq_norm_sq`), and with `z = e^{iφ}` it factors,
`1 − ∑ w z^{j+1} = (1 − z) ∑_{j ≤ T} (r z)^j` (`one_sub_sum_geomWeight`). Since
`(1 + r) |1 − z| ≤ 2 |1 − r z|` on the circle (`one_add_mul_norm_one_sub_le`) and
`(1 − r z) ∑_{j ≤ T} (r z)^j = 1 − (r z)^{T+1}`, the defect is at most `2 (1 + r^{T+1}) / (1 + r)`
(`defect_geomWeight_le`), and the entropy is at least `(1 − r^T)(r − ln(1 − r))`
(`le_entropy_geomWeight`).
-/

@[expose] public section

open Finset

namespace Transformer.LeRoPEBlock

/-- **The defect is a norm on the unit circle**: for weights at the offsets `e x`,
`(1 − ∑ w cos(e φ))² + (∑ w sin(e φ))² = |1 − ∑ w z^e|²`, `z = e^{iφ}`. -/
theorem defect_eq_norm_sq {ι : Type*} (S : Finset ι) (w : ι → ℝ) (e : ι → ℕ) (φ : ℝ) :
    (1 - ∑ x ∈ S, w x * Real.cos (e x * φ)) ^ 2 + (∑ x ∈ S, w x * Real.sin (e x * φ)) ^ 2 =
      ‖1 - ∑ x ∈ S, (w x : ℂ) * Complex.exp (φ * Complex.I) ^ e x‖ ^ 2 := by
  have hz (n : ℕ) :
      Complex.exp (φ * Complex.I) ^ n = Complex.exp (((n * φ : ℝ) : ℂ) * Complex.I) := by
    rw [← Complex.exp_nat_mul]
    congr 1
    push_cast
    ring
  simp only [hz, Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im,
    Complex.one_re, Complex.one_im, Complex.re_sum, Complex.im_sum, Complex.re_ofReal_mul,
    Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im]
  ring

/-- One offset of weight one: the defect is `|1 − e^{iφ}|²`. -/
example (φ : ℝ) := defect_eq_norm_sq (range 1) (fun _ => (1 : ℝ)) (fun _ => 1) φ

/-- **Geometric weights** on the offsets `j + 1`, `j ≤ T`: `(1 − r) r^j` for `j < T`, and the
rest, `r^T`, at `j = T`. -/
def geomWeight (r : ℝ) (T j : ℕ) : ℝ := if j < T then (1 - r) * r ^ j else r ^ T

lemma geomWeight_nonneg {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (T j : ℕ) : 0 ≤ geomWeight r T j := by
  unfold geomWeight
  split_ifs
  · exact mul_nonneg (by linarith) (pow_nonneg hr0 j)
  · exact pow_nonneg hr0 T

example : 0 ≤ geomWeight (1 / 2) 3 1 := geomWeight_nonneg (by norm_num) (by norm_num) 3 1

/-- The geometric weights sum to one. -/
theorem sum_geomWeight (r : ℝ) (T : ℕ) : ∑ j ∈ range (T + 1), geomWeight r T j = 1 := by
  have h : ∑ j ∈ range T, geomWeight r T j = ∑ j ∈ range T, (1 - r) * r ^ j :=
    sum_congr rfl fun j hj => ite_eq_left (mem_range.1 hj)
  rw [sum_range_succ, h, ← mul_sum, mul_neg_geom_sum, geomWeight, ite_eq_right (lt_irrefl T)]
  ring

/-- **The defect of geometric weights factors**:
`1 − ∑_{j ≤ T} w_j z^{j+1} = (1 − z) ∑_{j ≤ T} (r z)^j`. -/
theorem one_sub_sum_geomWeight (r : ℝ) (T : ℕ) (z : ℂ) :
    1 - ∑ j ∈ range (T + 1), (geomWeight r T j : ℂ) * z ^ (j + 1) =
      (1 - z) * ∑ j ∈ range (T + 1), ((r : ℂ) * z) ^ j := by
  have h : ∑ j ∈ range T, (geomWeight r T j : ℂ) * z ^ (j + 1) =
      (1 - r) * z * ∑ j ∈ range T, ((r : ℂ) * z) ^ j := by
    rw [mul_sum]
    refine sum_congr rfl fun j hj => ?_
    rw [geomWeight, ite_eq_left (mem_range.1 hj)]
    push_cast
    ring
  rw [sum_range_succ, h, geomWeight, ite_eq_right (lt_irrefl T), sum_range_succ]
  push_cast
  linear_combination (-1 : ℂ) * mul_neg_geom_sum ((r : ℂ) * z) T

example (z : ℂ) : 1 - ∑ j ∈ range (0 + 1), (geomWeight 0 0 j : ℂ) * z ^ (j + 1) =
    (1 - z) * ∑ j ∈ range (0 + 1), (((0 : ℝ) : ℂ) * z) ^ j :=
  one_sub_sum_geomWeight 0 0 z

/-- On the unit circle, `(1 + r) |1 − z| ≤ 2 |1 − r z|` for `r ≥ 0`. -/
lemma one_add_mul_norm_one_sub_le {r : ℝ} (hr : 0 ≤ r) (φ : ℝ) :
    (1 + r) * ‖1 - Complex.exp (φ * Complex.I)‖ ≤
      2 * ‖1 - (r : ℂ) * Complex.exp (φ * Complex.I)‖ := by
  have h1 : ‖1 - Complex.exp (φ * Complex.I)‖ ^ 2 = 2 - 2 * Real.cos φ := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    simp only [Complex.sub_re, Complex.sub_im, Complex.one_re, Complex.one_im,
      Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im]
    linear_combination Real.cos_sq_add_sin_sq φ
  have h2 : ‖1 - (r : ℂ) * Complex.exp (φ * Complex.I)‖ ^ 2 =
      1 - 2 * r * Real.cos φ + r ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    simp only [Complex.sub_re, Complex.sub_im, Complex.one_re, Complex.one_im,
      Complex.re_ofReal_mul, Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_re,
      Complex.exp_ofReal_mul_I_im]
    linear_combination r ^ 2 * Real.cos_sq_add_sin_sq φ
  have hc : 0 ≤ 1 + Real.cos φ := by linarith [Real.neg_one_le_cos φ]
  rw [← sq_le_sq₀ (mul_nonneg (by linarith) (norm_nonneg _)) (by positivity), mul_pow, mul_pow,
    h1, h2]
  nlinarith [mul_nonneg (sq_nonneg (1 - r)) hc]

example (φ : ℝ) : (1 + 0) * ‖1 - Complex.exp (φ * Complex.I)‖ ≤
    2 * ‖1 - ((0 : ℝ) : ℂ) * Complex.exp (φ * Complex.I)‖ :=
  one_add_mul_norm_one_sub_le le_rfl φ

/-- **The defect of geometric weights** is at most `2 (1 + r^{T+1}) / (1 + r)` at every
frequency. -/
theorem defect_geomWeight_le {r : ℝ} (hr : 0 ≤ r) (T : ℕ) (φ : ℝ) :
    (1 - ∑ j ∈ range (T + 1), geomWeight r T j * Real.cos (((j + 1 : ℕ) : ℝ) * φ)) ^ 2 +
      (∑ j ∈ range (T + 1), geomWeight r T j * Real.sin (((j + 1 : ℕ) : ℝ) * φ)) ^ 2 ≤
        (2 * (1 + r ^ (T + 1)) / (1 + r)) ^ 2 := by
  rw [defect_eq_norm_sq (range (T + 1)) (geomWeight r T) (· + 1) φ, one_sub_sum_geomWeight]
  set z := Complex.exp (φ * Complex.I)
  have hz : ‖z‖ = 1 := Complex.norm_exp_ofReal_mul_I φ
  have h1 : (1 + r) * ‖1 - z‖ ≤ 2 * ‖1 - (r : ℂ) * z‖ := one_add_mul_norm_one_sub_le hr φ
  have h2 : ‖1 - (r : ℂ) * z‖ * ‖∑ j ∈ range (T + 1), ((r : ℂ) * z) ^ j‖ ≤ 1 + r ^ (T + 1) := by
    rw [← norm_mul, mul_neg_geom_sum]
    refine (norm_sub_le _ _).trans ?_
    rw [norm_one, norm_pow, norm_mul, hz, mul_one, Complex.norm_real, Real.norm_of_nonneg hr]
  refine pow_le_pow_left₀ (norm_nonneg _) ?_ 2
  rw [le_div_iff₀ (by linarith), norm_mul]
  nlinarith [mul_le_mul_of_nonneg_right h1 (norm_nonneg (∑ j ∈ range (T + 1), ((r : ℂ) * z) ^ j))]

/-- All the weight on the offset `1` (`T = 0`): the defect is at most `2`. -/
example (φ : ℝ) := defect_geomWeight_le (r := 0) le_rfl 0 φ

/-- `∑_{j<T} (1 − r) j r^j + T r^T = ∑_{j<T} r^{j+1}`. -/
lemma sum_mul_pow_add (r : ℝ) (T : ℕ) :
    ∑ j ∈ range T, (1 - r) * j * r ^ j + T * r ^ T = ∑ j ∈ range T, r ^ (j + 1) := by
  induction T with
  | zero => simp
  | succ T ih =>
    rw [sum_range_succ, sum_range_succ, ← ih]
    push_cast
    ring

example := sum_mul_pow_add (1 / 2) 3

/-- **The entropy of geometric weights** is at least `(1 − r^T)(r − ln(1 − r))`. -/
theorem le_entropy_geomWeight {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) (T : ℕ) :
    (1 - r ^ T) * (r - Real.log (1 - r)) ≤
      -∑ j ∈ range (T + 1), geomWeight r T j * Real.log (geomWeight r T j) := by
  have hr' : 0 < 1 - r := by linarith
  have hj : ∀ j ∈ range T, geomWeight r T j * Real.log (geomWeight r T j) =
      (1 - r) * r ^ j * Real.log (1 - r) + (1 - r) * j * r ^ j * Real.log r := fun j hj => by
    rw [geomWeight, ite_eq_left (mem_range.1 hj), Real.log_mul hr'.ne' (pow_pos hr0 j).ne',
      Real.log_pow]
    ring
  have hT : geomWeight r T T * Real.log (geomWeight r T T) = T * r ^ T * Real.log r := by
    rw [geomWeight, ite_eq_right (lt_irrefl T), Real.log_pow]
    ring
  have hS : ∑ j ∈ range T, (1 - r) * r ^ j = 1 - r ^ T := by rw [← mul_sum, mul_neg_geom_sum]
  have hsum : -∑ j ∈ range (T + 1), geomWeight r T j * Real.log (geomWeight r T j) =
      -(1 - r ^ T) * Real.log (1 - r) - Real.log r * ∑ j ∈ range T, r ^ (j + 1) := by
    rw [sum_range_succ, sum_congr rfl hj, hT, sum_add_distrib, ← sum_mul, ← sum_mul, hS,
      ← sum_mul_pow_add r T]
    ring
  have hR : ∑ j ∈ range T, r ^ (j + 1) = r * ∑ j ∈ range T, r ^ j := by
    rw [mul_sum]
    exact sum_congr rfl fun j _ => pow_succ' r j
  have hP : 0 ≤ ∑ j ∈ range T, r ^ j := sum_nonneg fun j _ => pow_nonneg hr0.le j
  rw [hsum, hR, ← mul_neg_geom_sum r T]
  nlinarith [mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos hr0) (mul_nonneg hr0.le hP)]

/-- Half the weight on the offset `1`, half on `2`: entropy at least `(1/2)(1/2 + ln 2)`. -/
example := le_entropy_geomWeight (r := 1 / 2) (by norm_num) (by norm_num) 1

end Transformer.LeRoPEBlock
