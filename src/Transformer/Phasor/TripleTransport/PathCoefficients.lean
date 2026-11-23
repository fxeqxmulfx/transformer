module

public import Transformer.Phasor.TripleTransport.TypedLookupPaths
public import Mathlib.Analysis.Calculus.Deriv.Pow

/-!
# Contributions of learned coefficients to an associative read

For the repeated lookup experiment in `phasor_value_transport`, freeze the
learned addresses, the hidden state and its gate at one causal prefix.
For two output classes, each coefficient multiplies the total signed
difference of their tied logits over the endpoints in its bin. Several
witnesses can share a bin or an endpoint. The remaining network contributes
a separate margin.

The experiment compares masking coefficients by magnitude with a common
cubic gain. These are properties of the proposed operations on an arbitrary
finite coefficient table.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.PathCoefficients

open scoped BigOperators
variable {B : Type*} [Fintype B]

def margin (w v : B → ℝ) : ℝ := ∑ i, w i * v i

def mask (keep : B → Bool) (w : B → ℝ) (i : B) : ℝ := if keep i then w i else 0

/-- For the frozen lookup ablation the change in the margin is exactly the
signed contribution of the omitted bins, including all hash collisions. -/
theorem split (keep : B → Bool) (w v : B → ℝ) :
    margin w v = margin (mask keep w) v + margin (mask (fun i => !keep i) w) v := by
  unfold margin
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases h : keep i <;> simp [mask, h]

/-- A positive margin in the complete model survives the ablation if it
exceeds the sum of the omitted absolute contributions. -/
theorem preserves (keep : B → Bool) (w v : B → ℝ) (base : ℝ)
    (gap : (∑ i, |mask (fun j => !keep j) w i| * |v i|) < base + margin w v) :
    0 < base + margin (mask keep w) v := by
  have bound := Finset.abs_sum_le_sum_abs (fun i => mask (fun j => !keep j) w i * v i) Finset.univ
  simp only [abs_mul] at bound
  have diff := (le_abs_self (margin (mask (fun i => !keep i) w) v)).trans bound
  rw [split keep w v] at gap
  linarith

example : 0 < (1 : ℝ) + margin (mask id (fun b : Bool => if b then 3 else 1)) (fun _ => 1) := by
  apply preserves
  norm_num [margin, mask, Fintype.sum_bool]

/-- The common denominator uses the largest magnitude in the table. The
experiment sets the positive floor to the reciprocal of the table size. -/
noncomputable def gain (m b w : ℝ) : ℝ := w * ((w ^ 2 + b) / (m ^ 2 + b))

/-- The normalized lookup gain preserves the largest coefficient of either sign. -/
theorem gain_of_maximal (m b w : ℝ) (floor : 0 < b) (maximal : |w| = m) :
    gain m b w = w := by
  have sq : w ^ 2 = m ^ 2 := by rw [← maximal, sq_abs]
  have pos : 0 < m ^ 2 + b := by positivity
  simp [gain, sq, pos.ne']

/-- Every stronger coefficient gains relative magnitude against a strictly
weaker nonzero competitor. Cross multiplication also handles negative signs. -/
theorem gain_separates (m b a c : ℝ) (floor : 0 < b) (live : 0 < |c|) (order : |c| < |a|) :
    |a| * |gain m b c| < |c| * |gain m b a| := by
  have denom : 0 < m ^ 2 + b := by positivity
  have ac : c ^ 2 < a ^ 2 := by
    nlinarith [sq_abs c, sq_abs a, mul_pos (sub_pos.mpr order) (add_pos live (live.trans order))]
  have cc : 0 < (c ^ 2 + b) / (m ^ 2 + b) := by positivity
  have aa : 0 < (a ^ 2 + b) / (m ^ 2 + b) := by positivity
  simp only [gain, abs_mul, abs_of_pos cc, abs_of_pos aa]
  field_simp [denom.ne']
  nlinarith [mul_pos (mul_pos live (live.trans order)) (sub_pos.mpr ac)]

/-- The gain differs from the original coordinate by at most a cubic
error in the maximum norm, uniformly across the table. -/
theorem gain_error (m b w : ℝ) (floor : 0 < b) (bound : |w| ≤ m) :
    |gain m b w - w| ≤ m ^ 3 / b := by
  have nonneg : 0 ≤ m := (abs_nonneg _).trans bound
  have sq : w ^ 2 ≤ m ^ 2 := by
    nlinarith [sq_abs w, mul_nonneg (sub_nonneg.mpr bound) (add_nonneg nonneg (abs_nonneg w))]
  have denom : 0 < m ^ 2 + b := by positivity
  have rewrite : gain m b w - w = -w * ((m ^ 2 - w ^ 2) / (m ^ 2 + b)) := by
    unfold gain
    field_simp
    ring
  rw [rewrite, abs_mul, abs_neg, abs_of_nonneg (div_nonneg (sub_nonneg.mpr sq) denom.le)]
  have size : (m ^ 2 - w ^ 2) / (m ^ 2 + b) ≤ m ^ 2 / b := by
    apply (div_le_div_iff₀ denom floor).2
    nlinarith [mul_nonneg (sq_nonneg w) floor.le, mul_self_nonneg (m ^ 2)]
  calc
    _ ≤ m * (m ^ 2 / b) := mul_le_mul bound size (div_nonneg (sub_nonneg.mpr sq) denom.le) nonneg
    _ = _ := by ring

/-- The zero initialization passes every directional derivative through.
Along a table direction with largest magnitude m, the squared maximum
after scaling by t is (t*m)² for either sign of t. -/
theorem gain_initial_derivative (m b w : ℝ) (floor : 0 < b) :
    HasDerivAt (fun t : ℝ => gain (t * m) b (t * w)) w 0 := by
  have linear (v : ℝ) : HasDerivAt (fun t : ℝ => t * v) v 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).mul_const v
  have quad (v : ℝ) : HasDerivAt (fun t : ℝ => (t * v) ^ 2 + b) 0 0 := by
    simpa using ((linear v).fun_pow 2).add_const b
  have denom : (0 * m) ^ 2 + b ≠ 0 := by simpa using floor.ne'
  simpa [gain, floor.ne'] using (linear w).fun_mul ((quad w).fun_div (quad m) denom)

example : gain 3 1 (-3) = -3 := gain_of_maximal 3 1 (-3) (by norm_num) (by norm_num)

example : |(3 : ℝ)| * |gain 4 1 (-2)| < |(-2 : ℝ)| * |gain 4 1 3| :=
  gain_separates 4 1 3 (-2) (by norm_num) (by norm_num) (by norm_num)

example : |gain 3 2 (-1) - (-1)| ≤ 3 ^ 3 / 2 := gain_error _ _ _ (by norm_num) (by norm_num)

example : HasDerivAt (fun t : ℝ => gain (t * 3) 2 (t * (-1))) (-1) 0 :=
  gain_initial_derivative 3 2 (-1) (by norm_num)

end Transformer.Phasor.TripleTransport.PathCoefficients
