module

public import Transformer.Homogenized.Section7_CorrelationTaylor
public import Transformer.Homogenized.Defs

/-!
# A global cubic remainder for the actual normalized correlation

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the Taylor-Lagrange
expansion of normalized Gaussian query correlation. The local Taylor
bound extends globally for actual query vectors by Cauchy–Schwarz.
The Gram inequality is checked on the actual vector inner products,
without assuming a probabilistic covariance or normalization identity.

The genuine quadratic Taylor polynomial has absolute value at most
`1 + 2S + 2S²`, where `S = |w| + |u| + |v|`. Outside the local
event at least one deviation exceeds `1/2`, so `S ≥ 1/2` and
the polynomial and bounded normalized correlation are controlled by
`40S³`. Inside the local event the proved cubic Taylor estimate
already applies with the same constant.

The resulting estimate is pointwise for the actual normalized inner
product of any two vectors. It includes zero vectors under the
original normalization definition. No Gaussian covariance or moment
estimate is assumed; taking expectations will require actual third
absolute deviation moments of the original matrix-product queries.
-/

@[expose] public section

open Real

namespace Transformer.Homogenized

/-- A genuine Gram entry divided by the square root of its norm
product is bounded, including zero denominator. Source:
arXiv:2604.01978v1, §7.3, normalized correlation in `lem:delta_method`. -/
theorem delta_ratio_abs_le_one (a b : ℝ) (hab : a ^ 2 ≤ b) :
    |a / Real.sqrt b| ≤ 1 := by
  by_cases ht : Real.sqrt b = 0
  · simp [ht]
  have hb : 0 ≤ b := (sq_nonneg a).trans hab
  have htpos : 0 < Real.sqrt b := lt_of_le_of_ne (Real.sqrt_nonneg b) (fun h => ht h.symm)
  rw [abs_div, abs_of_pos htpos]
  apply (div_le_one htpos).mpr
  exact (Real.le_sqrt (abs_nonneg a) hb).mpr (by simpa only [sq_abs] using hab)

/-- The actual quadratic polynomial has a global quadratic growth
bound. Source: arXiv:2604.01978v1, §7.3, the Hessian polynomial
in `lem:delta_method`, including its linear terms. -/
theorem delta_correlation_taylor_abs_bound (r w u v : ℝ) (hr : |r| ≤ 1) :
    |deltaCorrelationTaylor r w u v| ≤
      1 + 2 * (|w| + |u| + |v|) + 2 * (|w| + |u| + |v|) ^ 2 := by
  let S := |w| + |u| + |v|
  let B := 3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4
  let L := r + w + (-(r * (u + v) / 2)) + (-(w * (u + v) / 2))
  have hS : 0 ≤ S := by dsimp [S]; positivity
  have hwS : |w| ≤ S := by dsimp [S]; linarith [abs_nonneg u, abs_nonneg v]
  have huS : |u| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg v]
  have hvS : |v| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg u]
  have huv : |u + v| ≤ S := (abs_add_le u v).trans (by dsimp [S]; linarith [abs_nonneg w])
  have hu2 : u ^ 2 ≤ S ^ 2 := by simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg u) huS 2
  have hv2 : v ^ 2 ≤ S ^ 2 := by simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg v) hvS 2
  have huv2 : |u * v| ≤ S ^ 2 := by
    rw [abs_mul, pow_two]
    exact mul_le_mul huS hvS (abs_nonneg v) hS
  have hB : |B| ≤ S ^ 2 := by
    have h := abs_add_le (3 * (u ^ 2 + v ^ 2) / 8) (u * v / 4)
    rw [abs_of_nonneg (by positivity : 0 ≤ 3 * (u ^ 2 + v ^ 2) / 8), abs_div] at h
    norm_num at h
    simp only [abs_mul] at huv2
    dsimp [B]
    linarith
  have hrterm : |r * (u + v) / 2| ≤ S / 2 := by
    rw [abs_div, abs_mul]
    norm_num
    have h := mul_le_mul hr huv (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)
    linarith
  have hwterm : |w * (u + v) / 2| ≤ S ^ 2 / 2 := by
    rw [abs_div, abs_mul]
    norm_num
    have h := mul_le_mul hwS huv (abs_nonneg _) hS
    nlinarith
  have hrB : |r * B| ≤ S ^ 2 := by
    rw [abs_mul]
    exact (mul_le_mul hr hB (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _)
  have hbase : |r + w| ≤ 1 + S := (abs_add_le r w).trans (add_le_add hr hwS)
  have hL : |L| ≤ 1 + S + S / 2 + S ^ 2 / 2 := by
    have h := abs_add_three (r + w) (-(r * (u + v) / 2)) (-(w * (u + v) / 2))
    simp only [abs_neg] at h
    exact h.trans (add_le_add (add_le_add hbase hrterm) hwterm)
  have he : deltaCorrelationTaylor r w u v = L + r * B := by
    dsimp [deltaCorrelationTaylor, L, B]; ring
  rw [he]
  have h := (abs_add_le L (r * B)).trans (add_le_add hL hrB)
  nlinarith [sq_nonneg S]

/-- **Global cubic remainder on genuine Gram data.** Cauchy–Schwarz
replaces any local event or exceptional-set assumption in this
pointwise estimate. Source: arXiv:2604.01978v1, §7.3,
the remainder in `lem:delta_method`, before integrating its actual moments. -/
theorem delta_correlation_taylor_global_bound (r w u v : ℝ)
    (hr : |r| ≤ 1) (hgram : (r + w) ^ 2 ≤ (1 + u) * (1 + v)) :
    |(r + w) / Real.sqrt ((1 + u) * (1 + v)) - deltaCorrelationTaylor r w u v| ≤
      40 * (|w| + |u| + |v|) ^ 3 := by
  by_cases hlocal : |w| ≤ 1 / 2 ∧ |u| ≤ 1 / 2 ∧ |v| ≤ 1 / 2
  · exact delta_correlation_taylor_bound r w u v hr hlocal.1 hlocal.2.1 hlocal.2.2
  let S := |w| + |u| + |v|
  have hS : 0 ≤ S := by dsimp [S]; positivity
  have hwS : |w| ≤ S := by dsimp [S]; linarith [abs_nonneg u, abs_nonneg v]
  have huS : |u| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg v]
  have hvS : |v| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg u]
  have hlarge : 1 / 2 ≤ S := by
    by_contra h
    have hs : S < 1 / 2 := lt_of_not_ge h
    exact hlocal ⟨hwS.trans hs.le, huS.trans hs.le, hvS.trans hs.le⟩
  have hratiob := delta_ratio_abs_le_one (r + w) ((1 + u) * (1 + v)) hgram
  have hpoly := delta_correlation_taylor_abs_bound r w u v hr
  have he := (abs_sub ((r + w) / Real.sqrt ((1 + u) * (1 + v)))
    (deltaCorrelationTaylor r w u v)).trans (add_le_add hratiob hpoly)
  have hp := mul_nonneg (sub_nonneg.mpr hlarge)
    (by positivity : 0 ≤ 40 * S ^ 2 + 18 * S + 7)
  change _ ≤ 40 * S ^ 3
  nlinarith

/-- The actual normalized vector correlation has the global cubic
remainder, including zero queries. Source: arXiv:2604.01978v1,
§7.3, the normalized-query observable in `lem:delta_method`. -/
theorem normalized_inner_delta_taylor_bound {d : ℕ} (r : ℝ) (hr : |r| ≤ 1)
    (p q : EucSpace d) :
    |inner ℝ (normalizeLayer p) (normalizeLayer q) -
      deltaCorrelationTaylor r (inner ℝ p q - r) (‖p‖ ^ 2 - 1) (‖q‖ ^ 2 - 1)| ≤
        40 * (|inner ℝ p q - r| + |‖p‖ ^ 2 - 1| + |‖q‖ ^ 2 - 1|) ^ 3 := by
  have hc := pow_le_pow_left₀ (abs_nonneg (inner ℝ p q)) (abs_real_inner_le_norm p q) 2
  simp only [sq_abs, mul_pow] at hc
  have hgram : (r + (inner ℝ p q - r)) ^ 2 ≤ (1 + (‖p‖ ^ 2 - 1)) * (1 + (‖q‖ ^ 2 - 1)) := by
    convert hc using 1 <;> ring
  have h := delta_correlation_taylor_global_bound r (inner ℝ p q - r) (‖p‖ ^ 2 - 1)
    (‖q‖ ^ 2 - 1) hr hgram
  have hs : Real.sqrt ((1 + (‖p‖ ^ 2 - 1)) * (1 + (‖q‖ ^ 2 - 1))) = ‖p‖ * ‖q‖ := by
    rw [show 1 + (‖p‖ ^ 2 - 1) = ‖p‖ ^ 2 by ring,
      show 1 + (‖q‖ ^ 2 - 1) = ‖q‖ ^ 2 by ring, ← mul_pow,
      Real.sqrt_sq (mul_nonneg (norm_nonneg p) (norm_nonneg q))]
  have hn : inner ℝ (normalizeLayer p) (normalizeLayer q) = inner ℝ p q / (‖p‖ * ‖q‖) := by
    simp only [normalizeLayer, real_inner_smul_left, real_inner_smul_right, div_eq_mul_inv, mul_inv]
    ring
  rw [show r + (inner ℝ p q - r) = inner ℝ p q by ring, hs, ← hn] at h
  exact h

/-- A nontrivial correlation and actual collinear Gram data satisfy
the global hypotheses jointly far outside the local Taylor region. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧ ((1 / 2 : ℝ) + 11 / 2) ^ 2 ≤ (1 + 3) * (1 + 8) ∧
    (6 : ℝ) ^ 2 ≤ 36 ∧ |(11 / 2 : ℝ)| > 1 / 2 := by norm_num

end Transformer.Homogenized
