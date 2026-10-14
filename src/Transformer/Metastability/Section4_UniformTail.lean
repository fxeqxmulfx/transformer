/-
# Actual directional tails for uniform spherical initialization

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The exact even moments of the original uniform coordinate give
the original directional tail bound by Markov's inequality.
Both signs are included in the actual absolute-coordinate event.

At half-degree `k`, its probability is at most
`(k/(d r²))^k`. If `4k ≤ d r²`, this is at most `4^(-k)`
and then `exp(-k)`. Choosing the actual half-degree above
`log(d)/32` gives the original dimension power `d^(-1/32)`.
The remaining finite-dimensional projection and product-sample
steps can use this bound with the original law and directions.
No tail estimate or Gaussian representation is assumed.
-/

import Transformer.Metastability.Section4_UniformMomentBounds
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds

open Real MeasureTheory Set

namespace Transformer.Metastability

/-- The original absolute-coordinate tail is bounded by the actual even
moment divided by its original threshold power.
Source: arXiv:2410.06833v1, §4, the directional projection estimate
in `prop: concentration unif`. Positive thresholds suffice for every degree. -/
theorem uniform_directional_tail_le_moment {d : ℕ}
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) (u : EucSpace d)
    (r : ℝ) (hr : 0 < r) (k : ℕ) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤
      uniformEvenMoment ν u k / r ^ (2 * k) := by
  let : IsProbabilityMeasure ν := hν.1
  have hc : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) u (x : EucSpace d) ^ (2 * k)) :=
    (continuous_const.inner continuous_subtype_val).pow (2 * k)
  have hM := mul_meas_ge_le_integral_of_nonneg
    (ae_of_all ν fun x => (even_two_mul k).pow_nonneg (inner (𝕜 := ℝ) u (x : EucSpace d)))
    (hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)) (r ^ (2 * k))
  have hsub : {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ⊆
      {x : SSphere d | r ^ (2 * k) ≤ inner (𝕜 := ℝ) u (x : EucSpace d) ^ (2 * k)} := by
    intro x hx
    have h := pow_le_pow_left₀ hr.le hx (2 * k)
    rwa [(even_two_mul k).pow_abs] at h
  apply (measureReal_mono hsub).trans
  apply (le_div_iff₀ (pow_pos hr (2 * k))).mpr
  simpa only [uniformEvenMoment, mul_comm] using hM

/-- The actual uniform tail has the original dimension and threshold
moment bound. Source: arXiv:2410.06833v1, §4, the scalar estimate
used to obtain the original orthogonal-projection concentration. -/
theorem uniform_directional_tail_le_pow {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (r : ℝ) (hr : 0 < r) (k : ℕ) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤
      ((k : ℝ) / ((d : ℝ) * r ^ 2)) ^ k := by
  calc
    _ ≤ uniformEvenMoment ν u k / r ^ (2 * k) := uniform_directional_tail_le_moment ν hν u r hr k
    _ ≤ ((k : ℝ) / (d : ℝ)) ^ k / r ^ (2 * k) :=
      div_le_div_of_nonneg_right (uniformEvenMoment_le_pow hd ν hν hu k) (pow_pos hr _).le
    _ = _ := by rw [pow_mul, ← div_pow, div_div]

/-- The same original moment tail bound holds for actual directions of
norm at most one, including zero directions.
Source: arXiv:2410.06833v1, §4, the finite-direction reduction of
the original orthogonal-projection concentration estimate. -/
theorem uniform_directional_tail_le_pow_of_norm_le_one {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) (u : EucSpace d)
    (hu : ‖u‖ ≤ 1) (r : ℝ) (hr : 0 < r) (k : ℕ) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤
      ((k : ℝ) / ((d : ℝ) * r ^ 2)) ^ k := by
  calc
    _ ≤ uniformEvenMoment ν u k / r ^ (2 * k) := uniform_directional_tail_le_moment ν hν u r hr k
    _ ≤ ((k : ℝ) / (d : ℝ)) ^ k / r ^ (2 * k) :=
      div_le_div_of_nonneg_right (uniformEvenMoment_le_pow_of_norm_le_one hd ν hν u hu k)
        (pow_pos hr _).le
    _ = _ := by rw [pow_mul, ← div_pow, div_div]

/-- The moment degree below one quarter of the original squared threshold
times dimension gives the original geometric decay.
Source: arXiv:2410.06833v1, §4, the logarithmic degree used in the
original uniform initialization concentration estimate. -/
theorem uniform_directional_tail_le_quarter_pow {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * r ^ 2) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤ (1 / 4 : ℝ) ^ k := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  have hbase : (k : ℝ) / ((d : ℝ) * r ^ 2) ≤ (1 / 4 : ℝ) := by
    apply (div_le_iff₀ (by positivity : 0 < (d : ℝ) * r ^ 2)).mpr
    linarith
  exact (uniform_directional_tail_le_pow hd ν hν hu r hr k).trans
    (pow_le_pow_left₀ (by positivity) hbase k)

/-- The actual geometric bound is at most the actual exponential bound.
Source: arXiv:2410.06833v1, §4, the exponential moment-degree estimate
used for the original uniform initialization concentration rate. -/
theorem quarter_pow_le_exp_neg (k : ℕ) : (1 / 4 : ℝ) ^ k ≤ Real.exp (-(k : ℝ)) := by
  have he : Real.exp 1 ≤ 4 := (Real.exp_one_lt_three.le).trans (by norm_num)
  have hbase : (1 / 4 : ℝ) ≤ Real.exp (-1) := by
    rw [one_div, Real.exp_neg]
    exact (inv_le_inv₀ (by norm_num) (Real.exp_pos 1)).mpr he
  calc
    _ ≤ Real.exp (-1) ^ k := pow_le_pow_left₀ (by norm_num) hbase k
    _ = _ := by rw [← Real.exp_nat_mul]; congr 1; ring

/-- The actual directional tail decays exponentially in the original
half-degree. Source: arXiv:2410.06833v1, §4,
the original logarithmic-degree projection-concentration estimate. -/
theorem uniform_directional_tail_le_exp {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * r ^ 2) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤ Real.exp (-(k : ℝ)) :=
  (uniform_directional_tail_le_quarter_pow hd ν hν hu r hr k hk).trans (quarter_pow_le_exp_neg k)

/-- An actual logarithmic half-degree gives the original dimension-power
tail, uniformly over the original unit directions.
Source: arXiv:2410.06833v1, §4, the scalar rate used in
`prop: concentration unif`; the finite-dimensional projection step follows. -/
theorem uniform_directional_tail_le_dimension_power {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * r ^ 2) (hlog : Real.log (d : ℝ) / 32 ≤ k) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤ (d : ℝ) ^ (-(1 : ℝ) / 32) := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  apply (uniform_directional_tail_le_exp hd ν hν hu r hr k hk).trans
  rw [Real.rpow_def_of_pos hdpos]
  apply Real.exp_le_exp.mpr
  linarith

/-- The actual dimension-power tail also applies to every actual
direction of norm at most one. Source: arXiv:2410.06833v1, §4,
the finite-direction projection bound for `prop: concentration unif`. -/
theorem uniform_directional_tail_le_dimension_power_of_norm_le_one {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) (u : EucSpace d)
    (hu : ‖u‖ ≤ 1) (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * r ^ 2) (hlog : Real.log (d : ℝ) / 32 ≤ k) :
    ν.real {x : SSphere d | r ≤ |inner (𝕜 := ℝ) u (x : EucSpace d)|} ≤ (d : ℝ) ^ (-(1 : ℝ) / 32) := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  have hbase : (k : ℝ) / ((d : ℝ) * r ^ 2) ≤ (1 / 4 : ℝ) := by
    apply (div_le_iff₀ (by positivity : 0 < (d : ℝ) * r ^ 2)).mpr
    linarith
  have hquarter := (uniform_directional_tail_le_pow_of_norm_le_one hd ν hν u hu r hr k).trans
    (pow_le_pow_left₀ (by positivity) hbase k)
  apply (hquarter.trans (quarter_pow_le_exp_neg k)).trans
  rw [Real.rpow_def_of_pos hdpos]
  apply Real.exp_le_exp.mpr
  linarith

/-- Positive dimension, the actual uniform law, a nonzero unit direction,
positive radius, degree scale and logarithmic scale are jointly realized
at `d=8`, `r=3/4`, `k=1`, inside the actual nonempty unit sphere. -/
example : (0 : ℕ) < 8 ∧ IsUniformOn 8 (MeanField.uniformLaw 8) ∧
    ‖(EuclideanSpace.single (0 : Fin 8) 1 : EucSpace 8)‖ = 1 ∧ (0 : ℝ) < 3 / 4 ∧
    4 * (1 : ℝ) ≤ 8 * (3 / 4 : ℝ) ^ 2 ∧ Real.log (8 : ℝ) / 32 ≤ 1 := by
  refine ⟨by decide, isUniformOn_uniformLaw (by norm_num), by simp, by norm_num, by norm_num, ?_⟩
  have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 8)
  linarith

end Transformer.Metastability
