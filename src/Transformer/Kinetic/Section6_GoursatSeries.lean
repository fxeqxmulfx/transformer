module

public import Transformer.Kinetic.Section6_ModeProfile

/-!
# Kinetic theory for Transformers — the actual Goursat solution series

The solution in arXiv:2605.09213v1, §6, proof of `prop:Bessel`, is
`U(t,y)=e^{λσ₀} Σ_k (aty)^k/(k!)²`. Its derivative in `y` is the
original `ψ_{at}(y)`, and the latter's derivative in time is
`a Σ_k (aty)^k/(k!)²`. We prove both identities from the actual entire
series. The coefficient identity `Σ(k+1)a_k z^k=Σa_k z^k+z(Σa_k z^k)'`
justifies the time derivative, including at `t=0`.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer
namespace Kinetic

/-- The actual squared-factorial coefficients of the source's Goursat series.
Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`, the formula for `U`. -/
noncomputable def squaredFactorialCoeffs (k : ℕ) : ℝ := 1 / (k.factorial * k.factorial)

/-- The actual Goursat series without the constant factor `e^{λσ₀}`.
Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`, `U(t,y)`. -/
noncomputable def goursatSeries (c y : ℝ) : ℝ := powerSum squaredFactorialCoeffs (c * y)

/-- The source's actual Goursat coefficients converge against every power.
Source: arXiv:2605.09213v1, §6, the series in the proof of `prop:Bessel`. -/
theorem isEntire_squaredFactorialCoeffs : IsEntire squaredFactorialCoeffs := by
  intro R hR
  refine Summable.of_nonneg_of_le (fun _ => by positivity) (fun k => ?_)
    (isEntire_inv_factorial R hR)
  have hk : (1 : ℝ) ≤ k.factorial := by exact_mod_cast Nat.factorial_pos k
  rw [abs_of_pos (by unfold squaredFactorialCoeffs; positivity),
    abs_of_pos (by positivity : (0 : ℝ) < 1 / k.factorial)]
  unfold squaredFactorialCoeffs
  gcongr
  simpa using mul_le_mul_of_nonneg_left hk (by positivity : (0 : ℝ) ≤ k.factorial)

/-- Weighting actual entire coefficients by `k+1` still gives an entire series.
Source: arXiv:2605.09213v1, §6, the differentiated Goursat series. -/
theorem IsEntire.succ_mul {a : ℕ → ℝ} (ha : IsEntire a) : IsEntire fun k => (k + 1) * a k := by
  intro R hR
  refine Summable.of_nonneg_of_le (fun _ => by positivity) (fun k => ?_)
    ((ha (R + 1) (by linarith)).mul_left (R + 1))
  rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ k + 1)]
  calc (k + 1 : ℝ) * |a k| * R ^ k = |a k| * ((k + 1) * R ^ k) := by ring
    _ ≤ |a k| * (R + 1) ^ (k + 1) :=
      mul_le_mul_of_nonneg_left (succ_mul_pow_le hR k) (abs_nonneg _)
    _ = _ := by ring

example := isEntire_inv_factorial.succ_mul

/-- The actual entire-series Euler identity used for the time derivative.
Source: arXiv:2605.09213v1, §6, the differentiations in the proof of `prop:Bessel`. -/
theorem powerSum_succ_mul {a : ℕ → ℝ} (ha : IsEntire a) (z : ℝ) :
    powerSum (fun k => (k + 1) * a k) z = powerSum a z + z * powerSum (derivCoeffs a) z := by
  have hs := ha.summable z
  have hw := ha.succ_mul.summable z
  have htail := (summable_nat_add_iff 1).mpr hs
  have hD := (ha.derivCoeffs.summable z).mul_left z
  unfold powerSum
  rw [hw.tsum_eq_zero_add, hs.tsum_eq_zero_add]
  simp only [CharP.cast_eq_zero, zero_add, pow_zero, mul_one, one_mul]
  rw [← tsum_mul_left, add_assoc, ← htail.tsum_add hD]
  congr 1
  apply tsum_congr
  intro k
  unfold derivCoeffs
  push_cast
  rw [pow_succ]
  ring

example := powerSum_succ_mul isEntire_inv_factorial 1

/-- The actual Bessel-derived series has the scaling `ψ_c(y)=c ψ_1(cy)`.
Source: arXiv:2605.09213v1, §6, the definition of `ψ_c` in `prop:Bessel`. -/
theorem psi_scale (c y : ℝ) : psi c y = c * psi 1 (c * y) := by
  unfold psi powerSum
  rw [← tsum_mul_left]
  congr 1
  ext k
  simp only [psiCoeffs, one_pow, mul_pow, pow_succ]
  ring

/-- The source's squared-factorial coefficient is `(k+1)` times the actual
unit-parameter psi coefficient. Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`. -/
theorem squaredFactorialCoeffs_eq (k : ℕ) :
    squaredFactorialCoeffs k = (k + 1) * psiCoeffs 1 k := by
  have hk : (k.factorial : ℝ) ≠ 0 := by positivity
  have hk1 : (k + 1 : ℝ) ≠ 0 := by positivity
  rw [squaredFactorialCoeffs, psiCoeffs, Nat.factorial_succ]
  push_cast
  simp only [one_pow]
  field_simp

/-- The actual Goursat series is the psi series plus its Euler derivative term.
Source: arXiv:2605.09213v1, §6, the time derivative in `prop:Bessel`. -/
theorem goursatSeries_euler (c y : ℝ) :
    goursatSeries c y = psi 1 (c * y) + (c * y) * psiDeriv 1 (c * y) := by
  rw [goursatSeries, show squaredFactorialCoeffs = fun k => (k + 1) * psiCoeffs 1 k from
    funext squaredFactorialCoeffs_eq, powerSum_succ_mul (isEntire_psiCoeffs 1)]
  rfl

/-- The actual derivative in time of the original `ψ_{at}(y)`.
Source: arXiv:2605.09213v1, §6, the equation derived for `g_a` in `prop:Bessel`. -/
theorem hasDerivAt_psi_time (a y t : ℝ) :
    HasDerivAt (fun t => psi (a * t) y) (a * goursatSeries (a * t) y) t := by
  have hi := ((hasDerivAt_id' (x := t)).const_mul a).mul_const y
  have hp := (hasDerivAt_powerSum (isEntire_psiCoeffs 1) (a * t * y)).comp t hi
  have h := ((hasDerivAt_id' (x := t)).const_mul a).mul hp
  convert h using 1
  · funext z
    exact psi_scale (a * z) y
  · rw [goursatSeries_euler]
    unfold psi psiDeriv
    simp only [Function.comp_apply]
    ring

/-- The derivative of the actual Goursat coefficients in its argument
is exactly the unit-parameter psi coefficient.
Source: arXiv:2605.09213v1, §6, the spatial derivative of `U` in `prop:Bessel`. -/
theorem derivCoeffs_squaredFactorialCoeffs (k : ℕ) :
    derivCoeffs squaredFactorialCoeffs k = psiCoeffs 1 k := by
  have hk : (k.factorial : ℝ) ≠ 0 := by positivity
  have hk1 : (k + 1 : ℝ) ≠ 0 := by positivity
  unfold derivCoeffs squaredFactorialCoeffs psiCoeffs
  rw [Nat.factorial_succ]
  push_cast
  simp only [one_pow]
  field_simp

/-- The actual spatial derivative of the source's Goursat series is `ψ_c`.
Source: arXiv:2605.09213v1, §6, the spatial derivative of `U` in `prop:Bessel`. -/
theorem hasDerivAt_goursatSeries (c y : ℝ) : HasDerivAt (goursatSeries c) (psi c y) y := by
  have h := (hasDerivAt_powerSum isEntire_squaredFactorialCoeffs (c * y)).comp y
    ((hasDerivAt_id' (x := y)).const_mul c)
  convert h using 1
  · rfl
  · rw [show derivCoeffs squaredFactorialCoeffs = psiCoeffs 1 from
      funext derivCoeffs_squaredFactorialCoeffs, psi_scale]
    unfold psi
    ring

/-- At zero spatial argument the actual Goursat series takes its source boundary value.
Source: arXiv:2605.09213v1, §6, `U(t,0)=e^{λσ₀}` in `prop:Bessel`. -/
theorem goursatSeries_zero (c : ℝ) : goursatSeries c 0 = 1 := by
  rw [goursatSeries, mul_zero, powerSum_zero]
  norm_num [squaredFactorialCoeffs]

end Kinetic
end Transformer
