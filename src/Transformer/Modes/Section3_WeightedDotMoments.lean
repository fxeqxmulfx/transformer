import Transformer.Modes.Section3_CharacteristicDerivatives
import Transformer.Modes.Section3_FourthOrderPhase

/-!
# Weighted dot moments for characteristic Taylor derivatives

The differentiated expansion in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, uses a product of direction dot products as the weight
of the unit phase. A Taylor polynomial of degree `m` for derivative
order `k` therefore needs moment order `k + m + 1`.

The estimates below prove integrability of each signed polynomial
coefficient and of its absolute remainder. All coefficients remain
actual integrals of the law, with the powers of `i` and factorials
retained. In particular, taking `k + m = 3` gives fourth-moment
remainders for every derivative used in the cubic Edgeworth bound.

The factor `2^(k + m + 1)` comes from the product sup norm on `ℝ²`.
No exponential moment or extra density hypothesis enters these
single-summand estimates.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Products of direction dots and a phase power are integrable at their total moment order.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the weighted moments
in the characteristic derivative expansion. The absolute integral
retains every direction norm and the exact total moment order. -/
theorem integrable_prod_dot_mul_pow_and_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (k j : ℕ)
    (hmom : MemLp id (k + j : ℕ) μ) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    Integrable (fun z => (∏ i, (z.1 * (v i).1 + z.2 * (v i).2)) *
      (z.1 * ξ.1 + z.2 * ξ.2) ^ j) μ ∧
    Integrable (fun z => |∏ i, (z.1 * (v i).1 + z.2 * (v i).2)| *
      |z.1 * ξ.1 + z.2 * ξ.2| ^ j) μ ∧
    (∫ z, |∏ i, (z.1 * (v i).1 + z.2 * (v i).2)| *
      |z.1 * ξ.1 + z.2 * ξ.2| ^ j ∂μ) ≤
      2 ^ (k + j) * (∏ i, ‖v i‖) * ‖ξ‖ ^ j * ∫ z, ‖z‖ ^ (k + j) ∂μ := by
  let C : ℝ := 2 ^ (k + j) * (∏ i, ‖v i‖) * ‖ξ‖ ^ j
  have hp (z : ℝ × ℝ) : |∏ i, (z.1 * (v i).1 + z.2 * (v i).2)| *
      |z.1 * ξ.1 + z.2 * ξ.2| ^ j ≤ C * ‖z‖ ^ (k + j) := by
    have hw : |∏ i, (z.1 * (v i).1 + z.2 * (v i).2)| ≤
        (2 * ‖z‖) ^ k * ∏ i, ‖v i‖ := by
      rw [Finset.abs_prod]
      calc
        _ ≤ ∏ i : Fin k, (2 * ‖z‖) * ‖v i‖ :=
          Finset.prod_le_prod₀ (fun _ _ => abs_nonneg _)
            (fun i _ => abs_dot_le_two_norm_mul z (v i))
        _ = _ := by
          rw [Finset.prod_mul_distrib, Finset.prod_const]
          simp only [Finset.card_univ, Fintype.card_fin]
    calc
      _ ≤ ((2 * ‖z‖) ^ k * ∏ i, ‖v i‖) * (2 * ‖z‖ * ‖ξ‖) ^ j :=
        mul_le_mul hw (pow_le_pow_left₀ (abs_nonneg _)
          (abs_dot_le_two_norm_mul z ξ) j) (by positivity) (by positivity)
      _ = C * ‖z‖ ^ (k + j) := by
        dsimp [C]
        rw [pow_add, pow_add, mul_pow, mul_pow, mul_pow]
        ring
  have hM := hmom.integrable_norm_pow'.const_mul C
  have hs : Integrable (fun z => (∏ i, (z.1 * (v i).1 + z.2 * (v i).2)) *
      (z.1 * ξ.1 + z.2 * ξ.2) ^ j) μ := by
    apply hM.mono' (by fun_prop)
    exact ae_of_all _ fun z => by
      rw [Real.norm_eq_abs, abs_mul, abs_pow]
      exact hp z
  have ha : Integrable (fun z => |∏ i, (z.1 * (v i).1 + z.2 * (v i).2)| *
      |z.1 * ξ.1 + z.2 * ξ.2| ^ j) μ := by
    simpa only [abs_mul, abs_pow] using hs.abs
  refine ⟨hs, ha, ?_⟩
  calc
    _ ≤ ∫ z, C * ‖z‖ ^ (k + j) ∂μ :=
      integral_mono_ae ha hM (ae_of_all _ hp)
    _ = _ := integral_const_mul _ _

example (k j : ℕ) : IsProbabilityMeasure stdGauss2 ∧
    MemLp id (k + j : ℕ) stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Every characteristic derivative admits its actual weighted phase Taylor polynomial.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, before normalized-sum
powers. Derivative order `k` and polynomial degree `m` require only
moment `k + m + 1`; the denominator `m!` is deliberately nonsharp. -/
theorem norm_iteratedFDeriv_characteristic2_sub_phase_taylor
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (k m : ℕ)
    (hmom : MemLp id (k + m + 1 : ℕ) μ) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (characteristic2 μ) ξ v - Complex.I ^ k *
      ∑ j ∈ Finset.range (m + 1),
        ((∫ z, (∏ i, (z.1 * (v i).1 + z.2 * (v i).2)) *
          (z.1 * ξ.1 + z.2 * ξ.2) ^ j ∂μ : ℝ) : ℂ) *
          Complex.I ^ j / (j.factorial : ℂ)‖ ≤
      (2 ^ (k + m + 1) * (∏ i, ‖v i‖) * ‖ξ‖ ^ (m + 1) *
        ∫ z, ‖z‖ ^ (k + m + 1) ∂μ) / (m.factorial : ℝ) := by
  let w : ℝ × ℝ → ℝ := fun z => ∏ i, (z.1 * (v i).1 + z.2 * (v i).2)
  let f : ℝ × ℝ → ℝ := fun z => z.1 * ξ.1 + z.2 * ξ.2
  have hpoly (j : ℕ) (hj : j ∈ Finset.range (m + 1)) :
      Integrable (fun z => w z * f z ^ j) μ := by
    have hjm : j ≤ m := by have := Finset.mem_range.mp hj; omega
    have hjmom : MemLp id (k + j : ℕ) μ :=
      hmom.mono_exponent (by exact_mod_cast (show k + j ≤ k + m + 1 by omega))
    exact (integrable_prod_dot_mul_pow_and_bound μ k j hjmom ξ v).1
  have hnext : MemLp id (k + (m + 1) : ℕ) μ := by
    simpa only [Nat.add_assoc] using hmom
  have hM := integrable_prod_dot_mul_pow_and_bound μ k (m + 1) hnext ξ v
  have ht := norm_integral_real_weighted_phase_sub_taylor μ w f
    (by dsimp [f]; fun_prop) m hpoly hM.2.1
  rw [integral_real_weighted_phase_taylor μ w f m hpoly] at ht
  have hk : MemLp id (k : ℝ≥0∞) μ :=
    hmom.mono_exponent (by exact_mod_cast (show k ≤ k + m + 1 by omega))
  rw [iteratedFDeriv_characteristic2 μ hk, ← mul_sub, norm_mul,
    norm_pow, Complex.norm_I, one_pow, one_mul]
  calc
    _ ≤ (∫ z, |w z| * |f z| ^ (m + 1) ∂μ) / (m.factorial : ℝ) := ht
    _ ≤ _ := by
      apply div_le_div_of_nonneg_right _ (by positivity)
      simpa only [Nat.add_assoc] using hM.2.2

example (k m : ℕ) : IsProbabilityMeasure stdGauss2 ∧
    MemLp id (k + m + 1 : ℕ) stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- One further norm moment controls the change of the actual characteristic derivative from zero.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the constant
phase expansion. At order three this is the fourth-moment error
after subtracting the constant third derivative of the cubic term. -/
theorem norm_iteratedFDeriv_characteristic2_sub_zero_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (k : ℕ)
    (hmom : MemLp id (k + 1 : ℕ) μ) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (characteristic2 μ) ξ v -
      iteratedFDeriv ℝ k (characteristic2 μ) 0 v‖ ≤
      2 ^ (k + 1) * (∏ i, ‖v i‖) * ‖ξ‖ * ∫ z, ‖z‖ ^ (k + 1) ∂μ := by
  have h := norm_iteratedFDeriv_characteristic2_sub_phase_taylor μ k 0
    (by simpa using hmom) ξ v
  have hk : MemLp id (k : ℝ≥0∞) μ :=
    hmom.mono_exponent (by exact_mod_cast (show k ≤ k + 1 by omega))
  rw [iteratedFDeriv_characteristic2 μ hk 0 v]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero,
    zero_mul, Complex.exp_zero, mul_one, integral_complex_ofReal]
  simpa only [Nat.add_zero, zero_add, Finset.sum_range_succ, Finset.sum_range_zero,
    pow_zero, pow_one, Nat.factorial_zero, Nat.cast_one, mul_one, div_one] using h

example (k : ℕ) : IsProbabilityMeasure stdGauss2 ∧
    MemLp id (k + 1 : ℕ) stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The full characteristic derivative has the same finite-moment change bound.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the operator
form of the constant phase remainder before Gaussian damping. -/
theorem norm_iteratedFDeriv_characteristic2_sub_zero_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (k : ℕ)
    (hmom : MemLp id (k + 1 : ℕ) μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (characteristic2 μ) ξ -
      iteratedFDeriv ℝ k (characteristic2 μ) 0‖ ≤
      2 ^ (k + 1) * (∫ z, ‖z‖ ^ (k + 1) ∂μ) * ‖ξ‖ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ (k + 1) ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  calc
    _ ≤ 2 ^ (k + 1) * (∏ i, ‖v i‖) * ‖ξ‖ * ∫ z, ‖z‖ ^ (k + 1) ∂μ :=
      norm_iteratedFDeriv_characteristic2_sub_zero_apply_le μ k hmom ξ v
    _ = _ := by ring

example (k : ℕ) : IsProbabilityMeasure stdGauss2 ∧
    MemLp id (k + 1 : ℕ) stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
