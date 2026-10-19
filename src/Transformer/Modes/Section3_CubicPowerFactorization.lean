module

public import Transformer.Modes.Section3_CubicFrequency
public import Transformer.Modes.Section3_ThirdPowerBounds
public import Mathlib.Algebra.Ring.GeomSum

/-!
# Exact factorization of the cubic-corrected power error

The differentiated comparison in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, requires the error itself, rather than separate large
derivatives of its terms. Two geometric-sum identities factor that
error into a fourth-order one-summand remainder and a product of two
third-order factors. The remaining factors are finite sums of mixed
powers. Their total degrees are `n-1` and `n-2`, respectively.

Every factor below is a literal polynomial in its arguments. No
division by the difference of the characteristic functions is used,
so the identities remain valid when both characteristic values agree.
The normalized-sum specialization keeps the actual Gaussian damping
and the exact cubic coefficient scale `1 / sqrt n`.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The second geometric quotient retains every mixed power of total degree `n-2`.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, an exact
algebraic identity used to differentiate the cubic power remainder. -/
theorem complex_geometric_sum_sub_linear_factorization (x y : ℂ) (n : ℕ) :
    (x - y) * (∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i,
      x ^ j * y ^ (n - 2 - j)) =
    (∑ i ∈ Finset.range n, x ^ i * y ^ (n - 1 - i)) -
      (n : ℂ) * y ^ (n - 1) := by
  have hterm (i : ℕ) (hi : i ∈ Finset.range n) :
      (x - y) * (∑ j ∈ Finset.range i, x ^ j * y ^ (n - 2 - j)) =
        (x ^ i - y ^ i) * y ^ (n - 1 - i) := by
    by_cases hi0 : i = 0
    · subst i
      simp
    have hiN : i < n := Finset.mem_range.mp hi
    calc
      _ = ((∑ j ∈ Finset.range i, x ^ j * y ^ (i - 1 - j)) *
          (x - y)) * y ^ (n - 1 - i) := by
        rw [Finset.mul_sum, Finset.sum_mul, Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro j hj
        have hjI : j < i := Finset.mem_range.mp hj
        have he : i - 1 - j + (n - 1 - i) = n - 2 - j := by omega
        calc
          _ = (x - y) * x ^ j * (y ^ (i - 1 - j) * y ^ (n - 1 - i)) := by
            rw [← pow_add, he]
            ring
          _ = _ := by ring
      _ = _ := by rw [geom_sum₂_mul]
  rw [Finset.mul_sum]
  calc
    _ = ∑ i ∈ Finset.range n, (x ^ i - y ^ i) * y ^ (n - 1 - i) :=
      Finset.sum_congr rfl hterm
    _ = _ := by
      simp_rw [sub_mul]
      rw [Finset.sum_sub_distrib, geom_sum₂_self]

/-- The actual corrected power error has a fourth-order factor and a quadratic third-order factor.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
algebraic cancellation before derivative and Gaussian bounds. -/
theorem complex_power_sub_cubic_factorization (x y b : ℂ) (n : ℕ) :
    x ^ n - y ^ n - (n : ℂ) * y ^ (n - 1) * b =
      (x - y - b) * (∑ i ∈ Finset.range n, x ^ i * y ^ (n - 1 - i)) +
        b * (x - y) * (∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i,
          x ^ j * y ^ (n - 2 - j)) := by
  have hfirst := geom_sum₂_mul x y n
  have hsecond := complex_geometric_sum_sub_linear_factorization x y n
  calc
    _ = (∑ i ∈ Finset.range n, x ^ i * y ^ (n - 1 - i)) * (x - y) -
        (n : ℂ) * y ^ (n - 1) * b := by rw [hfirst]
    _ = (x - y - b) * (∑ i ∈ Finset.range n, x ^ i * y ^ (n - 1 - i)) +
        b * ((∑ i ∈ Finset.range n, x ^ i * y ^ (n - 1 - i)) -
          (n : ℂ) * y ^ (n - 1)) := by ring
    _ = _ := by rw [← hsecond]; ring

/-- The normalized-sum correction has the exact factorization at the literal scaled frequency.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and
§5.4 `eq:br-9.10`. The factor `G B` retains the Gaussian
multiplication whose derivatives are part of the correction. -/
theorem characteristic_scaledSum_cubic_error_factorization
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {n : ℕ} (hn : 1 ≤ n)
    (ξ : ℝ × ℝ) :
    let η := (Real.sqrt n)⁻¹ • ξ
    let f := characteristic2 μ η
    let g := characteristic2 stdGauss2 η
    let b := g * cubicCharacteristicCorrection μ η
    characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
        characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ =
      (f - g - b) * (∑ i ∈ Finset.range n, f ^ i * g ^ (n - 1 - i)) +
        b * (f - g) * (∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i,
          f ^ j * g ^ (n - 2 - j)) := by
  dsimp only
  have hp : (n : ℂ) * characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
      (characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) *
        cubicCharacteristicCorrection μ ((Real.sqrt n)⁻¹ • ξ)) =
      (((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ *
        cubicCharacteristicCorrection μ ξ := by
    have he : n - 1 + 1 = n := by omega
    have hpower : characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) ^ n =
        characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
          characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) := by
      rw [← pow_succ, he]
    calc
      _ = ((n : ℂ) * cubicCharacteristicCorrection μ ((Real.sqrt n)⁻¹ • ξ)) *
          characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) ^ n := by
        rw [hpower]
        ring
      _ = _ := by
        rw [normalized_cubicCharacteristicCorrection μ hn ξ,
          characteristic2_stdGauss2_scaled_power n hn ξ]
        ring
  rw [characteristic_scaledSum, ← hp, ← characteristic2_stdGauss2_scaled_power n hn ξ]
  exact complex_power_sub_cubic_factorization _ _ _ n

example : IsProbabilityMeasure stdGauss2 ∧ 1 ≤ (2 : ℕ) :=
  ⟨inferInstance, by omega⟩

/-- The first mixed-power quotient preserves the actual functions' differentiability.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
finite geometric factor appearing in the differentiated error. -/
theorem contDiff_complex_geometric_sum (f g : ℝ × ℝ → ℂ) {k : ℕ}
    (hf : ContDiff ℝ k f) (hg : ContDiff ℝ k g) (n : ℕ) :
    ContDiff ℝ k (fun ξ => ∑ i ∈ Finset.range n,
      f ξ ^ i * g ξ ^ (n - 1 - i)) := by
  apply ContDiff.sum
  intro i hi
  exact (hf.pow i).mul (hg.pow (n - 1 - i))

example : ContDiff ℝ 3 (characteristic2 stdGauss2) ∧
    ContDiff ℝ 3 (characteristic2 stdGauss2) := by
  have h := contDiff_characteristic2 stdGauss2 (k := 3)
    (IsGaussian.memLp_id _ _ (by simp))
  exact ⟨h, h⟩

/-- The second mixed-power quotient remains smooth when the characteristic values coincide.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
finite polynomial factor for the product of third-order errors. -/
theorem contDiff_complex_second_geometric_sum (f g : ℝ × ℝ → ℂ) {k : ℕ}
    (hf : ContDiff ℝ k f) (hg : ContDiff ℝ k g) (n : ℕ) :
    ContDiff ℝ k (fun ξ => ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i,
      f ξ ^ j * g ξ ^ (n - 2 - j)) := by
  apply ContDiff.sum
  intro i hi
  apply ContDiff.sum
  intro j hj
  exact (hf.pow j).mul (hg.pow (n - 2 - j))

example : ContDiff ℝ 3 (characteristic2 stdGauss2) ∧
    ContDiff ℝ 3 (characteristic2 stdGauss2) := by
  have h := contDiff_characteristic2 stdGauss2 (k := 3)
    (IsGaussian.memLp_id _ _ (by simp))
  exact ⟨h, h⟩

end Transformer.Modes
