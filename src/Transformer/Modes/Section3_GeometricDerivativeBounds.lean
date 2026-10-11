import Transformer.Modes.Section3_CubicProductDerivativeBounds

/-!
# Derivative bounds for the two finite geometric factors

The cubic factorization for arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
uses two polynomial geometric sums. The first has `n` mixed powers
of total degree `n-1`; the second has at most `n²` mixed powers of
total degree `n-2`. Their actual derivatives through order three
are finite sums of the mixed-power derivatives already bounded.

Taking norms before summing loses no Gaussian factor. A common
base modulus bound `r ≤ 1` leaves `r^(n-7)` in the first sum and
`r^(n-8)` in the second. The second count uses the simple upper
bound `n²`, rather than its smaller triangular cardinality. These
constants suffice for the literal normalized frequency scale.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first geometric factor retains Gaussian decay after all derivative terms are summed.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
fourth-order one-summand remainder's polynomial factor. -/
theorem norm_iteratedFDeriv_complex_geometric_sum_le
    (f g : ℝ × ℝ → ℂ) (hf : ContDiff ℝ 3 f) (hg : ContDiff ℝ 3 g)
    (ξ : ℝ × ℝ) (n : ℕ) {r K : ℝ} (hr0 : 0 ≤ r) (hr : r ≤ 1) (hK : 0 ≤ K)
    (hDf : ∀ m : ℕ, m ≤ n → ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => f η ^ m) ξ‖ ≤ K * r ^ (m - 3))
    (hDg : ∀ m : ℕ, m ≤ n → ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => g η ^ m) ξ‖ ≤ K * r ^ (m - 3))
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n,
      f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤ (n : ℝ) * 8 * K ^ 2 * r ^ (n - 7) := by
  have hc (i : ℕ) : ContDiff ℝ k (fun η => f η ^ i * g η ^ (n - 1 - i)) :=
    ((hf.pow i).mul (hg.pow (n - 1 - i))).of_le (by exact_mod_cast hk)
  rw [iteratedFDeriv_sum (fun i hi => hc i)]
  simp only [Finset.sum_apply]
  change ‖∑ i ∈ Finset.range n,
    iteratedFDeriv ℝ k (fun η => f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤ _
  calc
    _ ≤ ∑ i ∈ Finset.range n,
        ‖iteratedFDeriv ℝ k (fun η => f η ^ i * g η ^ (n - 1 - i)) ξ‖ := norm_sum_le _ _
    _ ≤ ∑ i ∈ Finset.range n, 8 * K ^ 2 * r ^ (n - 7) := by
      apply Finset.sum_le_sum
      intro i hi
      have hiN : i < n := Finset.mem_range.mp hi
      have h := norm_iteratedFDeriv_mixed_power_le f g hf hg ξ hr0 hr hK
        i (n - 1 - i) (hDf i (by omega)) (hDg (n - 1 - i) (by omega)) k hk
      have he : i + (n - 1 - i) - 6 = n - 7 := by omega
      simpa only [he] using h
    _ = _ := by simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; ring

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (1 : ℂ)) ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ m : ℕ, m ≤ 2 → ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j
      (fun _ : ℝ × ℝ => (1 : ℂ) ^ m) (0, 0)‖ ≤ 1 * (1 : ℝ) ^ (m - 3)) ∧
    (3 : ℕ) ≤ 3 := by
  refine ⟨contDiff_const, zero_le_one, le_rfl, zero_le_one, ?_, le_rfl⟩
  intro m hm j hj
  by_cases hj0 : j = 0
  · subst j
    simp only [norm_iteratedFDeriv_zero, one_pow, norm_one, one_mul, le_refl]
  · rw [iteratedFDeriv_const_of_ne hj0]
    simp only [Pi.zero_apply, norm_zero, one_pow, one_mul, zero_le_one]

/-- The second geometric factor retains Gaussian decay with its triangular term count bounded by `n²`.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
polynomial factor multiplying the two third-order errors. -/
theorem norm_iteratedFDeriv_complex_second_geometric_sum_le
    (f g : ℝ × ℝ → ℂ) (hf : ContDiff ℝ 3 f) (hg : ContDiff ℝ 3 g)
    (ξ : ℝ × ℝ) (n : ℕ) {r K : ℝ} (hr0 : 0 ≤ r) (hr : r ≤ 1) (hK : 0 ≤ K)
    (hDf : ∀ m : ℕ, m ≤ n → ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => f η ^ m) ξ‖ ≤ K * r ^ (m - 3))
    (hDg : ∀ m : ℕ, m ≤ n → ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => g η ^ m) ξ‖ ≤ K * r ^ (m - 3))
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n, ∑ a ∈ Finset.range i,
      f η ^ a * g η ^ (n - 2 - a)) ξ‖ ≤ (n : ℝ) ^ 2 * 8 * K ^ 2 * r ^ (n - 8) := by
  let R : ℝ := 8 * K ^ 2 * r ^ (n - 8)
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hc (a : ℕ) : ContDiff ℝ k (fun η => f η ^ a * g η ^ (n - 2 - a)) :=
    ((hf.pow a).mul (hg.pow (n - 2 - a))).of_le (by exact_mod_cast hk)
  have hcs (i : ℕ) : ContDiff ℝ k (fun η => ∑ a ∈ Finset.range i,
      f η ^ a * g η ^ (n - 2 - a)) := ContDiff.sum (fun a ha => hc a)
  rw [iteratedFDeriv_sum (fun i hi => hcs i)]
  simp only [Finset.sum_apply]
  change ‖∑ i ∈ Finset.range n, iteratedFDeriv ℝ k
    (fun η => ∑ a ∈ Finset.range i, f η ^ a * g η ^ (n - 2 - a)) ξ‖ ≤ _
  calc
    _ ≤ ∑ i ∈ Finset.range n, ‖iteratedFDeriv ℝ k
        (fun η => ∑ a ∈ Finset.range i, f η ^ a * g η ^ (n - 2 - a)) ξ‖ := norm_sum_le _ _
    _ ≤ ∑ i ∈ Finset.range n, (n : ℝ) * R := by
      apply Finset.sum_le_sum
      intro i hi
      have hiN : i < n := Finset.mem_range.mp hi
      rw [iteratedFDeriv_sum (fun a ha => hc a)]
      simp only [Finset.sum_apply]
      change ‖∑ a ∈ Finset.range i,
        iteratedFDeriv ℝ k (fun η => f η ^ a * g η ^ (n - 2 - a)) ξ‖ ≤ _
      calc
        _ ≤ ∑ a ∈ Finset.range i,
            ‖iteratedFDeriv ℝ k (fun η => f η ^ a * g η ^ (n - 2 - a)) ξ‖ := norm_sum_le _ _
        _ ≤ ∑ a ∈ Finset.range i, R := by
          apply Finset.sum_le_sum
          intro a ha
          have haI : a < i := Finset.mem_range.mp ha
          have h := norm_iteratedFDeriv_mixed_power_le f g hf hg ξ hr0 hr hK
            a (n - 2 - a) (hDf a (by omega)) (hDg (n - 2 - a) (by omega)) k hk
          have he : a + (n - 2 - a) - 6 = n - 8 := by omega
          simpa only [he] using h
        _ = (i : ℝ) * R := by simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        _ ≤ (n : ℝ) * R :=
          mul_le_mul_of_nonneg_right (by exact_mod_cast hiN.le) hR
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      dsimp [R]
      ring

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (1 : ℂ)) ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ m : ℕ, m ≤ 2 → ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j
      (fun _ : ℝ × ℝ => (1 : ℂ) ^ m) (0, 0)‖ ≤ 1 * (1 : ℝ) ^ (m - 3)) ∧
    (3 : ℕ) ≤ 3 := by
  refine ⟨contDiff_const, zero_le_one, le_rfl, zero_le_one, ?_, le_rfl⟩
  intro m hm j hj
  by_cases hj0 : j = 0
  · subst j
    simp only [norm_iteratedFDeriv_zero, one_pow, norm_one, one_mul, le_refl]
  · rw [iteratedFDeriv_const_of_ne hj0]
    simp only [Pi.zero_apply, norm_zero, one_pow, one_mul, zero_le_one]

/-- Base derivative bounds supply both geometric-factor bounds through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
common mixed-power estimate needed by the cubic-error product rule. -/
theorem norm_iteratedFDeriv_geometric_sums_le_of_base_derivative_bounds
    (f g : ℝ × ℝ → ℂ) (hf : ContDiff ℝ 3 f) (hg : ContDiff ℝ 3 g)
    (ξ : ℝ × ℝ) (n : ℕ) {r A P : ℝ} (hr : r ≤ 1)
    (hmodf : ‖f ξ‖ ≤ r) (hmodg : ‖g ξ‖ ≤ r)
    (hA : 0 ≤ A) (hP : 0 ≤ P) (hnA : (n : ℝ) * A ≤ P)
    (hDf : ∀ j : ℕ, 1 ≤ j → j ≤ 3 → ‖iteratedFDeriv ℝ j f ξ‖ ≤ A)
    (hDg : ∀ j : ℕ, 1 ≤ j → j ≤ 3 → ‖iteratedFDeriv ℝ j g ξ‖ ≤ A)
    (k : ℕ) (hk : k ≤ 3) :
    let K : ℝ := 1 + P + 4 * P ^ 2 + P ^ 3
    ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n,
      f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤ (n : ℝ) * 8 * K ^ 2 * r ^ (n - 7) ∧
    ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n, ∑ a ∈ Finset.range i,
      f η ^ a * g η ^ (n - 2 - a)) ξ‖ ≤ (n : ℝ) ^ 2 * 8 * K ^ 2 * r ^ (n - 8) := by
  have hr0 : 0 ≤ r := (norm_nonneg _).trans hmodf
  have hK : 0 ≤ 1 + P + 4 * P ^ 2 + P ^ 3 := by positivity
  have hF := norm_iteratedFDeriv_complex_power_le_of_derivative_bounds
    f hf ξ n hr hmodf hA hP hnA hDf
  have hG := norm_iteratedFDeriv_complex_power_le_of_derivative_bounds
    g hg ξ n hr hmodg hA hP hnA hDg
  exact ⟨norm_iteratedFDeriv_complex_geometric_sum_le f g hf hg ξ n hr0 hr hK hF hG k hk,
    norm_iteratedFDeriv_complex_second_geometric_sum_le f g hf hg ξ n hr0 hr hK hF hG k hk⟩

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (1 : ℂ)) ∧
    ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (2 : ℝ) * 0 ≤ 0 ∧ (∀ j : ℕ, 1 ≤ j → j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun _ : ℝ × ℝ => (1 : ℂ)) (0, 0)‖ ≤ 0) ∧
    (3 : ℕ) ≤ 3 := by
  refine ⟨contDiff_const, by norm_num, le_rfl, le_rfl, by norm_num, ?_, le_rfl⟩
  intro j hj hj3
  rw [iteratedFDeriv_const_of_ne (by omega : j ≠ 0)]
  simp only [Pi.zero_apply, norm_zero, le_refl]

end Transformer.Modes
