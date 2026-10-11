import Transformer.Modes.Section3_MixedPowerDerivativeBounds

/-!
# Derivative bounds for the exact cubic power factors

The cancellation in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
leaves products of a small one-summand remainder and finite geometric
factors. The Leibniz sum through order three is at most eight times
the product of common derivative bounds. Applying it twice treats
the product of the two cubic factors without omitting any term.

For mixed powers, their retained modulus exponents combine to at
least the total degree minus six. This applies even when either
power is zero, one, or two. Finally the exact factorization gives
a bound for the derivative of the corrected error itself. Its
input bounds remain explicit hypotheses on actual derivatives;
they will be supplied by the local moment and geometric estimates.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- All actual product derivatives through order three obey a common Leibniz bound.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
product terms in the exact cubic-error factorization. -/
theorem norm_iteratedFDeriv_mul_le_uniform_three (f g : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ 3 f) (hg : ContDiff ℝ 3 g) (ξ : ℝ × ℝ)
    {A B : ℝ} (hA : 0 ≤ A) (hB : 0 ≤ B)
    (hDf : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j f ξ‖ ≤ A)
    (hDg : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j g ξ‖ ≤ B)
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => f η * g η) ξ‖ ≤ 8 * A * B := by
  have hs : (∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ)) ≤ 8 := by
    interval_cases k <;> norm_num [Finset.sum_range_succ, Nat.choose]
  calc
    _ ≤ ∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ) *
        ‖iteratedFDeriv ℝ i f ξ‖ * ‖iteratedFDeriv ℝ (k - i) g ξ‖ :=
      norm_iteratedFDeriv_mul_le hf hg ξ (by exact_mod_cast hk)
    _ ≤ ∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ) * A * B := by
      apply Finset.sum_le_sum
      intro i hi
      have hi3 : i ≤ 3 := by have := Finset.mem_range.mp hi; omega
      gcongr
      · exact hDf i hi3
      · exact hDg (k - i) (by omega)
    _ = (∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ)) * (A * B) := by
      simp only [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ ≤ 8 * (A * B) := mul_le_mul_of_nonneg_right hs (mul_nonneg hA hB)
    _ = _ := by ring

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (0 : ℂ)) ∧ (0 : ℝ) ≤ 0 ∧
    (∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j (fun _ : ℝ × ℝ => (0 : ℂ)) (0, 0)‖ ≤ 0) := by
  refine ⟨contDiff_const, le_rfl, fun j hj => ?_⟩
  simp only [iteratedFDeriv_fun_zero, Pi.zero_apply, norm_zero, le_refl]

/-- A mixed-power derivative retains the total degree minus six in the common modulus factor.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
polynomial geometric factors after the cubic cancellation. -/
theorem norm_iteratedFDeriv_mixed_power_le (f g : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ 3 f) (hg : ContDiff ℝ 3 g) (ξ : ℝ × ℝ)
    {r K : ℝ} (hr0 : 0 ≤ r) (hr : r ≤ 1) (hK : 0 ≤ K) (m l : ℕ)
    (hDf : ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => f η ^ m) ξ‖ ≤ K * r ^ (m - 3))
    (hDg : ∀ j : ℕ, j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun η => g η ^ l) ξ‖ ≤ K * r ^ (l - 3))
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => f η ^ m * g η ^ l) ξ‖ ≤
      8 * K ^ 2 * r ^ (m + l - 6) := by
  calc
    _ ≤ 8 * (K * r ^ (m - 3)) * (K * r ^ (l - 3)) :=
      norm_iteratedFDeriv_mul_le_uniform_three _ _ (hf.pow m) (hg.pow l) ξ
        (by positivity) (by positivity) hDf hDg k hk
    _ = 8 * K ^ 2 * r ^ ((m - 3) + (l - 3)) := by rw [pow_add]; ring
    _ ≤ _ := mul_le_mul_of_nonneg_left
      (pow_le_pow_of_le_one hr0 hr (by omega)) (by positivity)

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (0 : ℂ)) ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j
      (fun _ : ℝ × ℝ => (0 : ℂ) ^ 4) (0, 0)‖ ≤ 0 * (1 : ℝ) ^ (4 - 3 : ℕ)) := by
  refine ⟨contDiff_const, zero_le_one, le_rfl, le_rfl, fun j hj => ?_⟩
  simp only [zero_pow (by omega : 4 ≠ 0), iteratedFDeriv_fun_zero,
    Pi.zero_apply, norm_zero, zero_mul, le_refl]

/-- Actual derivative bounds for both geometric factors control the corrected power error.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, including
every derivative of the fourth-order and the two third-order factors. -/
theorem norm_iteratedFDeriv_complex_cubic_error_le_of_geometric_bounds
    (f g b : ℝ × ℝ → ℂ) (hf : ContDiff ℝ 3 f)
    (hg : ContDiff ℝ 3 g) (hb : ContDiff ℝ 3 b) (ξ : ℝ × ℝ) (n : ℕ)
    {A B D S T : ℝ} (hA : 0 ≤ A) (hB : 0 ≤ B) (hD : 0 ≤ D)
    (hS : 0 ≤ S) (hT : 0 ≤ T)
    (hR : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j (fun η => f η - g η - b η) ξ‖ ≤ A)
    (hDb : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j b ξ‖ ≤ B)
    (hDd : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j (fun η => f η - g η) ξ‖ ≤ D)
    (hDS : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j
      (fun η => ∑ i ∈ Finset.range n, f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤ S)
    (hDT : ∀ j : ℕ, j ≤ 3 → ‖iteratedFDeriv ℝ j
      (fun η => ∑ i ∈ Finset.range n, ∑ a ∈ Finset.range i,
        f η ^ a * g η ^ (n - 2 - a)) ξ‖ ≤ T)
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => f η ^ n - g η ^ n -
      (n : ℂ) * g η ^ (n - 1) * b η) ξ‖ ≤ 8 * A * S + 64 * B * D * T := by
  let s := fun η => ∑ i ∈ Finset.range n, f η ^ i * g η ^ (n - 1 - i)
  let t := fun η => ∑ i ∈ Finset.range n, ∑ a ∈ Finset.range i,
    f η ^ a * g η ^ (n - 2 - a)
  have hs : ContDiff ℝ 3 s := contDiff_complex_geometric_sum f g hf hg n
  have ht : ContDiff ℝ 3 t := contDiff_complex_second_geometric_sum f g hf hg n
  have hbd (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => b η * (f η - g η)) ξ‖ ≤ 8 * B * D :=
    norm_iteratedFDeriv_mul_le_uniform_three _ _ hb (hf.sub hg) ξ hB hD hDb hDd j hj
  have he : (fun η => f η ^ n - g η ^ n - (n : ℂ) * g η ^ (n - 1) * b η) =
      fun η => (f η - g η - b η) * s η + (b η * (f η - g η)) * t η := by
    funext η
    exact complex_power_sub_cubic_factorization (f η) (g η) (b η) n
  rw [he, fun_iteratedFDeriv_add
    (((hf.sub hg).sub hb).mul hs |>.of_le (by exact_mod_cast hk))
    ((hb.mul (hf.sub hg)).mul ht |>.of_le (by exact_mod_cast hk))]
  calc
    _ ≤ ‖iteratedFDeriv ℝ k (fun η => (f η - g η - b η) * s η) ξ‖ +
        ‖iteratedFDeriv ℝ k (fun η => (b η * (f η - g η)) * t η) ξ‖ := norm_add_le _ _
    _ ≤ 8 * A * S + 8 * (8 * B * D) * T := add_le_add
      (norm_iteratedFDeriv_mul_le_uniform_three _ _ ((hf.sub hg).sub hb) hs ξ
        hA hS hR hDS k hk)
      (norm_iteratedFDeriv_mul_le_uniform_three _ _ (hb.mul (hf.sub hg)) ht ξ
        (by positivity) hT hbd hDT k hk)
    _ = _ := by ring

/-- Equal constant characteristic values satisfy every factor hypothesis at sample count two. -/
example : ‖iteratedFDeriv ℝ 3 (fun _ : ℝ × ℝ =>
    (1 : ℂ) ^ 2 - 1 ^ 2 - (2 : ℂ) * 1 ^ (2 - 1 : ℕ) * 0) (0, 0)‖ ≤
    8 * (0 : ℝ) * 2 + 64 * 0 * 0 * 1 := by
  apply norm_iteratedFDeriv_complex_cubic_error_le_of_geometric_bounds
    (fun _ => (1 : ℂ)) (fun _ => (1 : ℂ)) (fun _ => (0 : ℂ))
    contDiff_const contDiff_const contDiff_const (0, 0) 2
    (A := 0) (B := 0) (D := 0) (S := 2) (T := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ?_ ?_ ?_ ?_ ?_ 3 le_rfl
  all_goals
    intro j hj
    by_cases hj0 : j = 0
    · subst j
      norm_num [norm_iteratedFDeriv_zero, Finset.sum_range_succ]
    · rw [iteratedFDeriv_const_of_ne hj0]
      simp only [Pi.zero_apply, norm_zero]
      norm_num

end Transformer.Modes
