module

public import Transformer.Modes.Section3_ThirdPowerDerivative

/-!
# Operator bounds retaining the undifferentiated characteristic powers

The third-order product rule of arXiv:2412.09080v3, §5.4
`eq:br-9.10` and `eq:big-z-exp`, retains powers `n-3`, `n-2`, and
`n-1` in its three kinds of terms. We bound the full multilinear norm,
preserving each of these powers and all three actual derivative norms.

The three arrangements of one second and one first derivative have
the same norm bound, giving the explicit coefficient three. Natural
falling factorials are bounded by powers of `n`, independently of
frequency. If the base function has modulus at most one, every term
can then retain the common power `n-3`. That common factor is essential
for integrability and geometric decay on the exterior frequency region.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- All third-power derivative terms have explicit operator bounds and retain their actual base powers.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
derivative order three. The falling factorials are bounded by `n³`
and `n²`; no modulus bound on the base function is needed. -/
theorem norm_iteratedFDeriv_three_complex_power_le (f : ℝ × ℝ → ℂ) (n : ℕ)
    (hf : ContDiff ℝ 3 f) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3 (fun η => f η ^ n) ξ‖ ≤
      (n : ℝ) ^ 3 * ‖f ξ‖ ^ (n - 3) * ‖fderiv ℝ f ξ‖ ^ 3 +
      3 * (n : ℝ) ^ 2 * ‖f ξ‖ ^ (n - 2) * ‖fderiv ℝ f ξ‖ *
        ‖iteratedFDeriv ℝ 2 f ξ‖ +
      (n : ℝ) * ‖f ξ‖ ^ (n - 1) * ‖iteratedFDeriv ℝ 3 f ξ‖ := by
  let A := ‖fderiv ℝ f ξ‖
  let B := ‖iteratedFDeriv ℝ 2 f ξ‖
  let C := ‖iteratedFDeriv ℝ 3 f ξ‖
  let r := ‖f ξ‖
  have hcoef3 : (n : ℝ) * (n - 1 : ℕ) * (n - 2 : ℕ) ≤ (n : ℝ) ^ 3 := by
    calc
      _ ≤ (n : ℝ) * n * n := by
        gcongr <;> omega
      _ = _ := by ring
  have hcoef2 : (n : ℝ) * (n - 1 : ℕ) ≤ (n : ℝ) ^ 2 := by
    calc
      _ ≤ (n : ℝ) * n := by gcongr; exact_mod_cast (show n - 1 ≤ n by omega)
      _ = _ := by ring
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  let P := ∏ i, ‖v i‖
  have hDv (w : ℝ × ℝ) : ‖fderiv ℝ f ξ w‖ ≤ A * ‖w‖ :=
    (fderiv ℝ f ξ).le_opNorm w
  have hD2v (a b : ℝ × ℝ) : ‖iteratedFDeriv ℝ 2 f ξ ![a, b]‖ ≤ B * (‖a‖ * ‖b‖) := by
    simpa [Fin.prod_univ_two] using (iteratedFDeriv ℝ 2 f ξ).le_opNorm ![a, b]
  have hD3v : ‖iteratedFDeriv ℝ 3 f ξ v‖ ≤ C * P :=
    (iteratedFDeriv ℝ 3 f ξ).le_opNorm v
  have hprod : ‖fderiv ℝ f ξ (v 0)‖ * ‖fderiv ℝ f ξ (v 1)‖ *
      ‖fderiv ℝ f ξ (v 2)‖ ≤ A ^ 3 * P := by
    calc
      _ ≤ (A * ‖v 0‖) * (A * ‖v 1‖) * (A * ‖v 2‖) := by
        gcongr <;> exact hDv _
      _ = _ := by dsimp [P]; rw [Fin.prod_univ_three]; ring
  have hinner : ‖iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2) +
      iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1) +
      iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0)‖ ≤ 3 * A * B * P := by
    calc
      _ ≤ ‖iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2) +
          iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1)‖ +
          ‖iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0)‖ := norm_add_le _ _
      _ ≤ ‖iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2)‖ +
          ‖iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1)‖ +
          ‖iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0)‖ :=
        add_le_add (norm_add_le _ _) le_rfl
      _ ≤ (B * (‖v 0‖ * ‖v 1‖)) * (A * ‖v 2‖) +
          (B * (‖v 0‖ * ‖v 2‖)) * (A * ‖v 1‖) +
          (B * (‖v 1‖ * ‖v 2‖)) * (A * ‖v 0‖) := by
        simp only [norm_mul]
        gcongr <;> first | exact hDv _ | exact hD2v _ _
      _ = _ := by dsimp [P]; rw [Fin.prod_univ_three]; ring
  rw [iteratedFDeriv_three_complex_power_apply f n hf]
  calc
    _ ≤ ‖(n : ℂ) * (n - 1 : ℕ) * (n - 2 : ℕ) * f ξ ^ (n - 3) *
        fderiv ℝ f ξ (v 0) * fderiv ℝ f ξ (v 1) * fderiv ℝ f ξ (v 2)‖ +
      ‖(n : ℂ) * (n - 1 : ℕ) * f ξ ^ (n - 2) *
        (iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2) +
          iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1) +
          iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0))‖ +
      ‖(n : ℂ) * f ξ ^ (n - 1) * iteratedFDeriv ℝ 3 f ξ v‖ := by
        exact (norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)
    _ = ((n : ℝ) * (n - 1 : ℕ) * (n - 2 : ℕ)) * r ^ (n - 3) *
          (‖fderiv ℝ f ξ (v 0)‖ * ‖fderiv ℝ f ξ (v 1)‖ * ‖fderiv ℝ f ξ (v 2)‖) +
        ((n : ℝ) * (n - 1 : ℕ)) * r ^ (n - 2) *
          ‖iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2) +
            iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1) +
            iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0)‖ +
        (n : ℝ) * r ^ (n - 1) * ‖iteratedFDeriv ℝ 3 f ξ v‖ := by
      simp only [norm_mul, norm_pow, Complex.norm_natCast]
      dsimp [r]
      ring
    _ ≤ (n : ℝ) ^ 3 * r ^ (n - 3) * (A ^ 3 * P) +
        (n : ℝ) ^ 2 * r ^ (n - 2) * (3 * A * B * P) +
        (n : ℝ) * r ^ (n - 1) * (C * P) := by
      gcongr
    _ = _ := by dsimp [A, B, C, r, P]; ring

example : ContDiff ℝ 3 (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))

/-- A base of modulus at most one leaves the common power `n-3` after all third derivative terms.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, the exterior
majorant before retaining an integrable characteristic power. -/
theorem norm_iteratedFDeriv_three_complex_power_le_of_norm_le_one
    (f : ℝ × ℝ → ℂ) (n : ℕ) (hf : ContDiff ℝ 3 f)
    (ξ : ℝ × ℝ) (hξ : ‖f ξ‖ ≤ 1) :
    ‖iteratedFDeriv ℝ 3 (fun η => f η ^ n) ξ‖ ≤
      ((n : ℝ) ^ 3 * ‖fderiv ℝ f ξ‖ ^ 3 +
        3 * (n : ℝ) ^ 2 * ‖fderiv ℝ f ξ‖ * ‖iteratedFDeriv ℝ 2 f ξ‖ +
        (n : ℝ) * ‖iteratedFDeriv ℝ 3 f ξ‖) * ‖f ξ‖ ^ (n - 3) := by
  have h2 : ‖f ξ‖ ^ (n - 2) ≤ ‖f ξ‖ ^ (n - 3) :=
    pow_le_pow_of_le_one (norm_nonneg _) hξ (by omega)
  have h1 : ‖f ξ‖ ^ (n - 1) ≤ ‖f ξ‖ ^ (n - 3) :=
    pow_le_pow_of_le_one (norm_nonneg _) hξ (by omega)
  calc
    _ ≤ (n : ℝ) ^ 3 * ‖f ξ‖ ^ (n - 3) * ‖fderiv ℝ f ξ‖ ^ 3 +
        3 * (n : ℝ) ^ 2 * ‖f ξ‖ ^ (n - 2) * ‖fderiv ℝ f ξ‖ *
          ‖iteratedFDeriv ℝ 2 f ξ‖ +
        (n : ℝ) * ‖f ξ‖ ^ (n - 1) * ‖iteratedFDeriv ℝ 3 f ξ‖ :=
      norm_iteratedFDeriv_three_complex_power_le f n hf ξ
    _ ≤ (n : ℝ) ^ 3 * ‖f ξ‖ ^ (n - 3) * ‖fderiv ℝ f ξ‖ ^ 3 +
        3 * (n : ℝ) ^ 2 * ‖f ξ‖ ^ (n - 3) * ‖fderiv ℝ f ξ‖ *
          ‖iteratedFDeriv ℝ 2 f ξ‖ +
        (n : ℝ) * ‖f ξ‖ ^ (n - 3) * ‖iteratedFDeriv ℝ 3 f ξ‖ := by gcongr
    _ = _ := by ring

example : ContDiff ℝ 3 (characteristic2 stdGauss2) ∧
    ‖characteristic2 stdGauss2 (0, 0)‖ ≤ 1 :=
  ⟨contDiff_characteristic2 stdGauss2 (k := 3)
    (IsGaussian.memLp_id _ _ (by simp)), norm_characteristic2_le_one stdGauss2 _⟩

/-- Real frequency scaling has its exact absolute-power factor in multilinear operator norm.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, the
normalized-sum derivative scale before the exterior integral. -/
theorem norm_iteratedFDeriv_complex_comp_smul (f : ℝ × ℝ → ℂ) (c : ℝ) (k : ℕ)
    (hf : ContDiff ℝ k f) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (fun η => f (c • η)) ξ‖ =
      |c| ^ k * ‖iteratedFDeriv ℝ k f (c • ξ)‖ := by
  have he : iteratedFDeriv ℝ k (fun η => f (c • η)) ξ =
      (c ^ k : ℝ) • iteratedFDeriv ℝ k f (c • ξ) := by
    ext v
    rw [iteratedFDeriv_complex_comp_smul f c k hf]
    simp only [smul_apply, Complex.real_smul, Complex.ofReal_pow]
  rw [he, norm_smul, Real.norm_eq_abs, abs_pow]

example (k : ℕ) : ContDiff ℝ k (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := k) (IsGaussian.memLp_id _ _ (by simp))

end Transformer.Modes
