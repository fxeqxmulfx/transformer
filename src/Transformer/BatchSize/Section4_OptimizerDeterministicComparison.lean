/-
# Finite-horizon comparison for the actual optimizer diffusion

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The expected deterministic backward tests telescope along the constructed
continuous process. Uniform C2 bounds and the proved local defects yield
an unconditional O(eta) estimate on a physical time horizon.
-/

import Transformer.BatchSize.Section4_OptimizerDeterministicLocalError
import Transformer.BatchSize.Section4_DeterministicHorizonBounds
import Transformer.BatchSize.Section4_DriftDerivativeBounds

open MeasureTheory
open scoped BigOperators NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual continuous optimizer diffusion and the deterministic
mean Euler trajectory differ weakly by O(eta) on a fixed horizon,
Section 4.3 (2)--(3), Theorem 1. All propagated derivative and
one-step consistency bounds are proved from the stated Gaussian model. -/
theorem optimizerEulerPath_deterministic_weak_error {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (T : ℝ) (hT : 0 ≤ T) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ (η : NNReal) (hη : 0 < (η : ℝ)) (φ : EucSpace d → ℝ),
      BoundedSmoothTest 2 φ → ∀ n : ℕ, (n : ℝ) * η ≤ T → ∀ x₀,
      |(∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω ((n : ℝ) * η))
        ∂brownianNoiseLaw d) - deterministicEulerTest (diffusionDrift method B f σ) η φ n x₀| ≤
        C * η := by
  let b := diffusionDrift method B f σ
  have hb : ContDiff ℝ 2 b :=
    (regularGaussianModel_smooth_coefficients method 0 B f σ hmodel).1.of_le (by norm_num)
  obtain ⟨K, hK⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨H, hH⟩ := diffusionDrift_fderiv_lipschitz method B f σ hmodel
  obtain ⟨C₀, hC₀, hlocal⟩ := optimizerEulerPath_deterministic_local_error method B f σ hB hmodel
  let R : NNReal := ⟨deterministicTestBound K H T,
    (deterministicTestBound_one_le K H T hT).trans' (by norm_num)⟩
  have hR : 0 < R := by
    change 0 < deterministicTestBound K H T
    linarith [deterministicTestBound_one_le K H T hT]
  refine ⟨C₀ * R * T, by positivity, ?_⟩
  intro η hη φ hφ n hn x₀
  let Z := optimizerEulerPath method η B f σ hη.le hB hmodel x₀
  let τ := fun j : ℕ => (j : NNReal) * η
  let Q := fun j : ℕ => ∫ ω, deterministicEulerTest b η φ (n - j) (Z ω (τ j)) ∂brownianNoiseLaw d
  have hτ (j : ℕ) : τ (j + 1) = τ j + η := by
    dsimp [τ]
    push_cast
    ring
  have hstep (j : ℕ) (hj : j ∈ Finset.range n) : |Q (j + 1) - Q j| ≤ C₀ * R * (η : ℝ) ^ 2 := by
    have hjn : j < n := Finset.mem_range.mp hj
    let m := n - (j + 1)
    have hm : (m : ℝ) * η ≤ T := (mul_le_mul_of_nonneg_right
      (by exact_mod_cast (Nat.sub_le n (j + 1))) η.coe_nonneg).trans hn
    obtain ⟨hs, hd⟩ := deterministicEulerTest_horizon_bounds b K H hb hK hH T hT η m hm φ hφ
    have h := hlocal η hη R hR (deterministicEulerTest b η φ m) hs hd x₀ (τ j)
    have hpred : n - j = m + 1 := by dsimp [m]; omega
    change |(∫ ω, deterministicEulerTest b η φ (n - (j + 1)) (Z ω (τ (j + 1))) ∂brownianNoiseLaw d) -
      (∫ ω, deterministicEulerTest b η φ (n - j) (Z ω (τ j)) ∂brownianNoiseLaw d)| ≤ _
    rw [hτ, hpred]
    exact h
  have hsum : |Q n - Q 0| ≤ (n : ℝ) * (C₀ * R * (η : ℝ) ^ 2) := by
    rw [← Finset.sum_range_sub Q n]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc
      _ ≤ ∑ _j ∈ Finset.range n, C₀ * R * (η : ℝ) ^ 2 := Finset.sum_le_sum hstep
      _ = _ := by simp
  have hinit : Q 0 = deterministicEulerTest b η φ n x₀ := by
    change (∫ ω, deterministicEulerTest b η φ (n - 0) (Z ω (τ 0)) ∂brownianNoiseLaw d) = _
    simp only [Nat.sub_zero, τ, Nat.cast_zero, zero_mul, NNReal.coe_zero]
    calc
      _ = ∫ _ω : BrownianSample d, deterministicEulerTest b η φ n x₀ ∂brownianNoiseLaw d := by
        apply integral_congr_ae
        filter_upwards [optimizerEulerPath_initial method η B f σ hη.le hB hmodel x₀] with ω hω
        rw [hω]
      _ = _ := by simp
  have hfinal : Q n = ∫ ω, φ (Z ω ((n : ℝ) * η)) ∂brownianNoiseLaw d := by
    simp only [Q, Nat.sub_self, deterministicEulerTest, τ, NNReal.coe_mul, NNReal.coe_natCast]
  rw [hinit, hfinal] at hsum
  refine hsum.trans ?_
  calc
    (n : ℝ) * (C₀ * R * (η : ℝ) ^ 2) = (C₀ * R * η) * ((n : ℝ) * η) := by ring
    _ ≤ (C₀ * R * η) * T := mul_le_mul_of_nonneg_left hn (by positivity)
    _ = _ := by ring

/-- Joint nonvacuity of the unconditional continuous comparison,
Section 4.3: positive batch and horizon, flat regular loss and unit noise. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 :=
  ⟨by norm_num, regularGaussianModel_flat 1, by norm_num⟩

end Transformer.BatchSize
