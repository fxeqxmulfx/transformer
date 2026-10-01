/-
# Quantitative mean-square convergence of actual dyadic Euler states

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Summing geometric L2 increments gives an explicit error bound against
the constructed limit, not merely qualitative convergence.
-/

import Transformer.BatchSize.Section4_EulerLimitLaw

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- A geometric successive mean-square estimate gives the explicit
error K 2^(-m)/(1-sqrt(1/2))^2 against any actual mean-square limit,
Section 4.3, Theorem 1. -/
theorem meanSquare_geometric_rate {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (P : Measure Ω)
    (X : ℕ → Ω → E) (hX : ∀ m, MemLp (X m) 2 P) (Y : Ω → E) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun m => ∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0))
    (K : ℝ) (hK : 0 ≤ K)
    (hbound : ∀ m, (∫ ω, ‖X m ω - X (m + 1) ω‖ ^ 2 ∂P) ≤ K * (1 / 2 : ℝ) ^ m) (m : ℕ) :
    (∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂P) ≤
      K / (1 - Real.sqrt (1 / 2)) ^ 2 * (1 / 2 : ℝ) ^ m := by
  let Z (n : ℕ) : Lp E 2 P := (hX n).toLp (X n)
  let r : ℝ := Real.sqrt (1 / 2)
  have hr : 0 ≤ r := Real.sqrt_nonneg _
  have hr1 : r < 1 := by
    simpa only [Real.sqrt_one] using Real.sqrt_lt_sqrt (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num : (1 / 2 : ℝ) < 1)
  have hrsq : r ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have hdist (n : ℕ) : dist (Z n) (Z (n + 1)) ≤ Real.sqrt K * r ^ n := by
    have h := hbound n
    rw [← toLp_dist_meanSquare P (X n) (X (n + 1)) (hX n) (hX (n + 1))] at h
    have heq : (Real.sqrt K * r ^ n) ^ 2 = K * (1 / 2 : ℝ) ^ n := by
      rw [mul_pow, Real.sq_sqrt hK, ← pow_mul, Nat.mul_comm n 2, pow_mul, hrsq]
    exact (sq_le_sq₀ dist_nonneg (by positivity)).mp (h.trans_eq heq.symm)
  have hLp := meanSquare_tendsto_toLp P X hX Y hY hlim
  have hd := dist_le_of_le_geometric_of_tendsto r (Real.sqrt K) hr1 hdist hLp m
  rw [← toLp_dist_meanSquare P (X m) Y (hX m) hY]
  calc
    _ ≤ (Real.sqrt K * r ^ m / (1 - r)) ^ 2 := pow_le_pow_left₀ dist_nonneg hd 2
    _ = K / (1 - r) ^ 2 * (1 / 2 : ℝ) ^ m := by
      rw [div_pow, mul_pow, Real.sq_sqrt hK, ← pow_mul, Nat.mul_comm m 2, pow_mul, hrsq]
      ring

/-- Joint nonvacuity of quantitative mean-square limit hypotheses,
Section 4.3: a nonzero constant state with zero successive error. -/
example : (∀ n : ℕ, MemLp ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) n) 2 (brownianNoiseLaw 1)) ∧
    MemLp (fun _ : BrownianSample 1 => (1 : ℝ)) 2 (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1, ‖(fun _ : BrownianSample 1 => (1 : ℝ)) ω - 1‖ ^ 2
      ∂brownianNoiseLaw 1) atTop (𝓝 0) ∧ (0 : ℝ) ≤ 0 ∧
    (∀ n : ℕ, (∫ ω : BrownianSample 1, ‖(fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) n ω - 1‖ ^ 2
      ∂brownianNoiseLaw 1) ≤ 0 * (1 / 2 : ℝ) ^ n) := by
  refine ⟨fun _ => memLp_const _, memLp_const _, by simp, le_rfl, ?_⟩
  intro n
  simp

/-- The actual optimizer dyadic approximations converge to the
constructed adapted limit with mean-square error O(2^(-m)),
Section 4.3 (2)--(3). -/
theorem optimizerEulerLimit_meanSquare_rate {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (t : ℝ≥0) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ m : ℕ,
      (∫ ω, ‖dyadicEuler (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) x₀ t m ω -
        optimizerEulerLimit method η B f σ hη hB hmodel x₀ t ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
          C * (1 / 2 : ℝ) ^ m := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  obtain ⟨M, hbM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨A, haA⟩ := diffusionNoiseScale_uniform_bound method B f σ hB hmodel
  let Aη : NNReal := ‖Real.sqrt η‖₊ * A
  have hAη : ∀ x k, |diffusionNoiseScale method η B f σ x k| ≤ Aη := by
    simpa only [Aη, NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _)] using haA η hη
  let X := dyadicEuler (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) x₀ t
  have hX (m : ℕ) : MemLp (X m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp _ _ Kb (‖Real.sqrt η‖₊ * Ka) hb (ha η hη) _ (dyadicGrid_monotone t m) x₀ _
  obtain ⟨K, hK, hbound⟩ := dyadicEuler_refinement_bound _ _ Kb (‖Real.sqrt η‖₊ * Ka) M Aη
    hb (ha η hη) hbM hAη x₀ t
  refine ⟨K / (1 - Real.sqrt (1 / 2)) ^ 2, by positivity, ?_⟩
  intro m
  exact meanSquare_geometric_rate (brownianNoiseLaw d) X hX _
    (optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀ t)
    (optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀ t) K hK
    (fun n => by simpa only [norm_sub_rev] using hbound n) m

/-- Joint nonvacuity of optimizer convergence-rate hypotheses,
Section 4.3: a positive batch and rate with the flat regular model. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
