import Transformer.Modes.Section3_WeightedDensity
/-!
# The cited Edgeworth estimates for actual normalized sums

Section 3 `thm:br` of arXiv:2412.09080v3 cites Bhattacharya–Rao,
Theorems 19.2–19.3: an identity-covariance law with finite moment of
order `s+1` has density error bounded by `n^{-(s-1)/2}` with spatial
weight `1 + |x|^s`, under suitable conditions.

Here the suitable condition is the original integrable-characteristic-
power hypothesis `HasIntegrableCharFun`. For `s = 2` the bound follows
from the proved error integrals of the actual characteristic function
and its first two derivatives, then literal Fourier inversion. The
constants depend on the fixed law; neither a continuous density nor
a derivative estimate is added as a hypothesis of `edgeworth_two`.
Actual bounded continuous densities also exist for all sufficiently
large sample counts, by the separately proved inversion result.

The `s = 3` statement retains its original fourth moment and exponential
moment hypotheses. Its correction `psiOf` uses the mixed derivatives of
the literal logarithmic moment generating function, as in `eq:psi`.
The Gaussian examples satisfy every hypothesis of both original
statements. Their names, quantifiers, spatial weights, and rate powers
are unchanged from `Section3_BR`; the estimates follow the Fourier
modules so that the import graph contains no cycle.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every actual continuous normalized-sum density has quadratically
weighted Gaussian error `O(1 / sqrt n)` under the original source conditions.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:higher-error-goal`, combining derivative orders zero and two. -/
theorem eventually_weighted_abs_scaledSum_density_sub_phi2
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x| ≤ C * (Real.sqrt n)⁻¹ := by
  obtain ⟨C0, hC0, herror0⟩ := eventually_integral_characteristic_scaledSum_sub_gaussian_rate μ hμ hmom hcf
  obtain ⟨_, _, herror1⟩ := eventually_integral_norm_fderiv_characteristic_scaledSum_sub_gaussian μ hμ hmom hcf
  obtain ⟨C2, hC2, herror2⟩ := eventually_integral_norm_iteratedFDeriv_two_characteristic_scaledSum_sub_gaussian μ hμ hmom hcf
  refine ⟨((2 * Real.pi) ^ 2)⁻¹ * (C0 + ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 * C2), by positivity, ?_⟩
  filter_upwards [herror0, herror1, herror2, eventually_integrable_characteristic_scaledSum μ hcf] with n hn0 hn1 hn2 hIn
  intro q hq hqd x
  have h := continuous_density_sub_weighted_abs_le _ stdGauss2 hIn integrable_characteristic2_stdGauss2
    (contDiff_characteristic_scaledSum μ hμ.memLp n)
    (contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp)))
    hn1.1.norm hn2.1.norm hq (show Continuous phi2 by unfold phi2; fun_prop)
    hqd.nonneg isDensityOf_stdGauss2.nonneg hqd.map_eq stdGauss2_eq_withDensity_phi2 x
  have hG : characteristic2 stdGauss2 = fun ξ : ℝ × ℝ =>
      ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := funext characteristic2_stdGauss2
  have hn0' : (∫ ξ, ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
      characteristic2 stdGauss2 ξ‖) ≤ C0 * (Real.sqrt n)⁻¹ := by simpa only [hG] using hn0
  calc
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        ((∫ ξ, ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ - characteristic2 stdGauss2 ξ‖) +
          ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
            ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ -
              iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) := h
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        (C0 * (Real.sqrt n)⁻¹ + ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 * (C2 * (Real.sqrt n)⁻¹)) := by
      gcongr
      exact hn2.2
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- The quadratically weighted rate is attained by actual continuous densities
for every sufficiently large normalized sum. Source: arXiv:2412.09080v3,
§3 `thm:br`, `s = 2`; existence is proved, not assumed in this conclusion. -/
theorem eventually_exists_scaledSum_density_weighted_gaussian_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∃ q : ℝ × ℝ → ℝ,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
      ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x| ≤ C * (Real.sqrt n)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_weighted_abs_scaledSum_density_sub_phi2 μ hμ hmom hcf
  refine ⟨C, hC, ?_⟩
  filter_upwards [herror, eventually_exists_continuous_density_scaledSum μ hcf] with n hn hex
  obtain ⟨q, hq, hqd, _⟩ := hex
  exact ⟨q, hq, hqd, hn q hq hqd⟩

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- The actual densities converge uniformly to the Gaussian with the full
quadratic spatial weight. Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`,
the convergence consequence of the proved weighted rate. -/
theorem eventually_weighted_abs_scaledSum_density_sub_phi2_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x| ≤ ε := by
  obtain ⟨C, _, herror⟩ := eventually_weighted_abs_scaledSum_density_sub_phi2 μ hμ hmom hcf
  have hlim : Tendsto (fun n : ℕ => C * (Real.sqrt (n : ℝ))⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp (Real.tendsto_sqrt_atTop.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop))).const_mul C
  filter_upwards [herror, hlim.eventually (gt_mem_nhds hε)] with n hn hεn
  intro q hq hqd x
  exact (hn q hq hqd x).trans hεn.le

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- `thm:br` for `s = 2`: `sup_x (1 + ‖x‖²) |q_n(x) - φ(x)| ≲ n^{-1/2}`.
arXiv:2412.09080v3, §3, `thm:br` (Bhattacharya–Rao, Theorems 19.2–19.3). -/
theorem edgeworth_two (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C, ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ, Continuous q →
      IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x| ≤ C * (n : ℝ) ^ (-(1 : ℝ) / 2) := by
  obtain ⟨C, _, herror⟩ := eventually_weighted_abs_scaledSum_density_sub_phi2 μ hμ hmom hcf
  refine ⟨C, ?_⟩
  filter_upwards [herror] with n hn
  intro q hq hqd x
  have he : (n : ℝ) ^ (-(1 : ℝ) / 2) = (Real.sqrt n)⁻¹ := by
    rw [Real.sqrt_eq_rpow]
    have hhalf : -(1 : ℝ) / 2 = -(1 / 2 : ℝ) := by ring
    rw [hhalf, Real.rpow_neg (Nat.cast_nonneg n)]
  rw [he]
  exact hn q hq hqd x

/-- `thm:br` for `s = 3`:
`sup_x (1 + ‖x‖³) |q_n(x) - φ(x) - n^{-1/2} ψ(x)| ≲ n^{-1}`.
arXiv:2412.09080v3, §3, `thm:br` (Bhattacharya–Rao, Theorems 19.2–19.3), with
`ψ` of `eq:psi`; exponential moments are assumed so that the cumulants in `ψ`
are those the paper defines (see the module docstring). -/
theorem edgeworth_three (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ)
    (hcf : HasIntegrableCharFun μ) :
    ∃ C, ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ, Continuous q →
      IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x|
        ≤ C * (n : ℝ)⁻¹ := by
  sorry

/-- The hypotheses of `edgeworth_two` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- The hypotheses of `edgeworth_three` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
