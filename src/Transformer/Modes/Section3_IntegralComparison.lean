import Transformer.Modes.Section3_FrequencyMoments
import Transformer.Modes.Section3_GaussianComparison

/-!
# The integrated Gaussian error of the actual normalized sum

For a fixed standardized law with finite third moment and an integrable
characteristic norm power, small-frequency Gaussian comparison and the
proved exterior tail estimates give an `L¹` Fourier error of order
`1 / sqrt n`. This is the zero-derivative part of the argument in
arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:small-z`, `eq:big-z-exp`, and `eq:big-phi-poly`.

The boundary of the small region need not be discarded: its complement
is contained in the closed exterior region, so monotonicity of integrals
of nonnegative functions joins the two bounds. The constants depend on
the fixed law. Derivatives of the Fourier error, needed for the paper's
weight `1 + |x|²`, are not asserted by the zero-derivative result.
The reported error constant is nonnegative; its dependence on the law
and on the fixed small-frequency radius is retained explicitly.
Both eventual integrability and the quantitative integral bound apply
to the actual sum law, rather than an assumed approximate density.
-/

open Real MeasureTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The Gaussian Fourier error of the actual normalized sums is
integrable eventually, under the source's integrable-power condition.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4 Fourier inversion. -/
theorem eventually_integrable_characteristic_scaledSum_gaussian_error
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, Integrable (fun ξ : ℝ × ℝ =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
        ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ)) := by
  have hG : Integrable (fun ξ : ℝ × ℝ => Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2)) := by
    convert integrable_norm_pow_gaussian_frequency 0 (b := 1 / 2) (by norm_num) using 1
    funext ξ
    simp only [pow_zero, one_mul]
    congr 1
    ring
  filter_upwards [eventually_integrable_characteristic_scaledSum μ hcf] with n hn
  exact hn.sub hG.ofReal

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2⟩

/-- The full integral of the absolute characteristic error is eventually
`O(1 / sqrt n)` for the actual normalized sum of any fixed standardized
law with finite third moment and an integrable characteristic norm power.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:higher-error-goal`, the zero-derivative case. -/
theorem eventually_integral_characteristic_scaledSum_sub_gaussian_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ,
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ)‖) ≤
        C * (Real.sqrt n)⁻¹ := by
  obtain ⟨a, ha, hsmall⟩ := exists_characteristic_scaledSum_sub_gaussian_bound μ hμ hmom
  obtain ⟨Ctail, htail⟩ := eventually_integral_characteristic_scaledSum_tail_rate μ hcf ha
  let g : ℝ × ℝ → ℝ := fun ξ => Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2)
  let W : ℝ × ℝ → ℝ := fun ξ => ‖ξ‖ ^ 3 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  let M : ℝ := 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1
  have hM : 0 ≤ M := by
    have h : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
    dsimp [M]; linarith
  have hIg : Integrable g := by
    convert integrable_norm_pow_gaussian_frequency 0 (b := 1 / 2) (by norm_num) using 1
    funext ξ
    simp only [pow_zero, one_mul]
    dsimp [g]
    congr 1
    ring
  have hIW : Integrable W := by
    convert integrable_norm_pow_gaussian_frequency 3 (b := 1 / 8) (by norm_num) using 1
    funext ξ
    dsimp [W]
    congr 2
    ring
  have hW0 : 0 ≤ ∫ ξ, W ξ := integral_nonneg fun ξ => by dsimp [W]; positivity
  have hg10 : 0 ≤ ∫ ξ, ‖ξ‖ * g ξ := integral_nonneg fun ξ => by dsimp [g]; positivity
  refine ⟨M * (∫ ξ, W ξ) + max Ctail 0 + a⁻¹ * (∫ ξ, ‖ξ‖ * g ξ), by positivity, ?_⟩
  filter_upwards [eventually_integrable_characteristic_scaledSum μ hcf, htail,
    eventually_ge_atTop 2] with n hISum htailn hn
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast (show 0 < n by omega))
  let f : ℝ × ℝ → ℝ := fun ξ =>
    ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ - (g ξ : ℂ)‖
  have hIf : Integrable f := (hISum.sub hIg.ofReal).norm
  let S : Set (ℝ × ℝ) := {ξ | ‖ξ‖ ≤ a * Real.sqrt n}
  let E : Set (ℝ × ℝ) := {ξ | a * Real.sqrt n ≤ ‖ξ‖}
  have hS : MeasurableSet S := (isClosed_le continuous_norm continuous_const).measurableSet
  have hE : MeasurableSet E := (isClosed_le continuous_const continuous_norm).measurableSet
  have hsub : Sᶜ ⊆ E := by
    intro ξ hξ
    change ¬‖ξ‖ ≤ a * Real.sqrt n at hξ
    change a * Real.sqrt n ≤ ‖ξ‖
    exact le_of_lt (lt_of_not_ge hξ)
  have hlow : (∫ ξ in S, f ξ) ≤ M * (Real.sqrt n)⁻¹ * ∫ ξ, W ξ := by
    calc
      _ ≤ ∫ ξ in S, (M * (Real.sqrt n)⁻¹) * W ξ := by
        apply setIntegral_mono_on hIf.integrableOn (hIW.const_mul _).integrableOn hS
        intro ξ hξ
        simpa only [f, g, M, W, mul_assoc] using hsmall n hn ξ hξ
      _ ≤ ∫ ξ, (M * (Real.sqrt n)⁻¹) * W ξ :=
        setIntegral_le_integral (hIW.const_mul _) (ae_of_all _ fun ξ => by dsimp [W]; positivity)
      _ = _ := integral_const_mul _ _
  have hhigh : (∫ ξ in Sᶜ, f ξ) ≤
      (Ctail + a⁻¹ * (∫ ξ, ‖ξ‖ * g ξ)) * (Real.sqrt n)⁻¹ := by
    calc
      _ ≤ ∫ ξ in E, f ξ := setIntegral_mono_set hIf.integrableOn
        (ae_of_all _ fun ξ => norm_nonneg _) (ae_of_all _ fun ξ hξ => hsub hξ)
      _ ≤ ∫ ξ in E,
          ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ + g ξ := by
        apply setIntegral_mono_on hIf.integrableOn (hISum.norm.add hIg).integrableOn hE
        intro ξ _
        dsimp [f]
        have hgNorm : ‖(g ξ : ℂ)‖ = g ξ := by
          rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
        exact (norm_sub_le _ _).trans_eq (by rw [hgNorm])
      _ = (∫ ξ in E, ‖characteristic2
            ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) + ∫ ξ in E, g ξ :=
        integral_add hISum.norm.integrableOn hIg.integrableOn
      _ ≤ Ctail * (Real.sqrt n)⁻¹ + (a * Real.sqrt n)⁻¹ * ∫ ξ, ‖ξ‖ * g ξ := by
        apply add_le_add htailn
        have hgEq : (fun ξ : ℝ × ℝ => Real.exp (-(1 / 2) * (ξ.1 ^ 2 + ξ.2 ^ 2))) = g := by
          funext ξ
          dsimp [g]
          congr 1
          ring
        have ht := integral_gaussian_frequency_tail_le (b := 1 / 2) (by norm_num) (mul_pos ha hs)
        simpa only [← hgEq, E] using ht
      _ = _ := by rw [mul_inv_rev]; ring
  change (∫ ξ, f ξ) ≤ _
  rw [← integral_add_compl hS hIf]
  calc
    _ ≤ M * (Real.sqrt n)⁻¹ * (∫ ξ, W ξ) +
        (Ctail + a⁻¹ * (∫ ξ, ‖ξ‖ * g ξ)) * (Real.sqrt n)⁻¹ := add_le_add hlow hhigh
    _ = (M * (∫ ξ, W ξ) + Ctail + a⁻¹ * (∫ ξ, ‖ξ‖ * g ξ)) * (Real.sqrt n)⁻¹ := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_right
      (by linarith [le_max_left Ctail 0]) (by positivity)

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2,
    ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
