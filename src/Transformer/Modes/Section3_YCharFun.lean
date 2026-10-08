import Transformer.Modes.Section3_ChangeOfVar
import Transformer.Modes.Section5_PowerIntegrability

/-!
# The characteristic function condition for the standardized summand

The Bhattacharya–Rao statements cited in §3 of arXiv:2412.09080v3 require
some positive power of the characteristic function to be integrable.
The source applies them to the law of `Y(t)` from `eq:Yi`. This module
proves that condition for the actual `lawY β t` for every `β > 0`.

`Y(t)` is the affine whitening of `(G(t), G'(t))`. On the frequency
side, the whitening acts by the transpose of its explicit triangular
matrix. Its determinant equals the whitening determinant and is
nonzero when the covariance is positive definite. Pushing Lebesgue
measure through this linear frequency map multiplies it by a finite
constant, so it preserves integrability of the composed norm power.

Centering contributes a complex exponential of a real multiple of
`I`, whose norm is one. The remaining positive-sign characteristic
function is the actual `fourierNu` evaluated at the negative transformed
frequency. The exact norm identity retains the means, covariance
entries, and the bandwidth and translation in the original law.

`Section5_PowerIntegrability` provides an integrable Fourier power for
`n > 4(β + 1)`. Covariance positivity is already proved for every `β > 0`.
Together these results establish `HasIntegrableCharFun (lawY β t)`
without a density assumption or an unproved input. The final continuity
statement also follows directly from the unit-modulus integrand.
Source: arXiv:2412.09080v3, §2.3 `eq:Yi`, §3 `thm:br`, and §5.4–§5.5.
-/

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The transpose whitening matrix acting on frequencies, so that its
dot product with `z` is the original frequency's dot product with
`whiten a b d z`. Source: arXiv:2412.09080v3, §2.3, `eq:Yi`. -/
noncomputable def whitenFrequency (a b d : ℝ) : (ℝ × ℝ) →ₗ[ℝ] ℝ × ℝ :=
  Matrix.toLin (Module.Basis.finTwoProd ℝ) (Module.Basis.finTwoProd ℝ)
    !![1 / √a, -b / √(a * (a * d - b ^ 2)); 0, a / √(a * (a * d - b ^ 2))]

/-- The explicit frequency transformation for the triangular whitening.
Source: arXiv:2412.09080v3, §2.3, `eq:Yi`, with the whitening convention
of `Section3_Hermite`. -/
theorem whitenFrequency_apply (a b d : ℝ) (ξ : ℝ × ℝ) :
    whitenFrequency a b d ξ =
      (ξ.1 / √a - b * ξ.2 / √(a * (a * d - b ^ 2)),
        a * ξ.2 / √(a * (a * d - b ^ 2))) := by
  rw [whitenFrequency, Matrix.toLin_finTwoProd_apply]
  ext <;> simp only <;> ring

/-- Transposing the whitening matrix preserves its determinant.
Source: arXiv:2412.09080v3, §2.3, `eq:Yi`. -/
theorem det_whitenFrequency (a b d : ℝ) :
    LinearMap.det (whitenFrequency a b d) = LinearMap.det (whitenLin a b d) := by
  rw [whitenFrequency, whitenLin, LinearMap.det_toLin, LinearMap.det_toLin,
    Matrix.det_fin_two_of, Matrix.det_fin_two_of]
  ring

/-- A positive definite covariance gives an invertible frequency map;
its finite Lebesgue scaling preserves integrability of a continuous
function. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4. -/
theorem integrable_comp_whitenFrequency {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2)
    {F : ℝ × ℝ → ℝ} (hcont : Continuous F) (hF : Integrable F) :
    Integrable (fun ξ => F (whitenFrequency a b d ξ)) := by
  have hdet : LinearMap.det (whitenFrequency a b d) ≠ 0 := by
    rw [det_whitenFrequency, det_whitenLin ha hD]
    exact inv_ne_zero (Real.sqrt_pos.mpr hD).ne'
  have hL : Measurable (whitenFrequency a b d) :=
    (LinearMap.continuous_of_finiteDimensional _).measurable
  have hmap := Measure.map_linearMap_addHaar_eq_smul_addHaar volume hdet
  have hint : Integrable F (volume.map (whitenFrequency a b d)) := by
    rw [hmap]
    exact hF.smul_measure ENNReal.ofReal_ne_top
  exact (integrable_map_measure hcont.aestronglyMeasurable hL.aemeasurable).mp hint

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 ∧
    Continuous (fun _ : ℝ × ℝ => (0 : ℝ)) ∧ Integrable (fun _ : ℝ × ℝ => (0 : ℝ)) := by
  exact ⟨by norm_num, by norm_num, continuous_const, by simp⟩

/-- The exact characteristic norm of the actual standardized law:
centering has unit modulus, and whitening transforms the frequency.
Source: arXiv:2412.09080v3, §2.3 `eq:Yi` and §5.4. -/
theorem norm_characteristic_lawY (β t : ℝ) (ξ : ℝ × ℝ) :
    ‖∫ z : ℝ × ℝ, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) ∂lawY β t‖ =
      ‖fourierNu β t (-(whitenFrequency (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t) ξ))‖ := by
  let κ := whitenFrequency (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t) ξ
  let c : ℂ := Complex.exp (-((κ.1 * meanG β t + κ.2 * meanG' β t : ℝ) : ℂ) * Complex.I)
  have hc : Continuous fun z : ℝ × ℝ =>
      Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) := by fun_prop
  have he (x : ℝ) :
      Complex.exp (((ξ.1 * (singleY β t x).1 + ξ.2 * (singleY β t x).2 : ℝ) : ℂ) * Complex.I) =
        c * Complex.exp (((κ.1 * bigG β t x + κ.2 * bigG' β t x : ℝ) : ℂ) * Complex.I) := by
    dsimp [c]
    rw [← Complex.exp_add]
    apply congrArg Complex.exp
    dsimp [κ]
    rw [whitenFrequency_apply]
    simp only [singleY, whiten, Complex.ofReal_add, Complex.ofReal_sub,
      Complex.ofReal_mul, Complex.ofReal_div]
    ring
  unfold lawY
  rw [integral_map (measurable_singleY β t).aemeasurable hc.aestronglyMeasurable,
    integral_congr_ae (ae_of_all _ he), integral_const_mul, norm_mul]
  have hn : ‖c‖ = 1 := by dsimp [c]; simp [Complex.norm_exp]
  rw [hn, one_mul]
  have hsign : (∫ x, Complex.exp (((κ.1 * bigG β t x + κ.2 * bigG' β t x : ℝ) : ℂ) * Complex.I)
      ∂gaussianReal 0 1) = fourierNu β t (-κ) := by
    unfold fourierNu
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      apply congrArg Complex.exp
      simp only [Prod.fst_neg, Prod.snd_neg, Complex.ofReal_add, Complex.ofReal_mul, Complex.ofReal_neg]
      ring
  rw [hsign]

/-- The genuine characteristic function of `Y(t)` has integrable `n`th
norm power for `n > 4(β + 1)`, retaining the corrected bandwidth-dependent
threshold. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5. -/
theorem integrable_characteristic_lawY_pow {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) :
    Integrable (fun ξ : ℝ × ℝ =>
      ‖∫ z, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) ∂lawY β t‖ ^ n) := by
  have hcont : Continuous fun ξ : ℝ × ℝ => ‖fourierNu β t (-ξ)‖ ^ n :=
    ((continuous_fourierNu β t).comp continuous_neg).norm.pow n
  have hint := (integrable_fourierNu_pow_all_bandwidths hβ t hn).comp_neg
  have h := integrable_comp_whitenFrequency (sigmaFst_pos hβ t) (sigmaDet_pos hβ t) hcont hint
  simpa only [norm_characteristic_lawY] using h

example : (0 : ℝ) < 3 ∧ 4 * (3 + 1) < ((17 : ℕ) : ℝ) := by norm_num

/-- Every actual standardized summand with positive bandwidth satisfies
the integrable characteristic power hypothesis of the density expansion.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5. -/
theorem hasIntegrableCharFun_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    HasIntegrableCharFun (lawY β t) := by
  obtain ⟨n, hn⟩ := exists_nat_gt (4 * (β + 1))
  have hnpos : 0 < n := by
    have : (0 : ℝ) < (n : ℝ) := by linarith
    exact_mod_cast this
  exact ⟨n, hnpos, integrable_characteristic_lawY_pow hβ t hn⟩

example : (0 : ℝ) < 3 := by norm_num

/-- Continuity of the characteristic function of the actual `lawY`,
including at zero frequency, by dominated convergence.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4. -/
theorem continuous_characteristic_lawY (β t : ℝ) :
    Continuous (fun ξ : ℝ × ℝ =>
      ∫ z, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) ∂lawY β t) := by
  apply continuous_of_dominated (bound := fun _ => (1 : ℝ))
  · intro ξ
    have hc : Continuous fun z : ℝ × ℝ =>
        Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) := by fun_prop
    exact hc.aestronglyMeasurable
  · exact fun ξ => ae_of_all _ fun z => by simp [Complex.norm_exp]
  · exact integrable_const _
  · exact ae_of_all _ fun z => by fun_prop

end Transformer.Modes
