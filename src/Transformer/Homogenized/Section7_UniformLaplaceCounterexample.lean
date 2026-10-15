/-
# Counterexample to the dimension-uniform Laplace estimate

ArXiv:2604.01978v1, §7.3, `lemma:Laplace_method`.
The source asserts an error `O(d^(-3/2) + β^(-1/2))` when replacing
the original softmax barycenter correlation by normalized-query
correlation. This simultaneous-dimension bound is false, including
the source's original Gaussian product initialization, standard value
variance `1/d`, common query/key entry variance `1/d`, and the exact
uniform token law with density and gradient constants `(1,1,0)`.

Given any prefactor and any dimension and temperature thresholds,
choose a large `s`, temperature `β=s²`, cutoff `R=s⁴`, and integer
dimension larger than `16 exp(2s⁴)`. The original barycenter's expected
squared norm is less than one quarter by the proved cutoff bound and
the actual Gaussian query second moment. Its original normalized-query
self-correlation is exactly one. The proposed error bound is less than
one quarter after increasing `s` and dimension, a contradiction.

The source's Step 1 writes dimension-dependent terms such as
`O(d ‖w‖²/λ_x)` and then drops that dependence from the Laplace error.
The counterexample retains all original measures and normalizations.
Both dimension and actual temperature can exceed arbitrary thresholds;
this is a refutation of the printed simultaneous-limit estimate.
At the chosen variance, the actual effective temperature is also `β`.
The value matrix has the source's adopted variance `1/d`, and both
independent Gaussian factors of the attention matrix are retained.
-/

import Transformer.Homogenized.Section7_GaussianScaling
import Transformer.Metastability.Section4_UniformLaw
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Real MeasureTheory

namespace Transformer.Homogenized

/-- Actual finite parameters make the Gaussian cutoff bound and the
proposed dimension-uniform rate simultaneously small.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem exists_dimension_laplace_cutoff_fail (C B : ℝ) (D : ℕ) (hC : 0 < C) (hB : 0 ≤ B) :
    ∃ (d : ℕ) (s : ℝ), 2 ≤ d ∧ D ≤ d ∧ 0 < s ∧ B ≤ s ^ 2 ∧
      Real.exp (2 * s ^ 4) / d + 1 / s ^ 4 < 1 / 4 ∧
      C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + (s ^ 2) ^ (-(1 : ℝ) / 2)) < 1 / 4 := by
  let s := 16 * (C + B + 1)
  have hs16 : (16 : ℝ) ≤ s := by dsimp [s]; linarith
  have hs : 0 < s := by linarith
  have hs1 : 1 ≤ s := by linarith
  have hsb : B + 1 ≤ s := by dsimp [s]; linarith
  have htemp : B ≤ s ^ 2 := by nlinarith [mul_nonneg hs.le (sub_nonneg.mpr hs1)]
  obtain ⟨d, hd⟩ := exists_nat_gt (max (max (2 : ℝ) D) (max (16 * Real.exp (2 * s ^ 4)) (16 * C)))
  have hd2 : (2 : ℝ) < d := (le_max_left _ _).trans_lt ((le_max_left _ _).trans_lt hd)
  have hdD : (D : ℝ) < d := (le_max_right _ _).trans_lt ((le_max_left _ _).trans_lt hd)
  have hde : 16 * Real.exp (2 * s ^ 4) < (d : ℝ) :=
    (le_max_left _ _).trans_lt ((le_max_right _ _).trans_lt hd)
  have hdc : 16 * C < (d : ℝ) :=
    (le_max_right _ _).trans_lt ((le_max_right _ _).trans_lt hd)
  have hdR : (0 : ℝ) < d := by linarith
  have hdN : 2 ≤ d := by exact_mod_cast hd2.le
  have he : Real.exp (2 * s ^ 4) / d < 1 / 16 := by
    apply (div_lt_iff₀ hdR).mpr
    linarith
  have hs4 : (16 : ℝ) ^ 4 ≤ s ^ 4 := pow_le_pow_left₀ (by norm_num) hs16 4
  have hi : 1 / s ^ 4 < (1 / 16 : ℝ) := by
    apply (div_lt_iff₀ (pow_pos hs 4)).mpr
    norm_num at hs4
    linarith
  have hds : C / (d : ℝ) < 1 / 16 := by
    apply (div_lt_iff₀ hdR).mpr
    linarith
  have hss : C / s < 1 / 16 := by
    apply (div_lt_iff₀ hs).mpr
    dsimp [s]
    linarith
  have hpow : (d : ℝ) ^ (-(3 : ℝ) / 2) ≤ (d : ℝ)⁻¹ := by
    simpa only [Real.rpow_neg_one] using Real.rpow_le_rpow_of_exponent_le
      (by linarith : (1 : ℝ) ≤ d) (by norm_num : -(3 : ℝ) / 2 ≤ -1)
  have hhalf : (s ^ 2) ^ (-(1 : ℝ) / 2) = s⁻¹ := by
    rw [neg_div, Real.rpow_neg (sq_nonneg s), ← Real.sqrt_eq_rpow, Real.sqrt_sq hs.le]
  refine ⟨d, s, hdN, by exact_mod_cast hdD.le, hs, htemp, by linarith, ?_⟩
  rw [hhalf]
  have hp := mul_le_mul_of_nonneg_left hpow hC.le
  have hid : C * (d : ℝ)⁻¹ = C / d := (div_eq_mul_inv _ _).symm
  have his : C * s⁻¹ = C / s := (div_eq_mul_inv _ _).symm
  rw [hid] at hp
  nlinarith

/-- The original Gaussian head and exact uniform token law violate
the source's proposed rate beyond arbitrary dimension and temperature
thresholds. Every original density, unit-query and initialization premise
is realized jointly. Source: arXiv:2604.01978v1, §7.3,
`lemma:Laplace_method`, at the common original entry variance `1/d`. -/
theorem uniform_laplace_self_error_counterexample (C B : ℝ) (D : ℕ)
    (hC : 0 < C) (hB : 0 ≤ B) :
    ∃ d : ℕ, 2 ≤ d ∧ D ≤ d ∧
      ∃ (β : ℝ) (ν : Measure (SSphere d)) (x : EucSpace d), 0 < β ∧ B ≤ β ∧ ‖x‖ = 1 ∧
        Metastability.IsUniformOn d ν ∧
        IsLowTemperature d (ν.map (Subtype.val : SSphere d → EucSpace d))
          (fun _ => ν.map Subtype.val) (fun _ _ => 1) 1 1 0 ∧
        IsGaussianHeadLaw d (stdSigmaV d) (stdSigmaV d)
          (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) ∧
        C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + β ^ (-(1 : ℝ) / 2)) <
          |baryCorr β (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d))
            (ν.map (Subtype.val : SSphere d → EucSpace d)) x x -
            ∫ θ : HeadParam d, inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ x))
              ∂gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)| := by
  obtain ⟨d, s, hd, hDd, hs, hBs, hcut, herr⟩ := exists_dimension_laplace_cutoff_fail C B D hC hB
  have hdpos : 0 < d := by omega
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hdpos.ne'
  let ν : Measure (SSphere d) := MeanField.uniformLaw d
  have hν : Metastability.IsUniformOn d ν := Metastability.isUniformOn_uniformLaw (by omega)
  let j : Fin d := ⟨0, hdpos⟩
  let x : EucSpace d := EuclideanSpace.single j 1
  refine ⟨d, hd, hDd, s ^ 2, ν, x, pow_pos hs 2, hBs, by simp [x], hν,
    isLowTemperature_map_spherical_uniform ν hν, isGaussianHeadLaw_gaussHeadLaw d _ _, ?_⟩
  have hb := baryCorr_uniform_gaussian_query_cutoff_bound hdpos ν hν
    (s ^ 2 / d) (s ^ 4) (pow_pos hs 4) j
  have hratio : (s ^ 2 / d) ^ 2 * (d : ℝ) ^ 2 / (s ^ 4) ^ 2 = 1 / s ^ 4 := by
    field_simp [hdR, hs.ne']
  rw [hratio] at hb
  dsimp only [x]
  rw [integral_scaled_gaussianHead_normalized_query_self (stdSigmaV d) (stdSigmaV_pos hdpos),
    baryCorr_std_gaussianHead hdpos]
  linarith [neg_le_abs (baryCorr (s ^ 2 / d) (gaussHeadLaw d 1 1)
    (ν.map (Subtype.val : SSphere d → EucSpace d)) (EuclideanSpace.single j 1)
      (EuclideanSpace.single j 1) - 1)]

/-- No dimension-uniform prefactor and eventual thresholds give the
printed Laplace rate, even on the exact uniform law and common Gaussian
scaling. Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem not_uniform_laplace_self_error_rate :
    ¬ ∃ (C B : ℝ) (D : ℕ), 0 < C ∧ 0 ≤ B ∧ ∀ d : ℕ, 2 ≤ d → D ≤ d →
      ∀ β : ℝ, 0 < β → B ≤ β → ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ∀ x : EucSpace d, ‖x‖ = 1 →
        |baryCorr β (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d))
          (ν.map (Subtype.val : SSphere d → EucSpace d)) x x -
          ∫ θ : HeadParam d, inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ x))
            ∂gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)| ≤
          C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + β ^ (-(1 : ℝ) / 2)) := by
  rintro ⟨C, B, D, hC, hB, h⟩
  obtain ⟨d, hd, hDd, β, ν, x, hβ, hBβ, hx, hν, -, -, hbad⟩ :=
    uniform_laplace_self_error_counterexample C B D hC hB
  exact not_lt_of_ge (h d hd hDd β hβ hBβ ν hν x hx) hbad

/-- The counterexample's scalar input hypotheses are jointly satisfied
at a nonzero prefactor and a positive temperature threshold. Dimension
thresholds are arbitrary natural numbers, including the source's `d ≥ 2`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨one_pos, zero_le_one⟩

end Transformer.Homogenized
