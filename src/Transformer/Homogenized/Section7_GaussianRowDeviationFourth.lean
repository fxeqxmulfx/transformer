module

public import Transformer.Homogenized.Section7_GaussianRowFourth
public import Transformer.Homogenized.Section6_DiagonalQuadraticMGF

/-!
# Uniform fourth moments of the original Gaussian row-norm deviation

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the third-order
correlation remainder, using the proved exponential estimates of
§6.3, `lem:Hanson_wright`. This module bounds the actual centered
row squared norm `∑ zᵢ² - d` under its original independent Gaussian law.

The original diagonal quadratic MGF applies with every coefficient
equal to one. Choosing a coefficient proportional to `1/sqrt(d)`
keeps both signed exponential moments bounded independently of the
dimension. The elementary fourth-power term in the exponential series
then bounds the actual fourth centered moment by `C d²`.

The universal constant is explicit in the proof and positive. All
finite dimensions, including zero, are covered. The scalar exponential
comparison proves fourth-moment integrability before comparing its
integral. No fourth moment or Gaussian remainder is postulated.
Combined with the exact second moment and Hölder, this provides the
required third-moment scaling for one original Gaussian factor.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Two actual exponential moments control a scaled fourth moment.
Source: arXiv:2604.01978v1, §6.3, exponential moment estimates
used for the §7.3 remainder in `lem:delta_method`. -/
theorem scaled_fourth_moment_le_exponential_pair {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (t : ℝ) (ht : t ≠ 0)
    (hpos : Integrable (fun ω => Real.exp (t * X ω)) P)
    (hneg : Integrable (fun ω => Real.exp (-t * X ω)) P) :
    Integrable (fun ω => X ω ^ 4) P ∧
      t ^ 4 * (∫ ω, X ω ^ 4 ∂P) ≤
        24 * ((∫ ω, Real.exp (t * X ω) ∂P) + ∫ ω, Real.exp (-t * X ω) ∂P) := by
  have hi := integrable_pow_of_integrable_exp_mul ht hpos hneg 4
  have hb (x : ℝ) : t ^ 4 * x ^ 4 ≤ 24 * (Real.exp (t * x) + Real.exp (-t * x)) := by
    have h := Real.pow_div_factorial_le_exp |t * x| (abs_nonneg (t * x)) 4
    have he : |t * x| ^ 4 = t ^ 4 * x ^ 4 := by
      rw [show (4 : ℕ) = 2 * 2 by decide, pow_mul, sq_abs]
      ring
    norm_num only [Nat.factorial, Nat.cast_ofNat] at h
    rw [he] at h
    have hh : Real.exp |t * x| ≤ Real.exp (t * x) + Real.exp (-t * x) := by
      simpa only [neg_mul] using Real.exp_abs_le (t * x)
    nlinarith
  have h := integral_mono (hi.const_mul (t ^ 4)) ((hpos.add hneg).const_mul 24) (fun ω => hb (X ω))
  rw [integral_const_mul, integral_const_mul] at h
  simp only [Pi.add_apply] at h
  rw [integral_add hpos hneg] at h
  exact ⟨hi, h⟩

/-- The actual row-norm deviation has a quadratic MGF with its
true dimension factor. Source: arXiv:2604.01978v1, §6.3,
`lem:Hanson_wright`, used in §7.3, `lem:delta_method`. -/
theorem gaussianRow_centered_norm_exp (d : ℕ) (t : ℝ)
    (ht : |t| ≤ 1 / (32 * Real.exp 1)) :
    Integrable (fun z : Fin d → ℝ => Real.exp (t * ((∑ i : Fin d, z i ^ 2) - (d : ℝ))))
      (gaussianRowLaw d) ∧
    (∫ z : Fin d → ℝ, Real.exp (t * ((∑ i : Fin d, z i ^ 2) - (d : ℝ)))
      ∂gaussianRowLaw d) ≤ Real.exp (2048 * (Real.exp 1) ^ 4 * t ^ 2 * (d : ℝ)) := by
  have hind : iIndepFun (fun i : Fin d => fun z : Fin d → ℝ => z i) (gaussianRowLaw d) :=
    iIndepFun_pi (X := fun _ : Fin d => id) (fun _ => aemeasurable_id)
  have hZ : ∀ i : Fin d, HasSubgaussianMGF (fun z : Fin d → ℝ => z i) 1 (gaussianRowLaw d) :=
    hasSubgaussianMGF_standardProduct_coordinate
  have h := subgaussian_centered_diagonal_exp_integral_le (gaussianRowLaw d)
    (WithLp.toLp 2) 1 hind hZ (fun _ => 1) t (fun i => by simpa using ht)
  have hm (i : Fin d) : (∫ z : Fin d → ℝ, z i ^ 2 ∂gaussianRowLaw d) = 1 :=
    integral_sq_of_map_eq_gaussianReal (measurable_pi_apply i) (gaussianRowLaw_map_coordinate i)
  have he (z : Fin d → ℝ) : (∑ i : Fin d, (z i ^ 2 - 1)) = (∑ i : Fin d, z i ^ 2) - (d : ℝ) := by
    rw [Finset.sum_sub_distrib]
    simp
  simpa only [WithLp.ofLp_toLp, NNReal.coe_one, one_mul, one_pow, hm, he,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] using h

/-- **Actual uniform centered fourth-moment estimate.** The original
standard Gaussian row squared norm has fourth centered moment at
most `C d²`, with one positive constant for all finite dimensions.
Source: arXiv:2604.01978v1, §7.3, the deviation moments required
by `lem:delta_method`, derived from §6.3's actual diagonal MGF. -/
theorem gaussianRow_centered_norm_fourth_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ,
      Integrable (fun z : Fin d → ℝ => ((∑ i : Fin d, z i ^ 2) - (d : ℝ)) ^ 4) (gaussianRowLaw d) ∧
      (∫ z : Fin d → ℝ, ((∑ i : Fin d, z i ^ 2) - (d : ℝ)) ^ 4 ∂gaussianRowLaw d) ≤ C * (d : ℝ) ^ 2 := by
  let k : ℝ := 64 * (Real.exp 1) ^ 2
  have hk : 0 < k := by dsimp [k]; positivity
  refine ⟨48 * Real.exp 1 * k ^ 4, by positivity, ?_⟩
  intro d
  by_cases hd : d = 0
  · subst d; simp
  have hd1 : (1 : ℝ) ≤ d := by exact_mod_cast (Nat.one_le_iff_ne_zero.mpr hd)
  let s := Real.sqrt (d : ℝ)
  have hs0 : 0 < s := Real.sqrt_pos.mpr (by linarith)
  have hsq : s ^ 2 = (d : ℝ) := Real.sq_sqrt (Nat.cast_nonneg d)
  have hs1 : 1 ≤ s := by nlinarith
  let t : ℝ := 1 / (k * s)
  have htpos : 0 < t := by dsimp [t]; positivity
  have he1 : 1 ≤ Real.exp 1 := Real.one_le_exp (by norm_num)
  have he2 : Real.exp 1 ≤ (Real.exp 1) ^ 2 := by nlinarith [sq_nonneg (Real.exp 1 - 1)]
  have hden : 32 * Real.exp 1 ≤ k * s := by
    have h := mul_le_mul_of_nonneg_left hs1 hk.le
    dsimp [k] at h ⊢
    nlinarith
  have ht : |t| ≤ 1 / (32 * Real.exp 1) := by
    rw [abs_of_pos htpos]
    dsimp [t]
    apply (div_le_div_iff₀ (mul_pos hk hs0) (by positivity)).mpr
    simpa only [one_mul] using hden
  obtain ⟨hip, hbp⟩ := gaussianRow_centered_norm_exp d t ht
  obtain ⟨hin, hbn⟩ := gaussianRow_centered_norm_exp d (-t) (by simpa only [abs_neg] using ht)
  have hV : 2048 * (Real.exp 1) ^ 4 * t ^ 2 * (d : ℝ) = 1 / 2 := by
    rw [← hsq]
    dsimp [t, k]
    field_simp [ne_of_gt hs0]
    ring
  have hbp1 : (∫ z : Fin d → ℝ, Real.exp (t * ((∑ i : Fin d, z i ^ 2) - (d : ℝ)))
      ∂gaussianRowLaw d) ≤ Real.exp 1 := hbp.trans (Real.exp_le_exp.mpr (by rw [hV]; norm_num))
  have hbn1 : (∫ z : Fin d → ℝ, Real.exp (-t * ((∑ i : Fin d, z i ^ 2) - (d : ℝ)))
      ∂gaussianRowLaw d) ≤ Real.exp 1 := hbn.trans (Real.exp_le_exp.mpr (by rw [neg_sq, hV]; norm_num))
  obtain ⟨hi, hb⟩ := scaled_fourth_moment_le_exponential_pair (gaussianRowLaw d)
    (fun z : Fin d → ℝ => (∑ i : Fin d, z i ^ 2) - (d : ℝ)) t htpos.ne' hip hin
  refine ⟨hi, ?_⟩
  apply (mul_le_mul_iff_right₀ (pow_pos htpos 4)).mp
  calc
    _ ≤ 48 * Real.exp 1 := by linarith [hbp1, hbn1, hb]
    _ = t ^ 4 * (48 * Real.exp 1 * k ^ 4 * (d : ℝ) ^ 2) := by
      rw [← hsq]
      dsimp [t]
      field_simp [ne_of_gt hk, ne_of_gt hs0]

/-- An actual two-dimensional Gaussian row and a nonzero signed
coefficient satisfy the MGF interval and both exponential-integrability
hypotheses jointly. The same row is used throughout the moment argument. -/
example : ∃ t : ℝ, t ≠ 0 ∧ |t| ≤ 1 / (32 * Real.exp 1) ∧
    Integrable (fun z : Fin 2 → ℝ => Real.exp (t * ((∑ i : Fin 2, z i ^ 2) - 2))) (gaussianRowLaw 2) ∧
    Integrable (fun z : Fin 2 → ℝ => Real.exp (-t * ((∑ i : Fin 2, z i ^ 2) - 2))) (gaussianRowLaw 2) := by
  let t : ℝ := 1 / (64 * Real.exp 1)
  have htpos : 0 < t := by dsimp [t]; positivity
  have ht : |t| ≤ 1 / (32 * Real.exp 1) := by
    rw [abs_of_pos htpos]
    dsimp [t]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [Real.exp_pos 1]
  exact ⟨t, htpos.ne', ht, (gaussianRow_centered_norm_exp 2 t ht).1,
    (gaussianRow_centered_norm_exp 2 (-t) (by simpa only [abs_neg] using ht)).1⟩

end Transformer.Homogenized
