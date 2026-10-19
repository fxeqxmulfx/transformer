module

public import Transformer.Homogenized.Section7_GaussianRowDeviationFourth
public import Transformer.Homogenized.Section7_CorrelationTaylorExpectation

/-!
# The actual normalized Gaussian row deviation has a cubic-moment rate

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the third-order
correlation remainder. This module proves a uniform `d⁻³ᐟ²` bound
for the third absolute moment of `d⁻¹ ∑ zᵢ² - 1` under the actual
original standard Gaussian row law.

The centered unnormalized second moment is exactly `2d`, obtained
from the actual mean `d` and fourth row-norm moment `d(d+2)`.
The proved original Gaussian MGF gives a uniform fourth centered
moment bound `C d²`. Division by the actual dimension therefore
gives second moment `2/d` and fourth moment at most `C/d²`.
Hölder yields the third absolute moment with rate `d⁻³ᐟ²`.

All moments are proved integrable, all positive dimensions share one
constant, and no Gaussian concentration or Taylor remainder is assumed.
These are the single-factor row estimates needed to integrate the
correlation expansion. The original two-factor query still requires
its original joint law and mixed second-moment calculations.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Actual finite second and fourth moments make the third absolute
moment integrable. Source: arXiv:2604.01978v1, §7.3,
the third-order error in `lem:delta_method`. -/
theorem integrable_third_abs_of_second_fourth {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (hX : MemLp X 2 P)
    (hfourth : Integrable (fun ω => X ω ^ 4) P) :
    Integrable (fun ω => |X ω| ^ 3) P := by
  have hAbs : MemLp (fun ω => |X ω|) 2 P := by simpa only [Real.norm_eq_abs] using hX.norm
  have hSq : MemLp (fun ω => X ω ^ 2) 2 P := by
    apply (memLp_two_iff_integrable_sq (hX.aestronglyMeasurable.pow 2)).mpr
    simpa only [Pi.pow_apply, ← pow_mul] using hfourth
  convert hAbs.integrable_mul hSq using 1
  funext ω
  simp only [Pi.mul_apply]
  rw [← sq_abs (X ω)]
  ring

/-- The actual squared row norm lies in `L²`. Source:
arXiv:2604.01978v1, §7.3, Gaussian moments in `lem:delta_method`. -/
theorem gaussianRow_norm_sq_memLp_two (d : ℕ) :
    MemLp (fun z : Fin d → ℝ => ∑ i : Fin d, z i ^ 2) 2 (gaussianRowLaw d) := by
  have hm : Measurable (fun z : Fin d → ℝ => ∑ i : Fin d, z i ^ 2) := by fun_prop
  apply (memLp_two_iff_integrable_sq hm.aestronglyMeasurable).mpr
  exact integrable_gaussianRow_norm_fourth d

/-- The actually centered squared row norm lies in `L²`. Source:
arXiv:2604.01978v1, §7.3, the deviation in `lem:delta_method`. -/
theorem gaussianRow_centered_norm_memLp_two (d : ℕ) :
    MemLp (fun z : Fin d → ℝ => (∑ i : Fin d, z i ^ 2) - (d : ℝ)) 2 (gaussianRowLaw d) := by
  convert (gaussianRow_norm_sq_memLp_two d).sub (memLp_const (d : ℝ)) using 1

/-- The original squared row norm has exact centered second moment
`2d`. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_centered_norm_sq (d : ℕ) :
    (∫ z : Fin d → ℝ, ((∑ i : Fin d, z i ^ 2) - (d : ℝ)) ^ 2 ∂gaussianRowLaw d) = 2 * d := by
  have h := variance_eq_sub (gaussianRow_norm_sq_memLp_two d)
  rw [variance_eq_integral (gaussianRow_norm_sq_memLp_two d).aemeasurable,
    integral_gaussianRow_norm_sq] at h
  simp only [Pi.pow_apply] at h
  rw [integral_gaussianRow_norm_fourth] at h
  nlinarith

/-- Dimension-normalized actual row-norm deviations lie in `L²`.
Source: arXiv:2604.01978v1, §7.3, scaling in `lem:delta_method`. -/
theorem gaussianRow_normalized_norm_memLp_two (d : ℕ) (hd : 0 < d) :
    MemLp (fun z : Fin d → ℝ => (∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1) 2 (gaussianRowLaw d) := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  convert (gaussianRow_centered_norm_memLp_two d).const_mul (d : ℝ)⁻¹ using 1
  funext z
  field_simp

/-- The actual normalized row deviation has exact second moment
`2/d`. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_normalized_norm_sq (d : ℕ) (hd : 0 < d) :
    (∫ z : Fin d → ℝ, ((∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1) ^ 2 ∂gaussianRowLaw d) = 2 / (d : ℝ) := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have he (z : Fin d → ℝ) : (∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1 =
      ((∑ i : Fin d, z i ^ 2) - (d : ℝ)) / (d : ℝ) := by field_simp
  simp_rw [he, div_pow]
  rw [integral_div, integral_gaussianRow_centered_norm_sq]
  field_simp

/-- The actual normalized row deviation has a uniform fourth-moment
bound `C/d²`. Source: arXiv:2604.01978v1, §7.3,
the Gaussian deviation moments required by `lem:delta_method`. -/
theorem gaussianRow_normalized_norm_fourth_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d →
      Integrable (fun z : Fin d → ℝ => ((∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1) ^ 4) (gaussianRowLaw d) ∧
      (∫ z : Fin d → ℝ, ((∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1) ^ 4 ∂gaussianRowLaw d) ≤ C / (d : ℝ) ^ 2 := by
  obtain ⟨C, hC, h⟩ := gaussianRow_centered_norm_fourth_uniform
  refine ⟨C, hC, ?_⟩
  intro d hd
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have he (z : Fin d → ℝ) : (∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1 =
      ((∑ i : Fin d, z i ^ 2) - (d : ℝ)) / (d : ℝ) := by field_simp
  simp_rw [he, div_pow]
  refine ⟨(h d).1.div_const _, ?_⟩
  rw [integral_div]
  calc
    _ ≤ (C * (d : ℝ) ^ 2) / (d : ℝ) ^ 4 := div_le_div_of_nonneg_right (h d).2 (by positivity)
    _ = _ := by field_simp

/-- The square roots of the actual normalized second- and fourth-
moment rates combine to the source's `d⁻³ᐟ²` scale. Source:
arXiv:2604.01978v1, §7.3, `lem:delta_method`, third-order remainder. -/
theorem sqrt_second_fourth_dimension_rate (C d : ℝ) (hC : 0 ≤ C) (hd : 0 < d) :
    Real.sqrt (2 / d) * Real.sqrt (C / d ^ 2) = Real.sqrt (2 * C) * d ^ (-(3 : ℝ) / 2) := by
  have he : d ^ (-(3 : ℝ) / 2) = d⁻¹ * (Real.sqrt d)⁻¹ := by
    rw [show (-(3 : ℝ) / 2) = (-1 : ℝ) + -(1 / 2 : ℝ) by ring,
      Real.rpow_add hd, Real.rpow_neg_one, Real.rpow_neg hd.le, ← Real.sqrt_eq_rpow]
  rw [Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hC,
    Real.sqrt_sq hd.le, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), he]
  ring

/-- **Actual normalized third absolute moment with a uniform rate.**
The original standard Gaussian row law satisfies the `d⁻³ᐟ²`
moment estimate needed for the correlation remainder, with one
positive constant for every positive dimension.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem gaussianRow_normalized_norm_third_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d →
      Integrable (fun z : Fin d → ℝ => |(∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1| ^ 3) (gaussianRowLaw d) ∧
      (∫ z : Fin d → ℝ, |(∑ i : Fin d, z i ^ 2) / (d : ℝ) - 1| ^ 3 ∂gaussianRowLaw d) ≤
        C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussianRow_normalized_norm_fourth_uniform
  refine ⟨Real.sqrt (2 * C), Real.sqrt_pos.mpr (by positivity), ?_⟩
  intro d hd
  have hX := gaussianRow_normalized_norm_memLp_two d hd
  refine ⟨integrable_third_abs_of_second_fourth _ _ hX (h d hd).1, ?_⟩
  have hb := third_abs_moment_le_sqrt_second_fourth _ _ hX (h d hd).1
  rw [integral_gaussianRow_normalized_norm_sq d hd] at hb
  have hm := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (h d hd).2) (Real.sqrt_nonneg (2 / (d : ℝ)))
  exact (hb.trans hm).trans_eq (sqrt_second_fourth_dimension_rate C d hC.le (by exact_mod_cast hd))

/-- Positive dimension and positive moment coefficient occur jointly
with the actual Gaussian scalar second- and fourth-moment hypotheses. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    MemLp (id : ℝ → ℝ) 2 (gaussianReal 0 1) ∧
    Integrable (fun x : ℝ => x ^ 4) (gaussianReal 0 1) := by
  exact ⟨by decide, by norm_num, zero_le_one, memLp_id_gaussianReal 2, integrable_gaussianReal_pow 1 4⟩

end Transformer.Homogenized
