import Transformer.Modes.Section3_CharacteristicTaylor
/-!
# Gaussian damping for small frequencies

For a standardized law with finite third moment, the quadratic
characteristic-function remainder is `O(|ξ|³)`. At a sufficiently small
radius this leaves `|χ(ξ)| ≤ 1 - |ξ|²/4 ≤ exp(-|ξ|²/4)`. The radius is
explicitly chosen from the third moment and is positive for every such
law.

Using the exact identity `χ(Sₙ)(ξ) = χ(ξ/sqrt n)^n`, this proves Gaussian
damping for the actual normalized sum throughout `|ξ| ≤ a sqrt n`.
These are the damping estimates in arXiv:2412.09080v3, §5.4
`eq:br-9.10`; the error expansion and its derivatives require further
arguments. The exponent uses the Euclidean square, while the radius
condition uses the equivalent product sup norm of the Lean model.
-/

open Real MeasureTheory Filter
open scoped ENNReal
namespace Transformer.Modes

/-- The product sup-norm square is at most the Euclidean square.
Source: arXiv:2412.09080v3, §5.4, comparison with the Euclidean frequency
in `eq:br-9.10`. -/
theorem norm_sq_le_dot_self (ξ : ℝ × ℝ) : ‖ξ‖ ^ 2 ≤ ξ.1 ^ 2 + ξ.2 ^ 2 := by
  rw [Prod.norm_def, Real.norm_eq_abs, Real.norm_eq_abs]
  rcases le_total |ξ.1| |ξ.2| with h | h
  · rw [max_eq_right h, sq_abs]
    nlinarith [sq_nonneg ξ.1]
  · rw [max_eq_left h, sq_abs]
    nlinarith [sq_nonneg ξ.2]

/-- The Euclidean square is at most twice the product sup-norm square.
Source: arXiv:2412.09080v3, §5.4, norm comparison in `eq:br-9.10`. -/
theorem dot_self_le_two_norm_sq (ξ : ℝ × ℝ) : ξ.1 ^ 2 + ξ.2 ^ 2 ≤ 2 * ‖ξ‖ ^ 2 := by
  have hf := pow_le_pow_left₀ (norm_nonneg ξ.1) (norm_fst_le ξ) 2
  have hs := pow_le_pow_left₀ (norm_nonneg ξ.2) (norm_snd_le ξ) 2
  simp only [Real.norm_eq_abs, sq_abs] at hf hs
  linarith

/-- A standardized law with finite third moment has a Gaussian envelope
near the origin. Its radius depends only on the third moment. This is
the single-summand damping step for arXiv:2412.09080v3, §5.4
`eq:br-9.10`; no characteristic-function integrability is assumed. -/
theorem exists_characteristic2_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a : ℝ, 0 < a ∧ ∀ ξ : ℝ × ℝ, ‖ξ‖ ≤ a →
      ‖characteristic2 μ ξ‖ ≤ Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4) := by
  let M : ℝ := ∫ z, ‖z‖ ^ 3 ∂μ
  have hM : 0 ≤ M := integral_nonneg fun z => by positivity
  let C : ℝ := 4 * M + 1
  have hC : 0 < C := by dsimp [C]; linarith
  refine ⟨min 1 (1 / (4 * C)), by positivity, ?_⟩
  intro ξ hξ
  have hr := le_min_iff.mp hξ
  have hr1 : ‖ξ‖ ≤ 1 := hr.1
  have hrC : C * ‖ξ‖ ≤ 1 / 4 := by
    have h := (le_div_iff₀ (by positivity : 0 < 4 * C)).mp hr.2
    linarith
  have hQlo := norm_sq_le_dot_self ξ
  have hQhi := dot_self_le_two_norm_sq ξ
  have hr2 : ‖ξ‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg ξ]
  have hQ : 0 ≤ 1 - (ξ.1 ^ 2 + ξ.2 ^ 2) / 2 := by linarith
  have hcube : 4 * M * ‖ξ‖ ^ 3 ≤ ‖ξ‖ ^ 2 / 4 := by
    calc
      _ ≤ C * ‖ξ‖ ^ 3 := mul_le_mul_of_nonneg_right (by dsimp [C]; linarith) (by positivity)
      _ = (C * ‖ξ‖) * ‖ξ‖ ^ 2 := by ring
      _ ≤ (1 / 4) * ‖ξ‖ ^ 2 := mul_le_mul_of_nonneg_right hrC (sq_nonneg _)
      _ = _ := by ring
  have hn : ‖(1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 : ℂ)‖ =
      1 - (ξ.1 ^ 2 + ξ.2 ^ 2) / 2 := by
    have he : (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 : ℂ) =
        ((1 - (ξ.1 ^ 2 + ξ.2 ^ 2) / 2 : ℝ) : ℂ) := by push_cast; rfl
    rw [he, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hQ]
  have ht := norm_le_norm_add_norm_sub
    (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2) (characteristic2 μ ξ)
  have he := norm_characteristic2_sub_quadratic μ hμ hmom ξ
  rw [hn, norm_sub_rev (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2) (characteristic2 μ ξ)] at ht
  change ‖characteristic2 μ ξ - _‖ ≤ 4 * M * ‖ξ‖ ^ 3 at he
  calc
    _ ≤ 1 - (ξ.1 ^ 2 + ξ.2 ^ 2) / 4 := by linarith
    _ ≤ Real.exp (-((ξ.1 ^ 2 + ξ.2 ^ 2) / 4)) := Real.one_sub_le_exp_neg _
    _ = _ := congrArg Real.exp (by ring)

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)⟩

/-- The exact normalized-sum frequency factor converts a single-summand
Gaussian envelope into one valid for `|ξ| ≤ a sqrt n`. Source:
arXiv:2412.09080v3, §5.4 `eq:br-9.10`. The damping hypothesis is proved
from finite moments in `exists_characteristic2_gaussian_bound`. -/
theorem norm_characteristic_scaledSum_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {a : ℝ}
    (hb : ∀ ξ : ℝ × ℝ, ‖ξ‖ ≤ a →
      ‖characteristic2 μ ξ‖ ≤ Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4))
    {n : ℕ} (hn : 1 ≤ n) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ a * Real.sqrt n) :
    ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ ≤
      Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4) := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  have hscaled : ‖(Real.sqrt (n : ℝ))⁻¹ • ξ‖ ≤ a := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa [mul_comm] using hξ)
  rw [characteristic_scaledSum, norm_pow]
  calc
    _ ≤ Real.exp (-(((Real.sqrt n)⁻¹ • ξ).1 ^ 2 +
        ((Real.sqrt n)⁻¹ • ξ).2 ^ 2) / 4) ^ n :=
      pow_le_pow_left₀ (norm_nonneg _) (hb _ hscaled) n
    _ = _ := by
      rw [← Real.exp_nat_mul]
      apply congrArg Real.exp
      simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
      have hnS : (n : ℝ) * (Real.sqrt n)⁻¹ ^ 2 = 1 := by
        calc
          _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
          _ = _ := by field_simp
      calc
        _ = ((n : ℝ) * (Real.sqrt n)⁻¹ ^ 2) * (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4) := by ring
        _ = _ := by rw [hnS, one_mul]

example : (∀ ξ : ℝ × ℝ, ‖ξ‖ ≤ 1 →
      ‖characteristic2 stdGauss2 ξ‖ ≤ Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4)) ∧
    1 ≤ (1 : ℕ) ∧ ‖(0 : ℝ × ℝ)‖ ≤ 1 * Real.sqrt (1 : ℕ) := by
  refine ⟨?_, le_rfl, by simp⟩
  intro ξ
  have hb : ‖characteristic2 stdGauss2 ξ‖ ≤ Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4) := by
    unfold characteristic2
    rw [norm_charFun_stdGauss2, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    nlinarith [sq_nonneg ξ.1, sq_nonneg ξ.2]
  exact fun _ => hb

/-- The actual normalized sums of a standardized law have a common
small-frequency Gaussian envelope for every `n ≥ 1`, assuming only the
third moment. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4
`eq:br-9.10`; this proves its damping factor, before derivative estimates. -/
theorem exists_characteristic_scaledSum_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a : ℝ, 0 < a ∧ ∀ n : ℕ, 1 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n →
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ ≤
        Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 4) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ hmom
  refine ⟨a, ha, ?_⟩
  intro n hn ξ hξ
  exact norm_characteristic_scaledSum_gaussian_bound μ hb hn ξ hξ

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
