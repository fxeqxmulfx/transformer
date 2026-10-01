/-
# First-moment optimizer increments on one learning-rate interval

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The proved fourth moments retain the squared learning rate in their
noise term. On an interval of length at most eta, this yields an O(eta)
first-moment increment for the actual continuous optimizer path.
-/

import Transformer.BatchSize.Section4_EulerLimitLaw

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Rescaling a fourth-moment bound gives a first-moment estimate
without any assumption about samplewise bounded noise,
Section 4.3, Theorem 1's weak comparison. -/
theorem firstMoment_le_of_small_fourthMoment {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] (P : Measure Ω) [IsProbabilityMeasure P]
    (V : Ω → E) (hi : Integrable V P)
    (hi4 : Integrable (fun ω => ‖V ω‖ ^ 4) P) (η C : ℝ) (hη : 0 < η)
    (hfour : (∫ ω, ‖V ω‖ ^ 4 ∂P) ≤ C * η ^ 4) :
    (∫ ω, ‖V ω‖ ∂P) ≤ (1 + C) * η := by
  have hη3 : 0 < η ^ 3 := pow_pos hη _
  have hpoint (ω : Ω) : ‖V ω‖ ≤ η + ‖V ω‖ ^ 4 / η ^ 3 := by
    let r : ℝ := ‖V ω‖
    have hr : 0 ≤ r := norm_nonneg _
    have hm : r * η ^ 3 ≤ η ^ 4 + r ^ 4 := by
      by_cases h : r ≤ η
      · have hh := mul_le_mul_of_nonneg_right h hη3.le
        nlinarith [pow_nonneg hr 4]
      · have hh := mul_le_mul_of_nonneg_left
          (pow_le_pow_left₀ hη.le (le_of_not_ge h) 3) hr
        nlinarith [pow_nonneg hη.le 4]
    have hd := (le_div_iff₀ hη3).mpr hm
    have he : (η ^ 4 + r ^ 4) / η ^ 3 = η + r ^ 4 / η ^ 3 := by
      field_simp
    rw [he] at hd
    exact hd
  have hb := integral_mono hi.norm ((integrable_const η).add (hi4.div_const (η ^ 3))) hpoint
  change (∫ ω, ‖V ω‖ ∂P) ≤ ∫ ω, η + ‖V ω‖ ^ 4 / η ^ 3 ∂P at hb
  have he : (∫ ω, η + ‖V ω‖ ^ 4 / η ^ 3 ∂P) = η + (∫ ω, ‖V ω‖ ^ 4 ∂P) / η ^ 3 := by
    rw [integral_add (integrable_const η) (hi4.div_const (η ^ 3))]
    simp only [div_eq_mul_inv, integral_mul_const, integral_const, probReal_univ, one_smul]
  rw [he] at hb
  refine hb.trans ?_
  calc
    η + (∫ ω, ‖V ω‖ ^ 4 ∂P) / η ^ 3 ≤ η + (C * η ^ 4) / η ^ 3 :=
      add_le_add le_rfl (div_le_div_of_nonneg_right hfour hη3.le)
    _ = _ := by field_simp

/-- Joint nonvacuity of the first-moment estimate's hypotheses,
Section 4.3: unit update under a genuine probability measure. -/
example : Integrable (fun _ : Unit => (1 : ℝ)) (Measure.dirac ()) ∧
    Integrable (fun _ : Unit => ‖(1 : ℝ)‖ ^ 4) (Measure.dirac ()) ∧
    (0 : ℝ) < 1 ∧ (∫ _ : Unit, ‖(1 : ℝ)‖ ^ 4 ∂Measure.dirac ()) ≤ 1 * 1 ^ 4 :=
  ⟨integrable_const _, integrable_const _, by norm_num, by simp⟩

/-- The actual continuous optimizer path has a mesh-independent
O(eta) expected increment over every interval of length at most eta,
Section 4.3 (2)--(3), Theorem 1. All model constants precede eta,
the initial state and the interval in the quantifier order. -/
theorem optimizerEulerPath_small_increment_bound {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ (η : ℝ) (hη : 0 < η) (x₀ : EucSpace d) (s t : NNReal),
      s ≤ t → (t : ℝ) - s ≤ η →
      Integrable (fun ω => ‖optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω t -
        optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s‖) (brownianNoiseLaw d) ∧
      (∫ ω, ‖optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω t -
        optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s‖ ∂brownianNoiseLaw d) ≤ C * η := by
  obtain ⟨M, A, hmoment⟩ := optimizerEulerLimit_increment_fourthMoment method B f σ hB hmodel
  let D : ℝ := 8 * (M : ℝ) ^ 4 + 24 * (d : ℝ) ^ 2 * (A : ℝ) ^ 4
  refine ⟨1 + D, by dsimp [D]; positivity, ?_⟩
  intro η hη x₀ s t hst hlen
  let Y := fun ω => optimizerEulerLimit method η B f σ hη.le hB hmodel x₀ t ω -
    optimizerEulerLimit method η B f σ hη.le hB hmodel x₀ s ω
  let V := fun ω => optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω t -
    optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s
  have heq : V =ᵐ[brownianNoiseLaw d] Y := by
    filter_upwards [optimizerEulerPath_eval_ae_eq method η B f σ hη.le hB hmodel x₀ t,
      optimizerEulerPath_eval_ae_eq method η B f σ hη.le hB hmodel x₀ s] with ω ht hs
    exact congrArg₂ (· - ·) ht hs
  have hiY : Integrable Y (brownianNoiseLaw d) :=
    ((optimizerEulerLimit_memLp method η B f σ hη.le hB hmodel x₀ t).sub
      (optimizerEulerLimit_memLp method η B f σ hη.le hB hmodel x₀ s)).integrable (by norm_num)
  have hiV : Integrable V (brownianNoiseLaw d) := hiY.congr heq.symm
  obtain ⟨hiY4, hY4⟩ := hmoment η hη.le x₀ s t hst
  have heq4 : (fun ω => ‖V ω‖ ^ 4) =ᵐ[brownianNoiseLaw d] (fun ω => ‖Y ω‖ ^ 4) :=
    heq.fun_comp (fun v => ‖v‖ ^ 4)
  have hiV4 : Integrable (fun ω => ‖V ω‖ ^ 4) (brownianNoiseLaw d) := hiY4.congr heq4.symm
  have hδ : 0 ≤ (t : ℝ) - s := sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst)
  have hV4 : (∫ ω, ‖V ω‖ ^ 4 ∂brownianNoiseLaw d) ≤ D * η ^ 4 := by
    rw [integral_congr_ae heq4]
    refine hY4.trans ?_
    calc
      _ ≤ 8 * (M : ℝ) ^ 4 * η ^ 4 + 24 * (d : ℝ) ^ 2 * η ^ 2 * (A : ℝ) ^ 4 * η ^ 2 := by
        gcongr
      _ = _ := by dsimp [D]; ring
  exact ⟨hiV.norm, firstMoment_le_of_small_fourthMoment (brownianNoiseLaw d) V hiV hiV4 η D hη hV4⟩

/-- Joint nonvacuity of the actual optimizer small-increment
hypotheses, Section 4.3: unit noise and an interval of length eta. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : NNReal) ≤ 1 / 1000 ∧ ((1 / 1000 : NNReal) : ℝ) - 0 ≤ 1 / 1000 :=
  ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, by norm_num, by norm_num⟩

end Transformer.BatchSize
