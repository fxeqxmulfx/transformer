/-
# Exact fourth moment of an adapted Brownian step

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The state and the bounded amplitude may depend on the same entire past.
Odd future-increment terms vanish by independence from that past.
-/

import Transformer.BatchSize.Section4_FourthMomentTools

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Exact fourth-moment recursion for a scalar adapted state plus
its next bounded-amplitude Brownian kick, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_fourth_energy {d : ℕ} (k : Fin d)
    (Z H : BrownianSample d → ℝ) (s t : ℝ≥0) (hst : s ≤ t)
    (hZ : StronglyMeasurable[brownianFiltration d s] Z)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hZL4 : MemLp Z 4 (brownianNoiseLaw d)) (A : ℝ≥0) (hbound : ∀ ω, |H ω| ≤ A) :
    (∫ ω, (Z ω + H ω * brownianIncrement k s t ω) ^ 4 ∂brownianNoiseLaw d) =
      (∫ ω, Z ω ^ 4 ∂brownianNoiseLaw d) +
        6 * ((t : ℝ) - s) * (∫ ω, Z ω ^ 2 * H ω ^ 2 ∂brownianNoiseLaw d) +
          3 * ((t : ℝ) - s) ^ 2 * (∫ ω, H ω ^ 4 ∂brownianNoiseLaw d) := by
  let C (j l : ℕ) : BrownianSample d → ℝ := fun ω => Z ω ^ j * H ω ^ l
  have hCm (j l : ℕ) : StronglyMeasurable[brownianFiltration d s] (C j l) :=
    ((continuous_pow j).comp_stronglyMeasurable hZ).mul
      ((continuous_pow l).comp_stronglyMeasurable hH)
  have hCi (j : ℕ) (hj : j ≤ 4) (l : ℕ) : Integrable (C j l) (brownianNoiseLaw d) := by
    have hHl : AEStronglyMeasurable (fun ω => H ω ^ l) (brownianNoiseLaw d) :=
      ((continuous_pow l).comp_stronglyMeasurable hH).mono ((brownianFiltration d).le s)
        |>.aestronglyMeasurable
    exact (memLp_four_integrable_pow (brownianNoiseLaw d) Z hZL4 j hj).mul_bdd hHl
      (Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hbound ω) l)
  have h1 := brownianIncrement_adapted_power k s t hst (C 3 1) (hCm 3 1) (hCi 3 (by norm_num) 1)
    1 (by norm_num)
  have h2 := brownianIncrement_adapted_power k s t hst (C 2 2) (hCm 2 2) (hCi 2 (by norm_num) 2)
    2 (by norm_num)
  have h3 := brownianIncrement_adapted_power k s t hst (C 1 3) (hCm 1 3) (hCi 1 (by norm_num) 3)
    3 (by norm_num)
  have h4 := brownianIncrement_adapted_power k s t hst (C 0 4) (hCm 0 4) (hCi 0 (by norm_num) 4)
    4 (by norm_num)
  have hZ4 := memLp_four_integrable_pow (brownianNoiseLaw d) Z hZL4 4 le_rfl
  have hexpand (ω : BrownianSample d) : (Z ω + H ω * brownianIncrement k s t ω) ^ 4 =
      Z ω ^ 4 + (4 * (C 3 1 ω * brownianIncrement k s t ω ^ 1) +
        (6 * (C 2 2 ω * brownianIncrement k s t ω ^ 2) +
          (4 * (C 1 3 ω * brownianIncrement k s t ω ^ 3) +
            C 0 4 ω * brownianIncrement k s t ω ^ 4))) := by
    dsimp [C]
    ring
  simp_rw [hexpand]
  rw [integral_add (f := fun ω => Z ω ^ 4)
    (g := fun ω => 4 * (C 3 1 ω * brownianIncrement k s t ω ^ 1) +
      (6 * (C 2 2 ω * brownianIncrement k s t ω ^ 2) +
        (4 * (C 1 3 ω * brownianIncrement k s t ω ^ 3) +
          C 0 4 ω * brownianIncrement k s t ω ^ 4)))
    hZ4 ((h1.1.const_mul 4).add ((h2.1.const_mul 6).add ((h3.1.const_mul 4).add h4.1)))]
  rw [integral_add (f := fun ω => 4 * (C 3 1 ω * brownianIncrement k s t ω ^ 1))
    (g := fun ω => 6 * (C 2 2 ω * brownianIncrement k s t ω ^ 2) +
      (4 * (C 1 3 ω * brownianIncrement k s t ω ^ 3) + C 0 4 ω * brownianIncrement k s t ω ^ 4))
    (h1.1.const_mul 4) ((h2.1.const_mul 6).add ((h3.1.const_mul 4).add h4.1))]
  rw [integral_add (f := fun ω => 6 * (C 2 2 ω * brownianIncrement k s t ω ^ 2))
    (g := fun ω => 4 * (C 1 3 ω * brownianIncrement k s t ω ^ 3) + C 0 4 ω * brownianIncrement k s t ω ^ 4)
    (h2.1.const_mul 6) ((h3.1.const_mul 4).add h4.1)]
  rw [integral_add (f := fun ω => 4 * (C 1 3 ω * brownianIncrement k s t ω ^ 3))
    (g := fun ω => C 0 4 ω * brownianIncrement k s t ω ^ 4) (h3.1.const_mul 4) h4.1]
  simp only [integral_const_mul]
  rw [h1.2, h2.2, h3.2, h4.2]
  simp only [pow_one, brownianIncrement_mean, brownianIncrement_secondMoment k s t hst,
    brownianIncrement_thirdMoment, brownianIncrement_fourthMoment k s t hst, mul_zero,
    C, pow_zero, one_mul]
  ring

/-- Joint nonvacuity of fourth-energy hypotheses, Section 4.3:
a nonconstant past Brownian state and a constant positive amplitude. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (coordinateBrownian (0 : Fin 1) 1) ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    MemLp (coordinateBrownian (0 : Fin 1) 1) 4 (brownianNoiseLaw 1) ∧
    (∀ ω : BrownianSample 1, |(fun _ : BrownianSample 1 => (1 : ℝ)) ω| ≤ (1 : ℝ≥0)) :=
  ⟨by norm_num, (coordinateBrownian_filtered (0 : Fin 1)).stronglyAdapted 1,
    stronglyMeasurable_const,
    (coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1 |>.memLp
      (by norm_num), by simp⟩

end Transformer.BatchSize
