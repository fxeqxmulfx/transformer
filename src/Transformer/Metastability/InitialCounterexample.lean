/-
# A zero-scale counterexample to the uncorrected Gaussian-mixture proposition

At zero variance, the formula of `eq: gaussian.mixture` defines the zero
measure, not a probability law. The paper explicitly assumes positive scale;
this module records the counterexample to the earlier unrestricted Lean claim.
-/

import Transformer.Metastability.Initial

open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- At zero scale the displayed density of `eq: gaussian.mixture` vanishes
everywhere, so its product measure has total mass zero when there is a sample.
The source assumes `σ > 0`; this exposes why that hypothesis is necessary in
any formal statement using the density formula.

Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem mixtureLaw_sigma_zero_univ (d n r : ℕ) (hd : 0 < d) (w : Idx r → EucSpace d) :
    (mixtureLaw d (n + 1) r 0 w) Set.univ = 0 := by
  have hzero : ∀ x : EucSpace d, gaussianMixtureDensity d r 0 w x = 0 := by
    intro x
    have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
    have hdHalf : (0 : ℝ) < (d : ℝ) / 2 := by positivity
    simp [gaussianMixtureDensity, Real.zero_rpow (ne_of_gt hdHalf)]
  simp [mixtureLaw, hzero, Measure.pi_univ]

/-- The present Lean statement of `prop: mixture.of.gaussians` is false without
the source's `σ > 0` assumption: at `σ = 0` its purported mixture law is the
zero measure, although the stated lower probability bound is positive.  This
counterexample uses `d = n = r = 1`, also outside the source's `d, n ≥ 2`.

Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians` and
`eq: gaussian.mixture`. -/
theorem not_mixture_separation_without_scale_pos :
    isCentered 1 1 100 (1 / 32) 1 (fun _ => Transformer.basePoint 0) ∧
      (6 * ((0 : ℝ) / Real.sqrt 1) * Real.sqrt 1)
            / (1 + (0 / Real.sqrt 1) * Real.sqrt 1)
          + (0 / Real.sqrt 1) * Real.sqrt (2 * 1 * Real.log 1) ≤ 1 / 32 ∧
      ¬ 1 - 2 * Real.exp (-(1 : ℝ)) ≤
        (mixtureLaw 1 1 1 0 fun _ =>
          ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)).real
            (projectedSeparated 1 1 100 (1 / 32)) := by
  obtain ⟨hcent, hδ⟩ := centered_zero_scale_witness
  refine ⟨hcent, hδ, ?_⟩
  have hmass : (mixtureLaw 1 1 1 0 fun _ =>
      ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1))
      (projectedSeparated 1 1 100 (1 / 32)) = 0 := by
    exact le_antisymm
      ((measure_mono (Set.subset_univ _)).trans_eq (mixtureLaw_sigma_zero_univ 1 0 1 one_pos _))
      (by positivity)
  have hexp : (2 : ℝ) < Real.exp 1 := by
    have h := Real.add_one_lt_exp (x := (1 : ℝ)) (by norm_num)
    linarith
  have hpos : 0 < (1 : ℝ) - 2 * Real.exp (-1) := by
    have heq : Real.exp (-1 : ℝ) * Real.exp 1 = 1 := by
      rw [← Real.exp_add]
      norm_num
    have he0 : 0 < Real.exp (-1 : ℝ) := Real.exp_pos _
    nlinarith [mul_pos (sub_pos.mpr hexp) he0]
  intro h
  rw [Measure.real, hmass] at h
  norm_num at h
  linarith


end Metastability
end Transformer
