/-
# Continuous modifications of the actual optimizer Euler limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The proved fourth-moment estimates imply Kolmogorov's criterion after
an invertible time change. Returning to physical time gives continuous
sample paths whose evaluations equal the constructed adapted L2 limits.
-/

import Transformer.BatchSize.Section4_KolmogorovCriterion

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The actual optimizer Euler limit satisfies Kolmogorov's condition
on the auxiliary clock, Section 4.3 (2)--(3). The fourth-moment noise
constant retains its eta-squared factor. -/
theorem optimizerEulerLimit_timeChanged_isKolmogorov {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d) :
    ∃ C : ℝ≥0, IsKolmogorovProcess
      (fun t => optimizerEulerLimit method η B f σ hη hB hmodel x₀ (eulerTimeCompress t))
        (brownianNoiseLaw d) 4 2 C := by
  obtain ⟨M, A, hmoment⟩ := optimizerEulerLimit_increment_fourthMoment method B f σ hB hmodel
  let C : ℝ≥0 := ⟨8 * M ^ 4 + 24 * (d : ℝ) ^ 2 * η ^ 2 * A ^ 4, by positivity⟩
  refine ⟨C, isKolmogorovProcess_of_fourthMoment (brownianNoiseLaw d) _ ?_ C ?_⟩
  · intro t
    exact ((optimizerEulerLimit_adapted method η B f σ hη hB hmodel x₀ (eulerTimeCompress t)).mono
      ((brownianFiltration d).le (eulerTimeCompress t))).measurable
  · intro s v hsv
    obtain ⟨hint, hb⟩ := hmoment η hη x₀ (eulerTimeCompress s) (eulerTimeCompress v)
      (eulerTimeCompress_monotone hsv)
    obtain ⟨h2, h4⟩ := eulerTimeCompress_increment_bounds s v hsv
    refine ⟨hint, hb.trans ?_⟩
    calc
      _ ≤ 8 * M ^ 4 * ((v : ℝ) - s) ^ 2 +
          24 * (d : ℝ) ^ 2 * η ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2 := by gcongr
      _ = C * ((v : ℝ) - s) ^ 2 := by
        change _ = (8 * (M : ℝ) ^ 4 + 24 * (d : ℝ) ^ 2 * η ^ 2 * (A : ℝ) ^ 4) * ((v : ℝ) - s) ^ 2
        ring

/-- The constructed optimizer Euler limit has an actual continuous
modification at every physical time, Section 4.3 (2)--(3), Theorem 1.
Continuity is obtained from proved increments, rather than assumed as
a hypothesis about an unspecified diffusion. -/
theorem optimizerEulerLimit_exists_continuous_modification {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d) :
    ∃ Y : ℝ≥0 → BrownianSample d → EucSpace d,
      (∀ t, Measurable (Y t)) ∧
      (∀ t, Y t =ᵐ[brownianNoiseLaw d] optimizerEulerLimit method η B f σ hη hB hmodel x₀ t) ∧
      ∀ ω, Continuous (fun t => Y t ω) := by
  obtain ⟨C, hC⟩ := optimizerEulerLimit_timeChanged_isKolmogorov method η B f σ hη hB hmodel x₀
  obtain ⟨Z, hZm, hZeq, hZholder⟩ := exists_modification_holder_iSup
    (p := fun _ => 4) (q := fun _ => 2) (M := fun _ => C)
    isCoverWithBoundedCoveringNumber_Ico_nnreal (fun _ => hC.IsAEKolmogorovProcess)
    (fun _ => by finiteness) (by norm_num) (fun _ => by norm_num)
  have hZcont (ω : BrownianSample d) : Continuous (fun t => Z t ω) := by
    refine continuous_iff_continuousAt.mpr fun t => ?_
    obtain ⟨U, hU, K, hK⟩ := hZholder ω t (1 / 8) (by norm_num) (by norm_num [ciSup_const])
    exact (hK.continuousOn (by norm_num)).continuousAt hU
  refine ⟨fun t => Z (eulerTimeExpand t), fun t => hZm _, ?_,
    fun ω => (hZcont ω).comp eulerTimeExpand_continuous⟩
  intro t
  simpa only [eulerTimeCompress_expand] using hZeq (eulerTimeExpand t)

/-- Joint nonvacuity of continuous-limit hypotheses, Section 4.3:
positive rate and batch, flat loss, and unit Gaussian noise. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
