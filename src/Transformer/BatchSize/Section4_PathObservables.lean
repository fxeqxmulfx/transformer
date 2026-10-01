/-
# Measurable compensated and past-cylinder path observables

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
These are observables on the actual continuous path space, rather than
an assumed SDE law. The adapted cylinder representative uses the proved
Euler state limits at the finitely many specified past times.
-/

import Transformer.BatchSize.Section4_OptimizerMartingale

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Time integration of a continuous generator along genuine
continuous paths is measurable, Section 4.3 (2)--(3). -/
theorem continuousPath_intervalIntegral_measurable {d : ℕ} (g : EucSpace d → ℝ)
    (hg : Continuous g) (s t : ℝ) (hst : s ≤ t) :
    Measurable (fun ω : DiffusionPath d => ∫ u in s..t, g (ω u)) := by
  simp_rw [intervalIntegral.integral_of_le hst]
  have hm := hg.measurable.comp (continuousPath_measurable_time_sample
    (id : DiffusionPath d → DiffusionPath d) measurable_id)
  exact (hm.comp measurable_swap).stronglyMeasurable.integral_prod_right'.measurable

/-- The actual optimizer compensated path observable is measurable,
Section 4.3 (2)--(3), under the stated regularity hypotheses. -/
theorem compensatedIncrement_measurable {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) (s t : ℝ) (hst : s ≤ t) :
    Measurable (compensatedIncrement method η B f σ φ s t) :=
  ((hφ.1.continuous.measurable.comp (ContinuousMap.measurable_eval t)).sub
    (hφ.1.continuous.measurable.comp (ContinuousMap.measurable_eval s))).sub
      (continuousPath_intervalIntegral_measurable _
        (diffusionGenerator_continuous_bounded method η B f σ hη hB hmodel φ hφ).1 s t hst)

/-- Every bounded measurable cylinder of the actual continuous
optimizer path has a genuine adapted representative in the whole
Brownian past, Section 4.3 (2)--(3). Finite-time modification equalities
are sufficient; no simultaneous equality at all times is assumed. -/
theorem optimizerEulerPath_cylinder_adapted {d n : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d)
    (s : ℝ) (hs : 0 ≤ s) (times : Fin n → ℝ) (htimes : ∀ i, 0 ≤ times i ∧ times i ≤ s)
    (F : (Fin n → EucSpace d) → ℝ) (hFm : Measurable F) (hF1 : ∀ y, |F y| ≤ 1) :
    ∃ H : BrownianSample d → ℝ,
      StronglyMeasurable[brownianFiltration d s.toNNReal] H ∧ (∀ ω, |H ω| ≤ 1) ∧
      (fun ω => F (fun i => optimizerEulerPath method η B f σ hη hB hmodel x₀ ω (times i)))
        =ᵐ[brownianNoiseLaw d] H := by
  let H (ω : BrownianSample d) := F (fun i =>
    optimizerEulerLimit method η B f σ hη hB hmodel x₀ (times i).toNNReal ω)
  have hHm : StronglyMeasurable[brownianFiltration d s.toNNReal] H := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s.toNNReal
    apply Measurable.stronglyMeasurable
    apply hFm.comp
    apply Measurable.of_eval
    intro i
    have hts : (times i).toNNReal ≤ s.toNNReal := by
      apply NNReal.coe_le_coe.mp
      simpa only [Real.coe_toNNReal _ hs, Real.coe_toNNReal _ (htimes i).1] using (htimes i).2
    exact ((optimizerEulerLimit_adapted method η B f σ hη hB hmodel x₀ (times i).toNNReal).mono
      ((brownianFiltration d).mono hts)).measurable
  refine ⟨H, hHm, (fun ω => hF1 _), ?_⟩
  have hcoords : ∀ᵐ ω ∂brownianNoiseLaw d, ∀ i : Fin n,
      optimizerEulerPath method η B f σ hη hB hmodel x₀ ω (times i) =
        optimizerEulerLimit method η B f σ hη hB hmodel x₀ (times i).toNNReal ω := by
    apply ae_all_iff.mpr
    intro i
    have h := optimizerEulerPath_eval_ae_eq method η B f σ hη hB hmodel x₀ (times i).toNNReal
    simp only [Real.coe_toNNReal _ (htimes i).1] at h
    exact h
  filter_upwards [hcoords] with ω hω
  dsimp only [H]
  congr 1
  funext i
  exact hω i

/-- Joint nonvacuity of all compensated-cylinder hypotheses,
Section 4.3: positive rate and batch, flat loss, unit noise,
an actual positive past time and nonzero normalized observables. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 2 ∧
    (∀ i : Fin 1, 0 ≤ (fun i : Fin 1 => (i : ℝ) + 1) i ∧
      (fun i : Fin 1 => (i : ℝ) + 1) i ≤ 2) ∧
    Measurable (fun _ : Fin 1 → EucSpace 1 => (1 : ℝ)) ∧ |(1 : ℝ)| ≤ 1 ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨by norm_num, by norm_num, regularGaussianModel_flat 1, by norm_num,
    (fun i => by fin_cases i; norm_num), measurable_const, by norm_num, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
