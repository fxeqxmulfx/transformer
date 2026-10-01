/-
# Coordinate integration by parts for the genuine vector Gaussian law

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Splitting off one actual independent coordinate reduces the vector
Gaussian generator calculation to the proved scalar identity.
-/

import Transformer.BatchSize.Section4_GaussianIntegration
import Transformer.BatchSize.Section4_DiagonalNoise
import Mathlib.MeasureTheory.Integral.Pi

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal

noncomputable section

namespace Transformer.BatchSize

/-- The genuine standard Gaussian vector innovation law used for
Section 4.3's diagonal Brownian generator. -/
def standardGaussianVectorLaw (d : ℕ) : Measure (Fin d → ℝ) :=
  Measure.pi (fun _ => gaussianReal 0 1)

/-- Normalization of the actual Gaussian vector law, Section 4.3. -/
instance standardGaussianVectorLaw_probability (d : ℕ) :
    IsProbabilityMeasure (standardGaussianVectorLaw d) := by
  dsimp only [standardGaussianVectorLaw]
  infer_instance

/-- The vector generator innovation is exactly the standard
innovation already used by the discrete sampler, Section 4.3. -/
theorem standardGaussianVectorLaw_eq_noiseLaw (d : ℕ) :
    standardGaussianVectorLaw d = diagonalNoiseLaw 1 (fun _ : Fin d => 0) (fun _ => 1) := by
  simp [standardGaussianVectorLaw, diagonalNoiseLaw, gradientNoiseLaw]

/-- Fubini's identity splitting off an arbitrary Gaussian coordinate,
Section 4.3 (2)--(3). Integrability is of the actual full product law. -/
theorem standardGaussianVector_integral_split {n : ℕ} (k : Fin (n + 1))
    (g : (Fin (n + 1) → ℝ) → ℝ) (hg : Integrable g (standardGaussianVectorLaw (n + 1))) :
    (∫ z, g z ∂standardGaussianVectorLaw (n + 1)) =
      ∫ z, ∫ u, g (k.insertNth u z) ∂gaussianReal 0 1 ∂standardGaussianVectorLaw n := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) k
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => gaussianReal 0 1) k
  have hsym := MeasurePreserving.symm e hmp
  have hprod := (hsym.integrable_comp hg.aestronglyMeasurable).mpr hg
  have heq := hsym.integral_comp e.symm.measurableEmbedding g
  calc
    _ = ∫ p, g (e.symm p) ∂(gaussianReal 0 1).prod (standardGaussianVectorLaw n) := heq.symm
    _ = _ := integral_prod_symm (fun p => g (e.symm p)) hprod

/-- Nonvacuity of the full-vector Fubini integrability hypothesis,
Section 4.3: a nonzero constant observable in two dimensions. -/
example : Integrable (fun _ : Fin 2 → ℝ => (1 : ℝ)) (standardGaussianVectorLaw 2) := integrable_const 1

/-- Gaussian coordinate integration by parts, E[Z_k g(Z)]=E[partial_k g(Z)],
Section 4.3 (2)--(3). Each slice derivative is the genuine derivative;
both observables are continuous and bounded on the full vector space. -/
theorem standardGaussianVector_integration_by_parts {n : ℕ} (k : Fin (n + 1))
    (g g' : (Fin (n + 1) → ℝ) → ℝ) (hgc : Continuous g) (hg'c : Continuous g')
    (hg : ∀ (z : Fin n → ℝ) (u : ℝ), HasDerivAt (fun v => g (k.insertNth v z))
      (g' (k.insertNth u z)) u)
    (C D : ℝ) (hC : ∀ z, |g z| ≤ C) (hD : ∀ z, |g' z| ≤ D) :
    (∫ z, z k * g z ∂standardGaussianVectorLaw (n + 1)) =
      ∫ z, g' z ∂standardGaussianVectorLaw (n + 1) := by
  have hg'i : Integrable g' (standardGaussianVectorLaw (n + 1)) :=
    (MemLp.of_bound hg'c.aestronglyMeasurable D (Eventually.of_forall fun z =>
      by simpa only [Real.norm_eq_abs] using hD z)).integrable (by norm_num : (1 : ℝ≥0∞) ≤ 1)
  have hzi : Integrable (fun z : Fin (n + 1) → ℝ => z k) (standardGaussianVectorLaw (n + 1)) := by
    have hid := (memLp_id_gaussianReal (μ := 0) (v := 1) 1).integrable (by norm_num)
    exact ((measurePreserving_eval (fun _ : Fin (n + 1) => gaussianReal 0 1) k).integrable_comp
      aestronglyMeasurable_id).mpr hid
  have hzgi : Integrable (fun z => z k * g z) (standardGaussianVectorLaw (n + 1)) := by
    simpa only [mul_comm] using hzi.bdd_mul hgc.aestronglyMeasurable
      (Eventually.of_forall fun z => by simpa only [Real.norm_eq_abs] using hC z)
  rw [standardGaussianVector_integral_split k _ hzgi, standardGaussianVector_integral_split k _ hg'i]
  apply integral_congr_ae
  exact Eventually.of_forall fun z => by
    simpa only [Fin.insertNth_apply_same] using standardGaussian_integration_by_parts (hg z)
      (hg'c.comp (Continuous.finInsertNth k continuous_id continuous_const))
      (fun u => hC (k.insertNth u z)) (fun u => hD (k.insertNth u z))

/-- Joint nonvacuity of vector integration-by-parts hypotheses,
Section 4.3: sin of the first coordinate, with genuine cosine derivative. -/
example : Continuous (fun z : Fin 2 → ℝ => Real.sin (z 0)) ∧
    Continuous (fun z : Fin 2 → ℝ => Real.cos (z 0)) ∧
    (∀ (z : Fin 1 → ℝ) (u : ℝ), HasDerivAt
      (fun v => Real.sin (Fin.insertNth (α := fun _ : Fin 2 => ℝ) (0 : Fin 2) v z 0))
      (Real.cos (Fin.insertNth (α := fun _ : Fin 2 => ℝ) (0 : Fin 2) u z 0)) u) ∧
    (∀ z : Fin 2 → ℝ, |Real.sin (z 0)| ≤ 1) ∧
    (∀ z : Fin 2 → ℝ, |Real.cos (z 0)| ≤ 1) := by
  refine ⟨Real.continuous_sin.comp (continuous_apply 0),
    Real.continuous_cos.comp (continuous_apply 0), ?_,
    fun z => Real.abs_sin_le_one (z 0), fun z => Real.abs_cos_le_one (z 0)⟩
  intro z u
  simpa only [Fin.insertNth_apply_same] using Real.hasDerivAt_sin u

end Transformer.BatchSize
