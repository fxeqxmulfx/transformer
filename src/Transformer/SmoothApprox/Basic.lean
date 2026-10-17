/-
# Smooth approximation of compactly supported functions

On a finite-dimensional real space, a continuous function with compact
support is uniformly as close as wished to a smooth function with compact
support, and a `C¹` one is so in the `C¹` norm.  The approximant is the
mollification `φ ⋆ f` by a normed bump function `φ` whose radius is below the
modulus of uniform continuity of `f` (`ContDiffBump.dist_normed_convolution_le`);
its derivative is `φ ⋆ Df` (`HasCompactSupport.hasFDerivAt_convolution_right`).

This is how a weak formulation tested against `C_c^∞` functions extends to
`C_c` and `C¹_c` ones, as arXiv:2305.05465v6, §6, uses for `d:solconti`.
-/

import Mathlib.Analysis.Calculus.BumpFunction.Convolution
import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.MeasureTheory.Measure.Haar.Basic

open MeasureTheory Metric ContinuousLinearMap
open scoped Convolution

namespace Transformer
namespace SmoothApprox

/-- A continuous map with compact support moves points closer than its
modulus of uniform continuity by at most `ε`. -/
theorem exists_pos_dist_lt {X Y : Type*} [PseudoMetricSpace X] [PseudoMetricSpace Y] [Zero Y]
    {f : X → Y} (hf : Continuous f) (hc : HasCompactSupport f) {ε : ℝ} (hε : 0 < ε) :
    ∃ δ > 0, ∀ x y : X, dist y x < δ → dist (f y) (f x) ≤ ε := by
  obtain ⟨δ, hδ, h⟩ := Metric.uniformContinuous_iff.1 (hc.uniformContinuous_of_continuous hf) ε hε
  exact ⟨δ, hδ, fun x y hxy => (h hxy).le⟩

variable {G : Type*} [NormedAddCommGroup G] [NormedSpace ℝ G] [FiniteDimensional ℝ G]
  [MeasurableSpace G] [BorelSpace G]
  {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]

/-- The mollification `φ ⋆ f` of `f` by the normed bump function `φ`, against
the Haar measure. -/
noncomputable def mollify (φ : ContDiffBump (0 : G)) (f : G → F) : G → F :=
  φ.normed Measure.addHaar ⋆[lsmul ℝ ℝ, Measure.addHaar] f

omit [CompleteSpace F] in
/-- A mollification of a continuous function is smooth. -/
theorem contDiff_mollify (φ : ContDiffBump (0 : G)) {f : G → F} (hf : Continuous f) :
    ContDiff ℝ (⊤ : ℕ∞) (mollify φ f) :=
  φ.hasCompactSupport_normed.contDiff_convolution_left _ φ.contDiff_normed hf.locallyIntegrable

omit [CompleteSpace F] in
/-- A mollification of a function with compact support has compact support. -/
theorem hasCompactSupport_mollify (φ : ContDiffBump (0 : G)) {f : G → F}
    (hf : HasCompactSupport f) : HasCompactSupport (mollify φ f) :=
  φ.hasCompactSupport_normed.convolution _ hf

/-- Mollifying moves a value by at most the oscillation of `f` on the bump's
ball. -/
theorem dist_mollify_le (φ : ContDiffBump (0 : G)) {f : G → F} (hf : Continuous f) {x : G}
    {ε : ℝ} (hε : ∀ y ∈ ball x φ.rOut, dist (f y) (f x) ≤ ε) :
    dist (mollify φ f x) (f x) ≤ ε :=
  φ.dist_normed_convolution_le hf.aestronglyMeasurable hε

omit [CompleteSpace F] in
/-- **The derivative of a mollification is the mollification of the
derivative**, for a `C¹` function with compact support. -/
theorem hasFDerivAt_mollify (φ : ContDiffBump (0 : G)) {f : G → F} (hf : ContDiff ℝ 1 f)
    (hc : HasCompactSupport f) (x : G) :
    HasFDerivAt (mollify φ f) (mollify φ (fderiv ℝ f) x) x := by
  have h := hc.hasFDerivAt_convolution_right (lsmul ℝ ℝ)
    (φ.integrable_normed (μ := Measure.addHaar)).locallyIntegrable hf x
  have he : (φ.normed Measure.addHaar ⋆[(lsmul ℝ ℝ : ℝ →L[ℝ] F →L[ℝ] F).precompR G,
      Measure.addHaar] fderiv ℝ f) x = mollify φ (fderiv ℝ f) x := by
    simp only [mollify, convolution_def]
    congr 1
  rwa [he] at h

/-- **Uniform approximation.**  A continuous function with compact support is
within `ε` of a smooth function with compact support, everywhere. -/
theorem exists_contDiff_near {f : G → F} (hf : Continuous f) (hc : HasCompactSupport f)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ g : G → F, ContDiff ℝ (⊤ : ℕ∞) g ∧ HasCompactSupport g ∧ ∀ x, ‖g x - f x‖ ≤ ε := by
  obtain ⟨δ, hδ, hfδ⟩ := exists_pos_dist_lt hf hc hε
  let φ : ContDiffBump (0 : G) := ⟨δ / 2, δ, by positivity, by linarith⟩
  refine ⟨mollify φ f, contDiff_mollify φ hf, hasCompactSupport_mollify φ hc, fun x => ?_⟩
  rw [← dist_eq_norm]
  exact dist_mollify_le φ hf fun y hy => hfδ x y (by simpa [φ] using hy)

/-- **`C¹` approximation.**  A `C¹` function with compact support is within `ε`
of a smooth function with compact support, together with its derivative. -/
theorem exists_contDiff_near_fderiv {f : G → F} (hf : ContDiff ℝ 1 f) (hc : HasCompactSupport f)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ g : G → F, ContDiff ℝ (⊤ : ℕ∞) g ∧ HasCompactSupport g ∧
      ∀ x, ‖g x - f x‖ ≤ ε ∧ ‖fderiv ℝ g x - fderiv ℝ f x‖ ≤ ε := by
  have hDf : Continuous (fderiv ℝ f) := hf.continuous_fderiv one_ne_zero
  obtain ⟨δ₁, hδ₁, h₁⟩ := exists_pos_dist_lt hf.continuous hc hε
  obtain ⟨δ₂, hδ₂, h₂⟩ := exists_pos_dist_lt hDf (hc.fderiv ℝ) hε
  set δ := min δ₁ δ₂
  have hδ : 0 < δ := lt_min hδ₁ hδ₂
  let φ : ContDiffBump (0 : G) := ⟨δ / 2, δ, by positivity, by linarith⟩
  refine ⟨mollify φ f, contDiff_mollify φ hf.continuous, hasCompactSupport_mollify φ hc,
    fun x => ⟨?_, ?_⟩⟩
  · rw [← dist_eq_norm]
    exact dist_mollify_le φ hf.continuous fun y hy =>
      h₁ x y ((by simpa [φ] using hy : dist y x < δ).trans_le (min_le_left _ _))
  · rw [(hasFDerivAt_mollify φ hf hc x).fderiv, ← dist_eq_norm]
    exact dist_mollify_le φ hDf fun y hy =>
      h₂ x y ((by simpa [φ] using hy : dist y x < δ).trans_le (min_le_right _ _))

/-- The hypotheses of the two approximation theorems are satisfiable: the
zero function on `ℝ` is smooth with compact support, and `1 > 0`. -/
example : ContDiff ℝ 1 (0 : ℝ → ℝ) ∧ HasCompactSupport (0 : ℝ → ℝ) ∧ (0 : ℝ) < 1 :=
  ⟨contDiff_const, HasCompactSupport.zero, one_pos⟩

end SmoothApprox
end Transformer
