module

public import Transformer.Perceptron.Section6_NormalizedCircleAtomicity
public import Transformer.Perceptron.Atomicity
public import Transformer.Metastability.Section4_UniformLaw
public import Mathlib.MeasureTheory.Measure.Typeclasses.NullSingletonClass

/-!
# Null support and singularity for normalized circle stationary measures

ArXiv:2601.21366v2, §6, `prop: unified.log` and
`rem:unifiedlog-B`. The proved finite circle support gives the
source's null-support and singularity conclusions in dimension two.
The higher-dimensional conclusions remain separate statements.

The uniform spherical law has zero mass on singletons in every
dimension at least two. Thus finite actual support is null for any
law satisfying the original rotation-invariance predicate. The
complement of support is null for the original measure, giving actual
mutual singularity. All uniform-law hypotheses below are exhibited
by the already constructed normalized spherical Lebesgue law.
-/

@[expose] public section

open MeasureTheory Set

namespace Transformer.Perceptron

/-- A uniform law has no atoms in dimensions at least two.
Source: arXiv:2601.21366v2, §6, the uniform measure in `prop: unified.log`.
The actual rotation-invariance predicate gives equal singleton masses;
infinitely many sphere points force that common mass to be zero. -/
theorem nullSingletonClass_of_isUniformOn {d : ℕ} (hd : 2 ≤ d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) : NullSingletonClass ν := by
  let : IsProbabilityMeasure ν := hν.1
  exact ⟨fun x => Perspective.measure_singleton_eq_zero_of_invariant d hd ν hν.2 x⟩

/-- A finite actual support is null for every original uniform law.
Source: arXiv:2601.21366v2, §6, the null-support consequence of
finite atomicity in `prop: unified.log`, part (b). -/
theorem measure_finite_support_eq_zero_of_uniform {d : ℕ} (hd : 2 ≤ d)
    (μ : Measure (SSphere d)) (hfinite : μ.support.Finite)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) : ν μ.support = 0 := by
  let : NullSingletonClass ν := nullSingletonClass_of_isUniformOn hd ν hν
  exact hfinite.measure_zero ν

/-- A finitely atomic probability measure is singular with respect to
any actual uniform law in dimension at least two.
Source: arXiv:2601.21366v2, §6, the singularity clause of `prop: unified.log`. -/
theorem mutuallySingular_uniform_of_isFinitelyAtomic {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (hμ : IsFinitelyAtomic μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    (μ : Measure (SSphere d)) ⟂ₘ ν :=
  mutuallySingular_of_measure_support_eq_zero
    (measure_finite_support_eq_zero_of_uniform hd _ (finite_support_of_finitely_atomic μ hμ) ν hν)

/-- An actual uniform probability law in dimension at least two is not
finitely atomic. Source: arXiv:2601.21366v2, §6, the separation of
finite atomic stationary measures from the uniform law. -/
theorem not_isFinitelyAtomic_of_isUniformOn {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (hν : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    ¬ IsFinitelyAtomic μ := by
  intro hμ
  have hs := mutuallySingular_uniform_of_isFinitelyAtomic hd μ hμ (μ : Measure (SSphere d)) hν
  have hz := (Measure.MutuallySingular.self_iff (μ : Measure (SSphere d))).mp hs
  have hm := congrArg (fun ρ : Measure (SSphere d) => ρ univ) hz
  simp at hm

/-- The source's actual null-support conclusion in dimension two for
normalized matrix attention. Source: arXiv:2601.21366v2, §6,
`prop: unified.log` and `rem:unifiedlog-B`.
Invertibility alone suffices, as in the proved finite-support result. -/
theorem normalized_circleMap_measure_support_eq_zero
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν → ν (μ : Measure (SSphere 2)).support = 0 :=
  fun ν hν => measure_finite_support_eq_zero_of_uniform le_rfl _
    (finite_support_of_normalizedStationaryMap_circle B hB ω a hnd μ hμ) ν hν

/-- The original normalized matrix stationary measure is actually
singular with respect to the uniform law on the circle.
Source: arXiv:2601.21366v2, §6, the singularity clause of `rem:unifiedlog-B`. -/
theorem normalized_circleMap_mutuallySingular_uniform
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν → (μ : Measure (SSphere 2)) ⟂ₘ ν :=
  fun ν hν => mutuallySingular_of_measure_support_eq_zero
    (normalized_circleMap_measure_support_eq_zero B hB ω a hnd μ hμ ν hν)

/-- The actual scalar normalized stationary support is uniform-null
on the circle. Source: arXiv:2601.21366v2, §6, `prop: unified.log`.
The source's positive parameter is retained in this scalar consequence. -/
theorem normalized_circle_measure_support_eq_zero (β : ℝ) (hβ : 0 < β)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν → ν (μ : Measure (SSphere 2)).support = 0 :=
  fun ν hν => measure_finite_support_eq_zero_of_uniform le_rfl _
    (finite_support_of_normalizedStationary_circle β hβ.ne' ω a hnd μ hμ) ν hν

/-- The original scalar normalized stationary measure is singular
with respect to every actual uniform circle law.
Source: arXiv:2601.21366v2, §6, the singularity clause of `prop: unified.log`. -/
theorem normalized_circle_mutuallySingular_uniform (β : ℝ) (hβ : 0 < β)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν → (μ : Measure (SSphere 2)) ⟂ₘ ν :=
  fun ν hν => mutuallySingular_of_measure_support_eq_zero
    (normalized_circle_measure_support_eq_zero β hβ ω a hnd μ hμ ν hν)

/-- The actual uniform circle probability measure cannot be normalized
stationary under the source's non-degeneracy and invertibility conditions.
Source: arXiv:2601.21366v2, §6, `prop: unified.log` and `rem:unifiedlog-B`. -/
theorem not_uniform_of_normalizedStationaryMap_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    ¬ Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) := by
  intro hν
  exact not_isFinitelyAtomic_of_isUniformOn le_rfl μ hν
    (isFinitelyAtomic_of_normalizedStationaryMap_circle B hB ω a hnd μ hμ)

/-- The generic finite-support, scalar stationary, matrix stationary,
non-degeneracy, positive parameter and actual uniform-law inputs all
hold together for the pinned Dirac and the constructed uniform circle law.
Source: arXiv:2601.21366v2, §6, `prop: unified.log` and `rem:unifiedlog-B`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
    IsNormalizedStationary 1 (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    IsFinitelyAtomic (Perspective.diracProb 2 (basePoint 1)) ∧
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support.Finite ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) := by
  have hfinite := isFinitelyAtomic_of_normalizedStationaryMap_circle _ Function.bijective_id
    _ _ isNonDegenerate_pin _ (isNormalizedStationaryMap_relu_pin 0 (basePoint 1))
  exact ⟨le_rfl, one_pos, Function.bijective_id, isNonDegenerate_pin,
    isNormalizedStationary_relu_pin 1 0 (basePoint 1),
    isNormalizedStationaryMap_relu_pin 0 (basePoint 1), hfinite,
    finite_support_of_finitely_atomic _ hfinite, Metastability.isUniformOn_uniformLaw (by norm_num)⟩

/-- The uniform-probability hypothesis is also realized by an actual
probability measure, separately from the finitely supported witness.
Source: arXiv:2601.21366v2, §6, the original uniform spherical law. -/
example : ∃ μ : Perspective.ProbSphere 2, Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨⟨MeanField.uniformLaw 2, hν.1⟩, hν⟩

end Transformer.Perceptron
