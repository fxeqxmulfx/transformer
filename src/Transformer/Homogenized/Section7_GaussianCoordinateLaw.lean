/-
# The actual joint Gaussian law of the original independent coordinates

ArXiv:2604.01978v1, §7, the Gaussian variables used in `eq:SIGMA_R`,
under assumption (G), `eq: tformers.at.initialization`.
The original independently distributed Gaussian coordinates have their
actual product law. Scaling an actual standard Gaussian vector coordinate
by coordinate realizes that product as an actual continuous linear image.

The argument never divides by a coordinate variance. Zero variances and
singular product laws are therefore included. The scaling preserves
the original coordinate variances. Original measurability and
independence identify the actual joint law on the original probability
space. Applied to the original value, query and key entries, this supplies
their joint Gaussian shape, before taking the linear feature images used
in the Gram-covariance argument. The attention matrix itself is a product
of Gaussian matrices and is not assigned a Gaussian law.
-/

import Transformer.Homogenized.GaussianEntries
import Transformer.Homogenized.GaussianEnsemble
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Basic
import Mathlib.Probability.Distributions.Gaussian.Multivariate

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual diagonal scaling from a standard Euclidean Gaussian
to the original coordinate space. Source: arXiv:2604.01978v1, §7,
the independent Gaussian coordinates behind `eq:SIGMA_R`. -/
noncomputable def gaussianCoordinateScale {ι : Type*} [Fintype ι] (σ : ι → ℝ≥0) :
    EuclideanSpace ℝ ι →L[ℝ] (ι → ℝ) :=
  ContinuousLinearMap.pi fun i => (σ i : ℝ) • EuclideanSpace.proj (𝕜 := ℝ) i

@[simp] theorem gaussianCoordinateScale_apply {ι : Type*} [Fintype ι]
    (σ : ι → ℝ≥0) (z : EuclideanSpace ℝ ι) (i : ι) :
    gaussianCoordinateScale σ z i = (σ i : ℝ) * z i := rfl

/-- The original product of centered Gaussian coordinate laws is an
actual linear image of the standard vector law, including zero variances.
Source: arXiv:2604.01978v1, §7, the original Gaussian representation
in the covariance construction of `eq:SIGMA_R`. -/
theorem gaussian_pi_eq_stdGaussian_map_coordinates {ι : Type*} [Fintype ι]
    (σ : ι → ℝ≥0) :
    (Measure.pi fun i => gaussianReal 0 (σ i ^ 2)) =
      (stdGaussian (EuclideanSpace ℝ ι)).map (gaussianCoordinateScale σ) := by
  symm
  rw [← map_pi_eq_stdGaussian, Measure.map_map (gaussianCoordinateScale σ).continuous.measurable
    (PiLp.continuous_toLp 2 _).measurable]
  change (Measure.pi fun _ : ι => gaussianReal 0 1).map (fun z i => (σ i : ℝ) * z i) = _
  rw [Measure.pi_map_pi fun i => (measurable_const_mul (σ i : ℝ)).aemeasurable]
  congr 1
  funext i
  rw [gaussianReal_map_const_mul, mul_zero, mul_one]
  congr 1

/-- Every original centered Gaussian product has its genuine joint
Gaussian shape. Source: arXiv:2604.01978v1, §7, the jointly Gaussian
entries whose linear images give the features in `eq:SIGMA_R`. -/
theorem isGaussian_pi_centered_coordinates {ι : Type*} [Fintype ι] (σ : ι → ℝ≥0) :
    IsGaussian (Measure.pi fun i => gaussianReal 0 (σ i ^ 2)) := by
  rw [gaussian_pi_eq_stdGaussian_map_coordinates σ]
  exact ((IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EuclideanSpace ℝ ι))).map
    (gaussianCoordinateScale σ)).isGaussian_map

/-- The original product coordinate space carries its genuine
Gaussian identity random vector, including singular original laws.
Source: arXiv:2604.01978v1, §7, the original jointly Gaussian
coordinates used to construct `eq:SIGMA_R`. -/
theorem hasGaussianLaw_pi_coordinate_identity {ι : Type*} [Fintype ι] (σ : ι → ℝ≥0) :
    HasGaussianLaw (id : (ι → ℝ) → (ι → ℝ)) (Measure.pi fun i => gaussianReal 0 (σ i ^ 2)) := by
  have : IsGaussian (Measure.pi fun i => gaussianReal 0 (σ i ^ 2)) :=
    isGaussian_pi_centered_coordinates σ
  exact IsGaussian.hasGaussianLaw_id

/-- The original independent centered Gaussian coordinates form an
actual Gaussian random vector on their original probability space.
Source: arXiv:2604.01978v1, §7, the joint-law step before `eq:SIGMA_R`.
Every original coordinate measurability and actual marginal law is explicit. -/
theorem hasGaussianLaw_independent_centered_coordinates {Ω ι : Type*}
    [MeasurableSpace Ω] [Fintype ι] (P : Measure Ω) [IsProbabilityMeasure P]
    (X : ι → Ω → ℝ) (σ : ι → ℝ≥0) (hX : ∀ i, AEMeasurable (X i) P)
    (hind : iIndepFun X P) (hmap : ∀ i, P.map (X i) = gaussianReal 0 (σ i ^ 2)) :
    HasGaussianLaw (fun ω i => X i ω) P := by
  refine ⟨AEMeasurable.of_eval hX, ?_⟩
  rw [hind.map_fun_eq_pi_map hX]
  simpa only [hmap] using isGaussian_pi_centered_coordinates σ

/-- The original value, query and key entry vector has an actual
joint Gaussian law under the source's independently Gaussian entries.
Source: arXiv:2604.01978v1, §2.5, `eq: tformers.at.initialization`,
and §7, its original linear Gaussian features in `eq:SIGMA_R`. -/
theorem hasGaussianLaw_head_entry_vector {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
    (hV : Measurable Vr) (hW : Measurable Wr) (hW' : Measurable Wr')
    (hind : iIndepFun (headEntries Vr Wr Wr') P) (σV σA : ℝ≥0)
    (hmV : ∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 (σV ^ 2))
    (hmW : ∀ i j, P.map (fun ω => Wr ω i j) = gaussianReal 0 (σA ^ 2))
    (hmW' : ∀ i j, P.map (fun ω => Wr' ω i j) = gaussianReal 0 (σA ^ 2)) :
    HasGaussianLaw (fun ω p => headEntries Vr Wr Wr' p ω) P := by
  let σ : GaussIdx d → ℝ≥0 := fun p => if p.1 = 0 then σV else σA
  apply hasGaussianLaw_independent_centered_coordinates P (headEntries Vr Wr Wr') σ
    (fun p => (measurable_headEntries hV hW hW' p).aemeasurable) hind
  intro p
  unfold headEntries
  obtain ⟨a, i, j⟩ := p
  fin_cases a
  · simpa [σ] using hmV i j
  · simpa [σ] using hmW i j
  · simpa [σ] using hmW' i j

/-- A genuinely singular product, with one standard Gaussian and
one zero-variance coordinate, satisfies every coordinate-law hypothesis. -/
example : ∃ (σ : Fin 2 → ℝ≥0) (P : Measure (Fin 2 → ℝ)),
    σ 0 = 1 ∧ σ 1 = 0 ∧ IsProbabilityMeasure P ∧
    (∀ i : Fin 2, AEMeasurable (fun z : Fin 2 → ℝ => z i) P) ∧
    iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i) P ∧
    (∀ i : Fin 2, P.map (fun z : Fin 2 → ℝ => z i) = gaussianReal 0 (σ i ^ 2)) := by
  let σ : Fin 2 → ℝ≥0 := ![1, 0]
  let μ (i : Fin 2) := gaussianReal 0 (σ i ^ 2)
  let P := Measure.pi μ
  refine ⟨σ, P, rfl, rfl, inferInstance, fun i => (measurable_pi_apply i).aemeasurable,
    iIndepFun_pi (X := fun _ : Fin 2 => id) (fun _ => measurable_id.aemeasurable), ?_⟩
  intro i
  exact (measurePreserving_eval μ i).map_eq

/-- The canonical nondegenerate source ensemble realizes all original
head-entry hypotheses, with variance one in both types of Gaussian block. -/
example : ∃ (P : Measure (GaussIdx 2 → ℝ)) (Vr Wr Wr' : (GaussIdx 2 → ℝ) → Matrix (Fin 2) (Fin 2) ℝ),
    IsProbabilityMeasure P ∧ Measurable Vr ∧ Measurable Wr ∧ Measurable Wr' ∧
    iIndepFun (headEntries Vr Wr Wr') P ∧
    (∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 ((1 : ℝ≥0) ^ 2)) ∧
    (∀ i j, P.map (fun ω => Wr ω i j) = gaussianReal 0 ((1 : ℝ≥0) ^ 2)) ∧
    (∀ i j, P.map (fun ω => Wr' ω i j) = gaussianReal 0 ((1 : ℝ≥0) ^ 2)) := by
  refine ⟨gaussEnsemble 2 1 1, gaussBlock 0, gaussBlock 1, gaussBlock 2, inferInstance,
    measurable_gaussBlock _, measurable_gaussBlock _, measurable_gaussBlock _, ?_, ?_, ?_, ?_⟩
  · convert (iIndepFun_pi (μ := fun p : GaussIdx 2 => gaussianReal 0 (entryVar 1 1 p))
      (X := fun _ : GaussIdx 2 => id) (fun _ => measurable_id.aemeasurable)) using 1
    · funext p ω
      obtain ⟨a, i, j⟩ := p
      fin_cases a <;> simp [headEntries, gaussBlock]
    · rfl
  · intro i j
    simpa only [entryVar_zero] using map_gaussBlock (d := 2) 1 1 0 i j
  · intro i j
    simpa only [entryVar_one] using map_gaussBlock (d := 2) 1 1 1 i j
  · intro i j
    simpa only [entryVar_two] using map_gaussBlock (d := 2) 1 1 2 i j

end Transformer.Homogenized
