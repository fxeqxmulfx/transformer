/-
# The actual Gram covariance of the original Gaussian head features

ArXiv:2604.01978v1, §7, `eq:SIGMA_R`, under initialization (G).
The source switches its initialization `A = WW′ᵀ` to `A = W′ᵀW`
in the Gram proof. We retain the exact original head scores by using
the original features `h_i = W′ᵀx_i` and `g_j = Wᵀx_j`.
Their covariance is still exactly the source's two Gram blocks.

The feature means, integrability and covariance below are derived from
the original entry measurability, independence and Gaussian marginal
laws. In particular, the displayed covariance matrix is not assumed
to be the covariance of the actual features. Its equality with their
actual second moments is proved by expanding their original linear
combinations and retaining precisely the shared-entry terms.
The assertions allow arbitrary token vectors, including zero vectors,
linearly dependent vectors, and singular original Gaussian laws.
-/

import Transformer.Homogenized.Section7_GaussianCoordinateMoments
import Transformer.Homogenized.Section7_GaussianHeadFeatures

open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The source's explicit block Gram matrix, with zero cross-block
and unequal-coordinate entries. Its equality with the actual feature
covariance is proved below. Source: arXiv:2604.01978v1, §7, `eq:SIGMA_R`. -/
noncomputable def gaussianFeatureCovariance {d n : ℕ} (σA : ℝ≥0) (X : Idx n → EucSpace d) :
    Matrix (GaussianFeatureIdx d n) (GaussianFeatureIdx d n) ℝ := fun p q =>
  if p.1 = q.1 ∧ p.2.2 = q.2.2 then
    (σA : ℝ) ^ 2 * inner (𝕜 := ℝ) (X p.2.1) (X q.2.1) else 0

/-- The original entry indices shared by two linear feature
coordinates yield exactly their Gram block. Source: arXiv:2604.01978v1,
§7, the finite covariance calculation in `eq:SIGMA_R`. -/
theorem sum_gaussianFeature_entry_covariance {d n : ℕ} (σA : ℝ≥0) (X : Idx n → EucSpace d)
    (a b : Fin 2) (i j : Idx n) (q r : Fin d) :
    (∑ s : Fin d, ∑ t : Fin d, (X i s * X j t) *
      (if ((if a = 0 then (2 : Fin 3) else 1), s, q) =
        ((if b = 0 then (2 : Fin 3) else 1), t, r) then (σA : ℝ) ^ 2 else 0)) =
      gaussianFeatureCovariance σA X (a, i, q) (b, j, r) := by
  have hslot : ∀ a b : Fin 2,
      (if a = 0 then (2 : Fin 3) else 1) = (if b = 0 then (2 : Fin 3) else 1) ↔ a = b := by
    intro a b
    fin_cases a <;> fin_cases b <;> simp
  simp only [gaussianFeatureCovariance, Prod.mk.injEq, hslot]
  by_cases hab : a = b
  · by_cases hqr : q = r
    · simp only [hab, hqr, true_and, and_true, mul_ite, mul_zero, Finset.sum_ite_eq,
        Finset.mem_univ, ite_true]
      rw [EuclideanSpace.inner_eq_star_dotProduct]
      simp only [dotProduct, star_trivial, Finset.mul_sum]
      exact Finset.sum_congr rfl fun s hs => by ring
    · simp [hqr]
  · simp [hab]

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {d n : ℕ}

/-- The actual feature coordinate is integrable on its original
probability space. Source: arXiv:2604.01978v1, §7, `eq:SIGMA_R`. -/
theorem integrable_gaussianHeadFeature [IsFiniteMeasure P]
    (Z : GaussIdx d → Ω → ℝ) (σV σA : ℝ≥0) (hZ : ∀ p, Measurable (Z p))
    (hmap : ∀ p, P.map (Z p) = gaussianReal 0 ((if p.1 = 0 then σV else σA) ^ 2))
    (X : Idx n → EucSpace d) (p : GaussianFeatureIdx d n) :
    Integrable (fun ω => gaussianHeadFeatureMap X (fun r => Z r ω) p) P := by
  simp_rw [gaussianHeadFeatureMap_apply]
  exact integrable_gaussian_coordinate_linear_combination Z _ hZ hmap _ _

/-- The actual original Gaussian features have zero coordinate mean.
Source: arXiv:2604.01978v1, §7, the zero mean in `eq:SIGMA_R`. -/
theorem integral_gaussianHeadFeature [IsFiniteMeasure P]
    (Z : GaussIdx d → Ω → ℝ) (σV σA : ℝ≥0) (hZ : ∀ p, Measurable (Z p))
    (hmap : ∀ p, P.map (Z p) = gaussianReal 0 ((if p.1 = 0 then σV else σA) ^ 2))
    (X : Idx n → EucSpace d) (p : GaussianFeatureIdx d n) :
    ∫ ω, gaussianHeadFeatureMap X (fun r => Z r ω) p ∂P = 0 := by
  simp_rw [gaussianHeadFeatureMap_apply]
  exact integral_gaussian_coordinate_linear_combination Z _ hZ hmap _ _

/-- Actual pairs of original Gaussian feature coordinates are integrable.
Source: arXiv:2604.01978v1, §7, the second moments in `eq:SIGMA_R`. -/
theorem integrable_gaussianHeadFeature_product
    (Z : GaussIdx d → Ω → ℝ) (σV σA : ℝ≥0) (hZ : ∀ p, Measurable (Z p))
    (hmap : ∀ p, P.map (Z p) = gaussianReal 0 ((if p.1 = 0 then σV else σA) ^ 2))
    (X : Idx n → EucSpace d) (p q : GaussianFeatureIdx d n) :
    Integrable (fun ω => gaussianHeadFeatureMap X (fun r => Z r ω) p *
      gaussianHeadFeatureMap X (fun r => Z r ω) q) P := by
  simp_rw [gaussianHeadFeatureMap_apply]
  exact integrable_gaussian_coordinate_linear_product Z _ hZ hmap _ _ _ _

/-- The original features have exactly the source's block Gram covariance.
The zero means are proved separately above, so these actual second moments
are their actual covariance entries. Source: arXiv:2604.01978v1, §7,
`eq:SIGMA_R`, with the original initialization convention retained. -/
theorem integral_gaussianHeadFeature_product
    (Z : GaussIdx d → Ω → ℝ) (σV σA : ℝ≥0) (hZ : ∀ p, Measurable (Z p))
    (hind : iIndepFun Z P)
    (hmap : ∀ p, P.map (Z p) = gaussianReal 0 ((if p.1 = 0 then σV else σA) ^ 2))
    (X : Idx n → EucSpace d) (p q : GaussianFeatureIdx d n) :
    ∫ ω, gaussianHeadFeatureMap X (fun r => Z r ω) p *
      gaussianHeadFeatureMap X (fun r => Z r ω) q ∂P = gaussianFeatureCovariance σA X p q := by
  obtain ⟨a, i, r⟩ := p
  obtain ⟨b, j, s⟩ := q
  simp_rw [gaussianHeadFeatureMap_apply]
  rw [integral_gaussian_coordinate_linear_product Z _ hZ hind hmap]
  have hvar : ∀ a : Fin 2,
      (if (if a = 0 then (2 : Fin 3) else 1) = 0 then σV else σA) = σA := by
    intro a
    fin_cases a <;> simp
  simp only [hvar]
  exact sum_gaussianFeature_entry_covariance σA X a b i j r s

/-- The original independently Gaussian matrices themselves have
the stated feature covariance on their original probability space.
Source: arXiv:2604.01978v1, §7, `eq:SIGMA_R`, derived from (G)
with the transpose convention corrected to original initialization. -/
theorem integral_original_gaussianHeadFeature_product
    (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ) (σV σA : ℝ≥0)
    (hV : Measurable Vr) (hW : Measurable Wr) (hW' : Measurable Wr')
    (hind : iIndepFun (headEntries Vr Wr Wr') P)
    (hmV : ∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 (σV ^ 2))
    (hmW : ∀ i j, P.map (fun ω => Wr ω i j) = gaussianReal 0 (σA ^ 2))
    (hmW' : ∀ i j, P.map (fun ω => Wr' ω i j) = gaussianReal 0 (σA ^ 2))
    (X : Idx n → EucSpace d) (p q : GaussianFeatureIdx d n) :
    ∫ ω, gaussianHeadFeatureMap X (fun r => headEntries Vr Wr Wr' r ω) p *
      gaussianHeadFeatureMap X (fun r => headEntries Vr Wr Wr' r ω) q ∂P =
      gaussianFeatureCovariance σA X p q := by
  apply integral_gaussianHeadFeature_product (headEntries Vr Wr Wr') σV σA
    (measurable_headEntries hV hW hW') hind
  intro r
  unfold headEntries
  obtain ⟨a, i, j⟩ := r
  fin_cases a
  · simpa using hmV i j
  · simpa using hmW i j
  · simpa using hmW' i j

/-- The nondegenerate canonical original entry law supplies every
measurability, independence and actual marginal-law hypothesis.
Orthogonal unit tokens provide the source's unit diagonal. -/
example : ∃ (P : Measure (GaussIdx 2 → ℝ)) (X : Idx 2 → EucSpace 2),
    IsProbabilityMeasure P ∧
    (∀ p : GaussIdx 2, Measurable (fun z : GaussIdx 2 → ℝ => z p)) ∧
    iIndepFun (fun p : GaussIdx 2 => fun z : GaussIdx 2 → ℝ => z p) P ∧
    (∀ p : GaussIdx 2, P.map (fun z : GaussIdx 2 → ℝ => z p) =
      gaussianReal 0 ((if p.1 = 0 then (1 : ℝ≥0) else 1) ^ 2)) ∧
    (∀ i, ‖X i‖ = 1) := by
  let μ (_ : GaussIdx 2) := gaussianReal 0 ((1 : ℝ≥0) ^ 2)
  refine ⟨Measure.pi μ, fun i => EuclideanSpace.single i 1, inferInstance,
    measurable_pi_apply,
    iIndepFun_pi (X := fun _ : GaussIdx 2 => id) (fun _ => measurable_id.aemeasurable),
    fun p => ?_, fun i => by simp⟩
  simpa only [ite_self] using (measurePreserving_eval μ p).map_eq

/-- The original source head law realizes every original matrix
premise, with positive variance in both query/key blocks. -/
example : IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) :=
  isGaussianHeadLaw_gaussHeadLaw 2 1 1

end Transformer.Homogenized
