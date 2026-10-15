/-
# Original Gaussian-product query moments and normalization

ArXiv:2604.01978v1, §2.3.3, `eq: tformers.at.initialization`,
and §7.3, `lemma:Laplace_method`.
Every actual entry of the original attention matrix `A = WW′ᵀ`
is the dot product of two independent original rows. Its genuine
pushforward is the bilinear row law, not a scalar Gaussian law.
The preceding row calculations therefore give actual entry second
moments and almost-sure nonvanishing on the original head probability.

At any original coordinate unit vector, the query has mean squared
norm exactly `d²` under unit entry variances. Each coordinate square
is proved integrable before distributing this vector norm integral.
This is the query scale retained in the dimension-dependent check
of the source's Laplace error estimate.

The query is almost surely nonzero in every dimension admitting the
chosen coordinate. Consequently its actual normalized vector has
norm one, and its expected self-inner-product is exactly one.
No invertibility hypothesis on a random product matrix is needed.
-/

import Transformer.Homogenized.Section7_GaussianRowNonzero

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original head entry has the actual independent-row dot law.
Source: arXiv:2604.01978v1, initialization (G), at unit variances. -/
theorem gaussianHead_entry_map {d : ℕ} (i j : Fin d) :
    (gaussHeadLaw d 1 1).map (fun θ => θ.2 i j) =
      ((gaussianRowLaw d).prod (gaussianRowLaw d)).map
        (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2) := by
  rw [gaussHeadLaw, Measure.map_map (by fun_prop) (measurable_gaussHead d)]
  rw [← originalGaussianRow_pair_map (1 : Fin 3) 2 (by decide) i j,
    Measure.map_map (by dsimp only [dotProduct]; fun_prop)
      ((measurable_originalGaussianRow 1 i).prodMk (measurable_originalGaussianRow 2 j))]
  rfl

/-- Original attention-entry squares have finite integrals.
Source: arXiv:2604.01978v1, §7.3, the Gaussian query estimates. -/
theorem integrable_gaussianHead_entry_sq {d : ℕ} (i j : Fin d) :
    Integrable (fun θ : HeadParam d => θ.2 i j ^ 2) (gaussHeadLaw d 1 1) := by
  have hdot : Measurable (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2) := by
    dsimp only [dotProduct]
    fun_prop
  have hi : Integrable (fun u : ℝ => u ^ 2)
      (((gaussianRowLaw d).prod (gaussianRowLaw d)).map (fun p => dotProduct p.1 p.2)) :=
    (integrable_map_measure (continuous_pow 2).aestronglyMeasurable hdot.aemeasurable).mpr
      (integrable_gaussianRow_pair_dot_sq d)
  rw [← gaussianHead_entry_map i j] at hi
  exact (integrable_map_measure (continuous_pow 2).aestronglyMeasurable
    (by fun_prop : Measurable (fun θ : HeadParam d => θ.2 i j)).aemeasurable).mp hi

/-- The actual product-matrix entry has second moment `d`.
Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`,
used in §7.3, `lemma:Laplace_method`. -/
theorem integral_gaussianHead_entry_sq {d : ℕ} (i j : Fin d) :
    (∫ θ : HeadParam d, θ.2 i j ^ 2 ∂gaussHeadLaw d 1 1) = d := by
  calc
    _ = ∫ u : ℝ, u ^ 2 ∂(gaussHeadLaw d 1 1).map (fun θ => θ.2 i j) :=
      (integral_map (by fun_prop : Measurable (fun θ : HeadParam d => θ.2 i j)).aemeasurable
        (continuous_pow 2).aestronglyMeasurable).symm
    _ = _ := by
      rw [gaussianHead_entry_map, integral_map
        (by dsimp only [dotProduct]; fun_prop : Measurable
          (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2)).aemeasurable
        (continuous_pow 2).aestronglyMeasurable]
      exact integral_gaussianRow_pair_dot_sq d

/-- Original attention entries are almost surely nonzero at these
nondegenerate entry variances. Source: arXiv:2604.01978v1,
§7.3, the actual normalized query in `lemma:Laplace_method`. -/
theorem ae_gaussianHead_entry_ne_zero {d : ℕ} (hd : 0 < d) (i j : Fin d) :
    ∀ᵐ θ ∂gaussHeadLaw d 1 1, θ.2 i j ≠ 0 := by
  have hdot : Measurable (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2) := by
    dsimp only [dotProduct]
    fun_prop
  have hrow := (ae_map_iff hdot.aemeasurable (measurableSet_singleton (0 : ℝ)).compl).mpr
    (ae_gaussianRow_pair_dot_ne_zero hd)
  rw [← gaussianHead_entry_map i j] at hrow
  exact (ae_map_iff (by fun_prop : Measurable (fun θ : HeadParam d => θ.2 i j)).aemeasurable
    (measurableSet_singleton (0 : ℝ)).compl).mp hrow

/-- A query at an original coordinate unit vector reads the actual
matrix column. Source: arXiv:2604.01978v1, `eq: mbetaA`. -/
theorem qkMap_single {d : ℕ} (θ : HeadParam d) (j i : Fin d) :
    qkMap θ (EuclideanSpace.single j (1 : ℝ)) i = θ.2 i j := by
  simp [qkMap, Matrix.toLpLin_apply]

/-- The original unit-coordinate query has a finite second moment.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem integrable_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    Integrable (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw d 1 1) := by
  simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
  exact integrable_finsetSum _ fun i hi => integrable_gaussianHead_entry_sq i j

/-- Its actual squared norm has mean `d²`, retaining both original
Gaussian factors. Source: arXiv:2604.01978v1, §7.3,
`lemma:Laplace_method`, under initialization (G) at unit variances. -/
theorem integral_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2
      ∂gaussHeadLaw d 1 1) = (d : ℝ) ^ 2 := by
  simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
  rw [integral_finsetSum _ fun i hi => integrable_gaussianHead_entry_sq i j]
  simp only [integral_gaussianHead_entry_sq]
  simp [pow_two]

/-- The actual unit-coordinate query is almost surely nonzero.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem ae_gaussianHead_query_ne_zero {d : ℕ} (j : Fin d) :
    ∀ᵐ θ ∂gaussHeadLaw d 1 1, qkMap θ (EuclideanSpace.single j (1 : ℝ)) ≠ 0 := by
  have hd : 0 < d := by have hj := j.isLt; omega
  filter_upwards [ae_gaussianHead_entry_ne_zero hd j j] with θ hθ
  intro he
  have h := congrArg (fun v : EucSpace d => v j) he
  rw [qkMap_single] at h
  exact hθ h

/-- The actual normalized-query self-correlation is exactly one.
Source: arXiv:2604.01978v1, §7.3, the right-hand side of
`lemma:Laplace_method` at identical original query points. -/
theorem integral_gaussianHead_normalized_query_self {d : ℕ} (j : Fin d) :
    (∫ θ : HeadParam d, inner ℝ
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))
      ∂gaussHeadLaw d 1 1) = 1 := by
  have he : (fun θ : HeadParam d => inner ℝ
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))) =ᵐ[gaussHeadLaw d 1 1]
      (fun _ => (1 : ℝ)) := by
    filter_upwards [ae_gaussianHead_query_ne_zero j] with θ hθ
    rw [real_inner_self_eq_norm_sq, norm_normalizeLayer hθ, one_pow]
  rw [integral_congr_ae he]
  simp

/-- The original nondegenerate Gaussian head, positive dimension
and actual coordinate unit query jointly realize the hypotheses. Its
expected squared query norm is four in dimension two. -/
example : (0 : ℕ) < 2 ∧ IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧
    ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 ∧
    (∫ θ : HeadParam 2, ‖qkMap θ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))‖ ^ 2
      ∂gaussHeadLaw 2 1 1) = 4 := by
  refine ⟨by decide, isGaussianHeadLaw_gaussHeadLaw 2 1 1, by simp, ?_⟩
  norm_num only [integral_gaussianHead_query_norm_sq]

end Transformer.Homogenized
