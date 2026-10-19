module

public import Transformer.Homogenized.Section7_GaussianCoordinateLaw

/-!
# Actual independent Gaussian coordinate moments and their linear images

ArXiv:2604.01978v1, §7, the covariance computation in `eq:SIGMA_R`,
under the original independently Gaussian initialization (G).
The manuscript takes the covariance of linear query/key features from
their independent entries. This module proves that finite calculation
on the original probability space, with the actual marginal laws.

Distinct original entries have zero mixed second moment. Each diagonal
second moment is its original variance. Products and finite linear
combinations are proved integrable before distributing their integrals.
The two linear combinations may use different index sets and repeat
entry indices; consequently the result also covers cross-block features.
No covariance identity is built into the definitions or supplied as a
hypothesis. Zero variances and empty finite sums are included.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

variable {Ω ι : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- Actual products of original Gaussian coordinates are integrable.
Source: arXiv:2604.01978v1, §7, the second moments in `eq:SIGMA_R`. -/
theorem integrable_gaussian_coordinate_product (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0)
    (hZ : ∀ i, Measurable (Z i))
    (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2)) (i j : ι) :
    Integrable (fun ω => Z i ω * Z j ω) P :=
  (memLp_two_of_map_eq_gaussianReal (hZ i) (hmap i)).integrable_mul
    (memLp_two_of_map_eq_gaussianReal (hZ j) (hmap j))

/-- The original independent centered entries have their actual
diagonal second moments and zero off-diagonal moments.
Source: arXiv:2604.01978v1, §7, the entries underlying `eq:SIGMA_R`. -/
theorem integral_gaussian_coordinate_product [DecidableEq ι]
    (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0) (hZ : ∀ i, Measurable (Z i))
    (hind : iIndepFun Z P) (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2))
    (i j : ι) :
    ∫ ω, Z i ω * Z j ω ∂P = if i = j then (σ i : ℝ) ^ 2 else 0 := by
  by_cases hij : i = j
  · subst j
    simp only [ite_true]
    simpa only [← sq, NNReal.coe_pow] using
      integral_sq_of_map_eq_gaussianReal (hZ i) (hmap i)
  · simp only [hij, ite_false]
    rw [(hind.indepFun hij).integral_fun_mul_eq_mul_integral
      (hZ i).aestronglyMeasurable (hZ j).aestronglyMeasurable,
      integral_eq_zero_of_map_eq_gaussianReal (hZ i) (hmap i), zero_mul]

/-- A finite linear combination of original Gaussian coordinates
is integrable on its original finite probability space.
Source: arXiv:2604.01978v1, §7, the original features in `eq:SIGMA_R`. -/
theorem integrable_gaussian_coordinate_linear_combination [IsFiniteMeasure P]
    {κ : Type*} [Fintype κ] (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0)
    (hZ : ∀ i, Measurable (Z i))
    (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2))
    (a : κ → ℝ) (u : κ → ι) :
    Integrable (fun ω => ∑ r, a r * Z (u r) ω) P := by
  exact integrable_finsetSum _ fun r hr =>
    ((memLp_two_of_map_eq_gaussianReal (hZ (u r)) (hmap (u r))).integrable
      (by norm_num)).const_mul (a r)

/-- Every actual finite linear combination is centered.
Source: arXiv:2604.01978v1, §7, the zero mean in `eq:SIGMA_R`. -/
theorem integral_gaussian_coordinate_linear_combination [IsFiniteMeasure P]
    {κ : Type*} [Fintype κ] (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0)
    (hZ : ∀ i, Measurable (Z i))
    (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2))
    (a : κ → ℝ) (u : κ → ι) : ∫ ω, ∑ r, a r * Z (u r) ω ∂P = 0 := by
  rw [integral_finsetSum _ fun r hr =>
    ((memLp_two_of_map_eq_gaussianReal (hZ (u r)) (hmap (u r))).integrable
      (by norm_num)).const_mul (a r)]
  simp only [integral_const_mul,
    integral_eq_zero_of_map_eq_gaussianReal (hZ _) (hmap _), mul_zero, Finset.sum_const_zero]

/-- The product of two actual finite linear combinations is integrable,
including combinations drawn from different original entry blocks.
Source: arXiv:2604.01978v1, §7, all blocks of `eq:SIGMA_R`. -/
theorem integrable_gaussian_coordinate_linear_product {κ κ' : Type*} [Fintype κ] [Fintype κ']
    (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0) (hZ : ∀ i, Measurable (Z i))
    (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2))
    (a : κ → ℝ) (b : κ' → ℝ) (u : κ → ι) (v : κ' → ι) :
    Integrable (fun ω => (∑ r, a r * Z (u r) ω) * (∑ s, b s * Z (v s) ω)) P := by
  have hexp : ∀ ω, (∑ r, a r * Z (u r) ω) * (∑ s, b s * Z (v s) ω) =
      ∑ r, ∑ s, (a r * b s) * (Z (u r) ω * Z (v s) ω) := by
    intro ω
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun r hr => Finset.sum_congr rfl fun s hs => by ring
  simp_rw [hexp]
  exact integrable_finsetSum _ fun r hr => integrable_finsetSum _ fun s hs =>
    (integrable_gaussian_coordinate_product Z σ hZ hmap (u r) (v s)).const_mul (a r * b s)

/-- The actual second moment of two original linear combinations,
computed from the original marginal variances and independence.
Source: arXiv:2604.01978v1, §7, the linear calculation behind `eq:SIGMA_R`.
Repeated indices are retained, so this also computes shared-entry terms. -/
theorem integral_gaussian_coordinate_linear_product [DecidableEq ι]
    {κ κ' : Type*} [Fintype κ] [Fintype κ']
    (Z : ι → Ω → ℝ) (σ : ι → ℝ≥0) (hZ : ∀ i, Measurable (Z i))
    (hind : iIndepFun Z P) (hmap : ∀ i, P.map (Z i) = gaussianReal 0 (σ i ^ 2))
    (a : κ → ℝ) (b : κ' → ℝ) (u : κ → ι) (v : κ' → ι) :
    ∫ ω, (∑ r, a r * Z (u r) ω) * (∑ s, b s * Z (v s) ω) ∂P =
      ∑ r, ∑ s, (a r * b s) * (if u r = v s then (σ (u r) : ℝ) ^ 2 else 0) := by
  have hterm : ∀ r s, Integrable
      (fun ω => (a r * b s) * (Z (u r) ω * Z (v s) ω)) P :=
    fun r s => (integrable_gaussian_coordinate_product Z σ hZ hmap (u r) (v s)).const_mul _
  have hexp : ∀ ω, (∑ r, a r * Z (u r) ω) * (∑ s, b s * Z (v s) ω) =
      ∑ r, ∑ s, (a r * b s) * (Z (u r) ω * Z (v s) ω) := by
    intro ω
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun r hr => Finset.sum_congr rfl fun s hs => by ring
  simp_rw [hexp]
  rw [integral_finsetSum _ fun r hr => integrable_finsetSum _ fun s hs => hterm r s]
  apply Finset.sum_congr rfl
  intro r hr
  rw [integral_finsetSum _ fun s hs => hterm r s]
  simp only [integral_const_mul, integral_gaussian_coordinate_product Z σ hZ hind hmap]

/-- A nondegenerate original product law supplies every actual
measurability, independence, marginal-law and finite-measure premise. -/
example : ∃ P : Measure (Fin 2 → ℝ), IsProbabilityMeasure P ∧
    (∀ i : Fin 2, Measurable (fun z : Fin 2 → ℝ => z i)) ∧
    iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i) P ∧
    (∀ i : Fin 2, P.map (fun z : Fin 2 → ℝ => z i) = gaussianReal 0 ((1 : ℝ≥0) ^ 2)) := by
  let μ (_ : Fin 2) := gaussianReal 0 ((1 : ℝ≥0) ^ 2)
  refine ⟨Measure.pi μ, inferInstance, measurable_pi_apply,
    iIndepFun_pi (X := fun _ : Fin 2 => id) (fun _ => measurable_id.aemeasurable), ?_⟩
  exact fun i => (measurePreserving_eval μ i).map_eq

/-- Two different nonconstant linear combinations of standard
Gaussian entries have the nonzero mixed second moment `11`.
This example also instantiates both finite coefficient families. -/
example : ∫ z : Fin 2 → ℝ, (z 0 + 2 * z 1) * (3 * z 0 + 4 * z 1)
    ∂(Measure.pi fun _ : Fin 2 => gaussianReal 0 1) = 11 := by
  let μ (_ : Fin 2) := gaussianReal 0 1
  have h := integral_gaussian_coordinate_linear_product
    (P := Measure.pi μ) (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i)
    (fun _ => (1 : ℝ≥0)) measurable_pi_apply
    (iIndepFun_pi (X := fun _ : Fin 2 => id) (fun _ => measurable_id.aemeasurable))
    (fun i => by simpa only [one_pow] using (measurePreserving_eval μ i).map_eq)
    ![1, 2] ![3, 4] id id
  norm_num [Fin.sum_univ_two] at h
  exact h

end Transformer.Homogenized
