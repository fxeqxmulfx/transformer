module

public import Transformer.Homogenized.Section7_GaussianCoordinateMoments
public import Mathlib.MeasureTheory.Integral.Prod

/-!
# Genuine Gaussian row laws and bilinear query moments

ArXiv:2604.01978v1, §2.3.3, assumption (G),
`eq: tformers.at.initialization`, and §7.3, `lemma:Laplace_method`.
An original query coordinate of `A = WW′ᵀ` is the dot product of
one Gaussian row of `W` with one independent Gaussian row of `W′`.
The row laws below are the actual finite products of standard Gaussians.
In particular, the dot product itself is not assigned a Gaussian law.

Each row-coordinate product is integrable and has its exact diagonal
second moment. Expanding the square of the original dot product and
using the product law gives its second moment exactly equal to the
number of row coordinates. All finite sums and products are proved
integrable before their integrals are distributed.

Conditionally on one fixed row, the dot product is a genuine Gaussian
linear form with variance equal to that row's squared Euclidean norm.
This includes zero rows and empty dimensions; no inverse covariance
or nonzero-variance assumption enters the conditional law.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual independent standard Gaussian entries of one row.
Source: arXiv:2604.01978v1, §2.3.3, initialization (G). -/
noncomputable def gaussianRowLaw (d : ℕ) : Measure (Fin d → ℝ) :=
  Measure.pi (fun _ => gaussianReal 0 1)

instance (d : ℕ) : IsProbabilityMeasure (gaussianRowLaw d) := by
  unfold gaussianRowLaw
  infer_instance

/-- Every actual row coordinate has its prescribed standard law.
Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem gaussianRowLaw_map_coordinate {d : ℕ} (i : Fin d) :
    (gaussianRowLaw d).map (fun z => z i) = gaussianReal 0 1 :=
  (measurePreserving_eval (fun _ : Fin d => gaussianReal 0 1) i).map_eq

/-- Original row-coordinate products are genuinely integrable.
Source: arXiv:2604.01978v1, §7.3, Gaussian query moments. -/
theorem integrable_gaussianRow_coordinate_product {d : ℕ} (i j : Fin d) :
    Integrable (fun z : Fin d → ℝ => z i * z j) (gaussianRowLaw d) := by
  exact integrable_gaussian_coordinate_product (P := gaussianRowLaw d)
    (fun (i : Fin d) (z : Fin d → ℝ) => z i) (fun _ => (1 : ℝ≥0))
    measurable_pi_apply (fun i => by simpa using gaussianRowLaw_map_coordinate i) i j

/-- The actual row second moments have unit diagonal and zero
off-diagonal entries. Source: arXiv:2604.01978v1, initialization (G). -/
theorem integral_gaussianRow_coordinate_product {d : ℕ} (i j : Fin d) :
    (∫ z : Fin d → ℝ, z i * z j ∂gaussianRowLaw d) = if i = j then 1 else 0 := by
  simpa only [NNReal.coe_one, one_pow] using integral_gaussian_coordinate_product (P := gaussianRowLaw d)
    (fun (i : Fin d) (z : Fin d → ℝ) => z i) (fun _ => (1 : ℝ≥0))
    measurable_pi_apply
    (iIndepFun_pi (X := fun _ : Fin d => id) (fun _ => aemeasurable_id))
    (fun i => by simpa using gaussianRowLaw_map_coordinate i) i j

/-- The actual bilinear query coordinate has a finite second moment.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`, Step 2. -/
theorem integrable_gaussianRow_pair_dot_sq (d : ℕ) :
    Integrable (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2 ^ 2)
      ((gaussianRowLaw d).prod (gaussianRowLaw d)) := by
  have he : ∀ p : (Fin d → ℝ) × (Fin d → ℝ),
      dotProduct p.1 p.2 ^ 2 = ∑ i : Fin d, ∑ j : Fin d,
        (p.1 i * p.1 j) * (p.2 i * p.2 j) := by
    intro p
    rw [dotProduct, pow_two, Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by ring
  simp_rw [he]
  exact integrable_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    (integrable_gaussianRow_coordinate_product i j).mul_prod
      (integrable_gaussianRow_coordinate_product i j)

/-- A dot product of two independent original standard rows has
second moment `d`. Source: arXiv:2604.01978v1, §7.3,
the original product-matrix query in `lemma:Laplace_method`. -/
theorem integral_gaussianRow_pair_dot_sq (d : ℕ) :
    (∫ p : (Fin d → ℝ) × (Fin d → ℝ), dotProduct p.1 p.2 ^ 2
      ∂(gaussianRowLaw d).prod (gaussianRowLaw d)) = (d : ℝ) := by
  have he : ∀ p : (Fin d → ℝ) × (Fin d → ℝ),
      dotProduct p.1 p.2 ^ 2 = ∑ i : Fin d, ∑ j : Fin d,
        (p.1 i * p.1 j) * (p.2 i * p.2 j) := by
    intro p
    rw [dotProduct, pow_two, Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by ring
  simp_rw [he]
  rw [integral_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    (integrable_gaussianRow_coordinate_product i j).mul_prod
      (integrable_gaussianRow_coordinate_product i j)]
  have hi : ∀ i : Fin d,
      (∫ p : (Fin d → ℝ) × (Fin d → ℝ),
        ∑ j : Fin d, (p.1 i * p.1 j) * (p.2 i * p.2 j)
        ∂(gaussianRowLaw d).prod (gaussianRowLaw d)) = 1 := by
    intro i
    rw [integral_finsetSum _ fun j hj =>
      (integrable_gaussianRow_coordinate_product i j).mul_prod
        (integrable_gaussianRow_coordinate_product i j)]
    have hp (j : Fin d) : (∫ p : (Fin d → ℝ) × (Fin d → ℝ),
        (p.1 i * p.1 j) * (p.2 i * p.2 j) ∂(gaussianRowLaw d).prod (gaussianRowLaw d)) =
        (if i = j then (1 : ℝ) else 0) * (if i = j then (1 : ℝ) else 0) := by
      rw [integral_prod_mul (μ := gaussianRowLaw d) (ν := gaussianRowLaw d)
        (fun z : Fin d → ℝ => z i * z j) (fun z : Fin d → ℝ => z i * z j),
        integral_gaussianRow_coordinate_product]
    simp_rw [hp]
    simp
  simp only [hi, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]

/-- The actual squared Euclidean norm of one standard row has mean
`d`. Source: arXiv:2604.01978v1, initialization (G). -/
theorem integral_gaussianRow_norm_sq (d : ℕ) :
    (∫ z : Fin d → ℝ, ∑ i : Fin d, z i ^ 2 ∂gaussianRowLaw d) = d := by
  rw [integral_finsetSum _ fun i hi => by
    simpa only [pow_two] using integrable_gaussianRow_coordinate_product i i]
  simp only [pow_two, integral_gaussianRow_coordinate_product, ite_true,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]

/-- Given a fixed first row, the original dot product has its exact
conditional scalar Gaussian law. Source: arXiv:2604.01978v1,
§7.3, `lemma:Laplace_method`, with the actual product initialization
retained rather than treating its product matrix as Gaussian. -/
theorem gaussianRow_linear_hasLaw {d : ℕ} (g : Fin d → ℝ) :
    HasLaw (fun z : Fin d → ℝ => dotProduct g z)
      (gaussianReal 0 (∑ i : Fin d, g i ^ 2).toNNReal) (gaussianRowLaw d) := by
  let L : (Fin d → ℝ) →L[ℝ] ℝ := ∑ i : Fin d, g i • ContinuousLinearMap.proj i
  have hL : ∀ z : Fin d → ℝ, L z = dotProduct g z := by
    intro z
    simp [L, dotProduct]
  have hG := (hasGaussianLaw_pi_coordinate_identity (fun _ : Fin d => (1 : ℝ≥0))).map L
  have hmap := hG.map_eq_gaussianReal
  simp only [Function.comp_def, id_eq, hL, one_pow] at hmap
  change (gaussianRowLaw d).map (fun z => dotProduct g z) =
    gaussianReal (∫ z, dotProduct g z ∂gaussianRowLaw d)
      (variance (fun z => dotProduct g z) (gaussianRowLaw d)).toNNReal at hmap
  have hm : (∫ z : Fin d → ℝ, dotProduct g z ∂gaussianRowLaw d) = 0 := by
    exact integral_gaussian_coordinate_linear_combination (P := gaussianRowLaw d)
      (fun i z => z i) (fun _ => (1 : ℝ≥0)) measurable_pi_apply
      (fun i => by simpa only [one_pow] using gaussianRowLaw_map_coordinate i) g id
  have hs : (∫ z : Fin d → ℝ, dotProduct g z ^ 2 ∂gaussianRowLaw d) = ∑ i : Fin d, g i ^ 2 := by
    have h := integral_gaussian_coordinate_linear_product (P := gaussianRowLaw d)
      (fun (i : Fin d) (z : Fin d → ℝ) => z i) (fun _ => (1 : ℝ≥0)) measurable_pi_apply
      (iIndepFun_pi (X := fun _ : Fin d => id) (fun _ => aemeasurable_id))
      (fun i => by simpa only [one_pow] using gaussianRowLaw_map_coordinate i) g g id id
    simpa [dotProduct, pow_two] using h
  refine ⟨(by fun_prop : Measurable (fun z : Fin d → ℝ => dotProduct g z)).aemeasurable, ?_⟩
  rw [hmap, hm, variance_of_integral_eq_zero (by fun_prop) hm, hs]

/-- Two genuine nondegenerate Gaussian rows have dot-product second
moment two; the fixed row `(1,2)` gives a conditional variance of five.
This exercises the original row law and both moment calculations. -/
example : (∫ p : (Fin 2 → ℝ) × (Fin 2 → ℝ), dotProduct p.1 p.2 ^ 2
      ∂(gaussianRowLaw 2).prod (gaussianRowLaw 2)) = 2 ∧
    HasLaw (fun z : Fin 2 → ℝ => dotProduct ![1, 2] z) (gaussianReal 0 5) (gaussianRowLaw 2) := by
  refine ⟨integral_gaussianRow_pair_dot_sq 2, ?_⟩
  have h := gaussianRow_linear_hasLaw (![1, 2] : Fin 2 → ℝ)
  have hv : (∑ i : Fin 2, (![1, 2] : Fin 2 → ℝ) i ^ 2).toNNReal = 5 := by
    norm_num [Fin.sum_univ_two]
  rw [hv] at h
  exact h

end Transformer.Homogenized
