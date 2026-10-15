/-
# The original matrix factor and an independent original key row

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The query matrix is the product `WW′ᵀ`.
Its action on a coordinate vector is the first Gaussian matrix
applied to one independent row of the second Gaussian matrix.

This module derives that joint law from the actual original entry
ensemble. Currying the finite product of scalar Gaussian entries
gives the finite product of Gaussian row laws. Independence of a
whole matrix block and a row from a distinct original block follows
by selecting disjoint sets of original independent entries.

The resulting conditional row linear forms have their exact scalar
Gaussian laws. Linear forms in distinct matrix rows are independent
conditionally on the fixed key row. Neither the product matrix nor
its bilinear coordinates are assigned a Gaussian law.
-/

import Transformer.Homogenized.Section7_GaussianRowNonzero
import Mathlib.Probability.ProductMeasure

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual standard Gaussian matrix, as independent original rows.
Source: arXiv:2604.01978v1, initialization (G). -/
noncomputable def gaussianMatrixLaw (d : ℕ) : Measure (Matrix (Fin d) (Fin d) ℝ) :=
  (Measure.pi (fun _ : Fin d => gaussianRowLaw d)).map
    (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ)

instance (d : ℕ) : IsProbabilityMeasure (gaussianMatrixLaw d) := by
  unfold gaussianMatrixLaw
  infer_instance

/-- An original standard Gaussian matrix block has its actual row
product law. Source: arXiv:2604.01978v1, initialization (G). -/
theorem originalGaussianMatrix_map {d : ℕ} (b : Fin 3) :
    (gaussEnsemble d 1 1).map (gaussBlock b) = gaussianMatrixLaw d := by
  let f : (GaussIdx d → ℝ) → (Fin d × Fin d → ℝ) := fun z p => z (b, p.1, p.2)
  have hm : Measurable f := Measurable.of_eval fun p => measurable_pi_apply _
  have hind : iIndepFun (fun p : GaussIdx d => fun z : GaussIdx d → ℝ => z p)
      (gaussEnsemble d 1 1) :=
    iIndepFun_pi (X := fun _ : GaussIdx d => id) (fun _ => aemeasurable_id)
  have hinj : Function.Injective (fun p : Fin d × Fin d => ((b, p.1, p.2) : GaussIdx d)) := by
    intro p q h
    exact congrArg (fun p : GaussIdx d => p.2) h
  have hflat := (hind.precomp hinj).map_fun_eq_pi_map
    (fun p => (measurable_pi_apply (b, p.1, p.2)).aemeasurable)
  change (gaussEnsemble d 1 1).map f = _ at hflat
  have hc (p : Fin d × Fin d) :
      (gaussEnsemble d 1 1).map (fun z => z (b, p.1, p.2)) = gaussianReal 0 1 := by
    simpa only [gaussBlock_apply, entryVar, one_pow, ite_self] using
      map_gaussBlock (d := d) 1 1 b p.1 p.2
  simp only [hc] at hflat
  have hcur : (Measure.pi (fun _ : Fin d × Fin d => gaussianReal 0 1)).map
      (MeasurableEquiv.curry (Fin d) (Fin d) ℝ) = Measure.pi (fun _ : Fin d => gaussianRowLaw d) := by
    simpa only [Measure.infinitePi_eq_pi, gaussianRowLaw] using
      Measure.infinitePi_map_curry (fun _ : Fin d => fun _ : Fin d => gaussianReal 0 1)
  unfold gaussianMatrixLaw
  rw [← hcur, ← hflat,
    Measure.map_map (MeasurableEquiv.curry (Fin d) (Fin d) ℝ).measurable hm,
    Measure.map_map (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ).measurable
      ((MeasurableEquiv.curry (Fin d) (Fin d) ℝ).measurable.comp hm)]
  rfl

/-- An actual matrix factor is independent of an actual key row
from a distinct original block. Source: arXiv:2604.01978v1,
initialization (G), retaining the two original Gaussian factors. -/
theorem indepFun_originalGaussianMatrix_row {d : ℕ} (b c : Fin 3)
    (hbc : b ≠ c) (j : Fin d) :
    IndepFun (gaussBlock b) (originalGaussianRow c j) (gaussEnsemble d 1 1) := by
  classical
  let S := Finset.univ.filter (fun p : GaussIdx d => p.1 = b)
  let T := Finset.univ.filter (fun p : GaussIdx d => p.1 = c ∧ p.2.1 = j)
  have hST : Disjoint S T := Finset.disjoint_filter.mpr
    (fun p hp hb hc => hbc (hb.symm.trans hc.1))
  have hind : iIndepFun (fun p : GaussIdx d => fun z : GaussIdx d → ℝ => z p)
      (gaussEnsemble d 1 1) :=
    iIndepFun_pi (X := fun _ : GaussIdx d => id) (fun _ => aemeasurable_id)
  let f : (S → ℝ) → Matrix (Fin d) (Fin d) ℝ := fun z i q => z ⟨(b, i, q), by simp [S]⟩
  let g : (T → ℝ) → Fin d → ℝ := fun z q => z ⟨(c, j, q), by simp [T]⟩
  exact (hind.indepFun_finset S T hST (fun p => measurable_pi_apply p)).comp
    (by fun_prop : Measurable f) (by fun_prop : Measurable g)

/-- The key row and the first original matrix have the actual
independent product law. Source: arXiv:2604.01978v1, §7.3,
the product-matrix query used in `lem:delta_method`. -/
theorem originalGaussianRow_matrix_pair_map {d : ℕ} (b c : Fin 3)
    (hbc : b ≠ c) (j : Fin d) :
    (gaussEnsemble d 1 1).map (fun z => (originalGaussianRow c j z, gaussBlock b z)) =
      (gaussianRowLaw d).prod (gaussianMatrixLaw d) := by
  rw [(indepFun_originalGaussianMatrix_row b c hbc j).symm.map_prod_eq_prod_map_map
    (measurable_originalGaussianRow c j).aemeasurable (measurable_gaussBlock b).aemeasurable,
    originalGaussianRow_map, originalGaussianMatrix_map]

/-- Each actual matrix row has the prescribed original row law.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem gaussianMatrixLaw_map_row {d : ℕ} (i : Fin d) :
    (gaussianMatrixLaw d).map (fun M => M i) = gaussianRowLaw d := by
  have hm : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => M i) :=
    Measurable.of_eval fun q => Matrix.measurable_apply (i := i) (j := q)
  rw [gaussianMatrixLaw, Measure.map_map hm (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ).measurable]
  exact (measurePreserving_eval (fun _ : Fin d => gaussianRowLaw d) i).map_eq

/-- With a fixed key row, an original matrix-row query is exactly
Gaussian with its actual conditional variance. Source:
arXiv:2604.01978v1, §7.3, moments for `lem:delta_method`. -/
theorem gaussianMatrix_row_linear_hasLaw {d : ℕ} (g : Fin d → ℝ) (i : Fin d) :
    HasLaw (fun M : Matrix (Fin d) (Fin d) ℝ => dotProduct g (M i))
      (gaussianReal 0 (∑ k : Fin d, g k ^ 2).toNNReal) (gaussianMatrixLaw d) := by
  refine ⟨(by fun_prop : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ =>
    dotProduct g (M i))).aemeasurable, ?_⟩
  have hm : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => M i) :=
    Measurable.of_eval fun q => Matrix.measurable_apply (i := i) (j := q)
  rw [← (gaussianRow_linear_hasLaw g).map_eq, ← gaussianMatrixLaw_map_row i,
    Measure.map_map (by fun_prop : Measurable (fun z : Fin d → ℝ => dotProduct g z)) hm]
  rfl

/-- Conditional queries from distinct actual matrix rows are
independent. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem indepFun_gaussianMatrix_row_linear {d : ℕ} (g : Fin d → ℝ) (i j : Fin d)
    (hij : i ≠ j) :
    IndepFun (fun M : Matrix (Fin d) (Fin d) ℝ => dotProduct g (M i))
      (fun M => dotProduct g (M j)) (gaussianMatrixLaw d) := by
  have hind : iIndepFun (fun i : Fin d => fun M : Matrix (Fin d) (Fin d) ℝ => M i)
      (gaussianMatrixLaw d) := by
    apply (iIndepFun_iff_map_fun_eq_pi_map (fun i =>
      (Measurable.of_eval fun q => Matrix.measurable_apply (i := i) (j := q)).aemeasurable)).mpr
    simp_rw [gaussianMatrixLaw_map_row]
    change ((Measure.pi (fun _ : Fin d => gaussianRowLaw d)).map
      (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ)).map
        (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ).symm = _
    exact (Matrix.ofMeasurableEquiv (Fin d) (Fin d) ℝ).map_symm_map
  exact (hind.indepFun hij).comp
    (by fun_prop : Measurable (fun z : Fin d → ℝ => dotProduct g z))
    (by fun_prop : Measurable (fun z : Fin d → ℝ => dotProduct g z))

/-- Distinct original blocks, an actual key-row index and distinct
conditional matrix-row indices occur jointly in dimension two. -/
example : (1 : Fin 3) ≠ 2 ∧ (0 : Fin 2) ≠ 1 ∧
    (gaussEnsemble 2 1 1).map
      (fun z => (originalGaussianRow 2 0 z, gaussBlock 1 z)) =
        (gaussianRowLaw 2).prod (gaussianMatrixLaw 2) := by
  exact ⟨by decide, by decide, originalGaussianRow_matrix_pair_map 1 2 (by decide) 0⟩

end Transformer.Homogenized
