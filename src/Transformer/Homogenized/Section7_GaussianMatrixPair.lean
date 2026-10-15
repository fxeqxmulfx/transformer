/-
# The actual two-factor Gaussian law of the original head matrix

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The original head uses `A=WW′ᵀ`. The two
actual factors `W′ᵀ` and `W` have the independent standard Gaussian
matrix laws used in the corrected product-correlation expansion.

Transposing all original matrix blocks permutes the original scalar
entries and preserves their genuine independent standard Gaussian
product measure. The transposed second original factor therefore
has exactly the same actual matrix law as an untransposed factor.
Whole original matrix blocks are independent because their entries
belong to disjoint original blocks, not because their product is Gaussian.

The original head expectation is then identified with the genuine
product-law expectation at the original unit Gaussian entry scale.
This identity holds for arbitrary deterministic input queries, including
zero vectors, and keeps the original transpose in the head definition.
Arbitrary original positive query scales are handled by positive scaling.
-/

import Transformer.Homogenized.Section7_GaussianProductDelta

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual independent standard entry ensemble is invariant
under transposing each original matrix block.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem gaussEnsemble_map_transpose_blocks (d : ℕ) :
    (gaussEnsemble d 1 1).map
      (fun z : GaussIdx d → ℝ => fun p : GaussIdx d => z (p.1, p.2.2, p.2.1)) =
        gaussEnsemble d 1 1 := by
  let e : GaussIdx d ≃ GaussIdx d :=
    Equiv.prodCongr (Equiv.refl (Fin 3)) (Equiv.prodComm (Fin d) (Fin d))
  have h := (measurePreserving_piCongrLeft (μ := fun _ : GaussIdx d => gaussianReal 0 1) e).map_eq
  have he : (MeasurableEquiv.piCongrLeft (fun _ : GaussIdx d => ℝ) e :
      (GaussIdx d → ℝ) → GaussIdx d → ℝ) = fun z p => z (p.1, p.2.2, p.2.1) := by
    funext z p
    rcases p with ⟨b, i, j⟩
    simp [MeasurableEquiv.coe_piCongrLeft, Equiv.piCongrLeft_apply_eq_cast, e]
  rw [he] at h
  simpa only [gaussEnsemble, entryVar, one_pow, ite_self] using h

/-- Every actual transposed original standard Gaussian block has
the original standard Gaussian matrix law.
Source: arXiv:2604.01978v1, initialization (G),
the transpose in `A=WW′ᵀ` used by §7.3, `lem:delta_method`. -/
theorem originalGaussianMatrix_transpose_map {d : ℕ} (b : Fin 3) :
    (gaussEnsemble d 1 1).map (fun z => (gaussBlock b z).transpose) = gaussianMatrixLaw d := by
  have hm : Measurable (fun z : GaussIdx d → ℝ => fun p : GaussIdx d => z (p.1, p.2.2, p.2.1)) := by fun_prop
  calc
    _ = ((gaussEnsemble d 1 1).map
        (fun z : GaussIdx d → ℝ => fun p : GaussIdx d => z (p.1, p.2.2, p.2.1))).map (gaussBlock b) := by
      rw [Measure.map_map (measurable_gaussBlock b) hm]
      rfl
    _ = (gaussEnsemble d 1 1).map (gaussBlock b) := by rw [gaussEnsemble_map_transpose_blocks]
    _ = _ := originalGaussianMatrix_map b

/-- Two actual whole original matrix blocks are independent when
their original block indices differ. Source: arXiv:2604.01978v1,
initialization (G), retaining the two distinct original factors. -/
theorem indepFun_originalGaussianMatrices {d : ℕ} (b c : Fin 3) (hbc : b ≠ c) :
    IndepFun (gaussBlock b) (gaussBlock c) (gaussEnsemble d 1 1) := by
  classical
  let S := Finset.univ.filter (fun p : GaussIdx d => p.1 = b)
  let T := Finset.univ.filter (fun p : GaussIdx d => p.1 = c)
  have hST : Disjoint S T := Finset.disjoint_filter.mpr
    (fun p hp hb hc => hbc (hb.symm.trans hc))
  have hind : iIndepFun (fun p : GaussIdx d => fun z : GaussIdx d → ℝ => z p)
      (gaussEnsemble d 1 1) :=
    iIndepFun_pi (X := fun _ : GaussIdx d => id) (fun _ => aemeasurable_id)
  let f : (S → ℝ) → Matrix (Fin d) (Fin d) ℝ := fun z i j => z ⟨(b, i, j), by simp [S]⟩
  let g : (T → ℝ) → Matrix (Fin d) (Fin d) ℝ := fun z i j => z ⟨(c, i, j), by simp [T]⟩
  exact (hind.indepFun_finset S T hST (fun p => measurable_pi_apply p)).comp
    (by fun_prop : Measurable f) (by fun_prop : Measurable g)

/-- The original second factor's transpose and the original first
factor have exactly the independent product law used in the expansion.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem originalGaussianMatrix_transpose_pair_map {d : ℕ} (b c : Fin 3) (hbc : b ≠ c) :
    (gaussEnsemble d 1 1).map (fun z => ((gaussBlock c z).transpose, gaussBlock b z)) =
      (gaussianMatrixLaw d).prod (gaussianMatrixLaw d) := by
  have ht : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => M.transpose) := by fun_prop
  have hind := (indepFun_originalGaussianMatrices b c hbc).symm.comp ht measurable_id
  have h := hind.map_prod_eq_prod_map_map (ht.comp (measurable_gaussBlock c)).aemeasurable
    (measurable_gaussBlock b).aemeasurable
  simp only [Function.comp_def] at h
  rw [originalGaussianMatrix_transpose_map, originalGaussianMatrix_map] at h
  exact h

/-- The original Euclidean action of a matrix product is the actual
composition of the two original actions. Source: arXiv:2604.01978v1,
§7.3, the original product query in `lem:delta_method`. -/
theorem gaussianMatrix_product_query {d : ℕ}
    (M N : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    Matrix.toEuclideanLin (M * N) x = Matrix.toEuclideanLin M (Matrix.toEuclideanLin N x) := by
  simp only [Matrix.toEuclideanLin, Matrix.toLpLin_mul 2 2 2, LinearMap.comp_apply]

/-- **The original standard head has the genuine independent product
correlation expectation.** The actual transpose in `WW′ᵀ` is retained.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem integral_gaussHead_unit_product_correlation {d : ℕ} (x y : EucSpace d) :
    (∫ θ : HeadParam d,
      inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂gaussHeadLaw d 1 1) =
      ∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
        inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
          (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y)))
            ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d) := by
  have hhead : Measurable (fun θ : HeadParam d =>
      inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y))) := by
    exact measurable_normalized_inner _ _
      ((measurable_gaussianMatrix_query x).comp measurable_snd)
      ((measurable_gaussianMatrix_query y).comp measurable_snd)
  have hpair : Measurable (fun z : GaussIdx d → ℝ => ((gaussBlock (2 : Fin 3) z).transpose, gaussBlock (1 : Fin 3) z)) := by
    have ht : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => M.transpose) := by fun_prop
    exact (ht.comp (measurable_gaussBlock 2)).prodMk (measurable_gaussBlock 1)
  have hcorr := measurable_normalized_inner _ _
    (measurable_gaussian_product_query x) (measurable_gaussian_product_query y)
  rw [gaussHeadLaw, integral_map (measurable_gaussHead d).aemeasurable hhead.aestronglyMeasurable,
    ← originalGaussianMatrix_transpose_pair_map (d := d) 1 2 (by decide),
    integral_map hpair.aemeasurable hcorr.aestronglyMeasurable]
  apply integral_congr_ae
  filter_upwards with z
  change inner ℝ (normalizeLayer (Matrix.toEuclideanLin (gaussBlock 1 z * (gaussBlock 2 z).transpose) x))
    (normalizeLayer (Matrix.toEuclideanLin (gaussBlock 1 z * (gaussBlock 2 z).transpose) y)) = _
  rw [gaussianMatrix_product_query, gaussianMatrix_product_query]

/-- **Corrected uniform expansion for the original head at unit entry scale.**
This retains the source's actual `WW′ᵀ` product and all original unit queries.
The source gives correction `(r³-r)/(2d)`; the actual correction is `(r³-r)/d`.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem gaussHead_unit_delta_method :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      |(∫ θ : HeadParam d,
          inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂gaussHeadLaw d 1 1) -
            (inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (d : ℝ))| ≤
              C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussianMatrix_product_delta_method
  refine ⟨C, hC, ?_⟩
  intro d hd x y hx hy
  rw [integral_gaussHead_unit_product_correlation]
  exact h d hd x y hx hy

/-- Distinct actual original block indices satisfy the whole-block
independence, transpose-product-law and unit-input hypotheses jointly. -/
example : (1 : Fin 3) ≠ 2 ∧ (0 : ℕ) < 2 ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧ ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 ∧
    (gaussEnsemble 2 1 1).map (fun z => ((gaussBlock (2 : Fin 3) z).transpose, gaussBlock (1 : Fin 3) z)) =
      (gaussianMatrixLaw 2).prod (gaussianMatrixLaw 2) := by
  exact ⟨by decide, by decide, by simp, by simp, originalGaussianMatrix_transpose_pair_map 1 2 (by decide)⟩

end Transformer.Homogenized
