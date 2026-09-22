/-
# Homogenized Transformers — the Gaussian head law is rotation invariant

Under assumption (G) of arXiv:2604.01978v1 the head is `(V, A) = (V, W W'ᵀ)` with
`V, W, W'` independent i.i.d. Gaussian matrices.  For an orthogonal `O`,
`Oᵀ A O = (Oᵀ W)(Oᵀ W')ᵀ` (`conjMat_mul_transpose`), and rotating the columns of
`W` and `W'` is an isometry of the i.i.d. Gaussian vector `(V, W, W')`, so the
law of the head is invariant under `(V, A) ↦ (V, Oᵀ A O)`
(`measurePreserving_conjHead`).  This is the "orthogonal invariance of the
Gaussian law" the source invokes.

Source: arXiv:2604.01978v1, assumption (G) and the proof of
`lem:drift_on_simplex_selfcontained`.
-/

import Transformer.Homogenized.HeadConj
import Transformer.Homogenized.GaussianInit
import Mathlib.Probability.Distributions.Gaussian.Multivariate

open scoped NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- Rotating each column of `g` does not change `Σ g_{ij}²`. -/
theorem sum_sq_symm_col {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (g : Fin d → Fin d → ℝ) :
    ∑ i, ∑ j, (O.symm (WithLp.toLp 2 fun k => g k j)) i ^ 2 = ∑ i, ∑ j, g i j ^ 2 := by
  rw [Finset.sum_comm, Finset.sum_comm (f := fun i j => g i j ^ 2)]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← EuclideanSpace.real_norm_sq_eq, O.symm.norm_map, EuclideanSpace.real_norm_sq_eq]

/-- `W ↦ Oᵀ W` on the two query-key blocks, the identity on the value block. -/
noncomputable def blockRot {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (Fin 3 × Fin d × Fin d → ℝ) →ₗ[ℝ] (Fin 3 × Fin d × Fin d → ℝ) where
  toFun z p := if p.1 = 0 then z p else O.symm (WithLp.toLp 2 fun k => z (p.1, k, p.2.2)) p.2.1
  map_add' z w := by
    funext p
    by_cases h : p.1 = 0
    · simp [h]
    · simp only [h, ite_false, Pi.add_apply]
      rw [show (WithLp.toLp 2 fun k => z (p.1, k, p.2.2) + w (p.1, k, p.2.2)) =
        WithLp.toLp 2 (fun k => z (p.1, k, p.2.2)) + WithLp.toLp 2 (fun k => w (p.1, k, p.2.2))
        from rfl, map_add]
      rfl
  map_smul' c z := by
    funext p
    by_cases h : p.1 = 0
    · simp [h]
    · simp only [h, ite_false, Pi.smul_apply, RingHom.id_apply]
      rw [show (WithLp.toLp 2 fun k => c • z (p.1, k, p.2.2)) =
        c • WithLp.toLp 2 (fun k => z (p.1, k, p.2.2)) from rfl, map_smul]
      rfl

/-- `blockRot O` is an isometry of the Euclidean norm of `(V, W, W')`. -/
theorem norm_blockRot {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (z : EuclideanSpace ℝ (Fin 3 × Fin d × Fin d)) :
    ‖WithLp.toLp 2 (blockRot O z.ofLp)‖ = ‖z‖ := by
  rw [← sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _), EuclideanSpace.real_norm_sq_eq,
    EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type, Fintype.sum_prod_type,
    Fin.sum_univ_three, Fin.sum_univ_three]
  simp only [Fintype.sum_prod_type]
  simp only [blockRot, LinearMap.coe_mk, AddHom.coe_mk, ite_true,
    show (1 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 0 by decide, ite_false]
  rw [sum_sq_symm_col O fun k j => z.ofLp (1, k, j), sum_sq_symm_col O fun k j => z.ofLp (2, k, j)]

/-- The standard Gaussian on `(V, W, W')` is invariant under `blockRot O`. -/
theorem map_blockRot_pi_std {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (Measure.pi fun _ : Fin 3 × Fin d × Fin d => gaussianReal 0 1).map (blockRot O) =
      Measure.pi fun _ => gaussianReal 0 1 := by
  set E := EuclideanSpace ℝ (Fin 3 × Fin d × Fin d)
  let T : E →ₗᵢ[ℝ] E :=
    { toLinearMap := (WithLp.linearEquiv 2 ℝ _).symm.toLinearMap ∘ₗ blockRot O ∘ₗ
        (WithLp.linearEquiv 2 ℝ _).toLinearMap
      norm_map' := norm_blockRot O }
  have hof : Measurable (WithLp.ofLp : E → _) := (PiLp.continuous_ofLp 2 _).measurable
  have hstd : (Measure.pi fun _ : Fin 3 × Fin d × Fin d => gaussianReal 0 1) =
      (stdGaussian E).map WithLp.ofLp := by
    rw [← map_pi_eq_stdGaussian, Measure.map_map hof (PiLp.continuous_toLp 2 _).measurable]
    exact Measure.map_id.symm
  have hB : Measurable (blockRot O) := (blockRot O).continuous_of_finiteDimensional.measurable
  rw [hstd, Measure.map_map hB hof]
  conv_rhs => rw [← stdGaussian_map (T.toLinearIsometryEquiv rfl),
    Measure.map_map hof (T.toLinearIsometryEquiv rfl).continuous.measurable]
  rfl

/-- The variance profile of `(V, W, W')`: `σV²` on the value block, `σA²` on
the two query-key blocks. -/
def blockSigma {d : ℕ} (σV σA : ℝ≥0) (p : Fin 3 × Fin d × Fin d) : ℝ≥0 :=
  if p.1 = 0 then σV else σA

/-- The Gaussian law of `(V, W, W')` of assumption (G) is invariant under
`blockRot O`. -/
theorem map_blockRot_pi {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (σV σA : ℝ≥0) :
    (Measure.pi fun p => gaussianReal 0 (blockSigma σV σA p ^ 2)).map (blockRot O) =
      Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2) := by
  set s := blockSigma (d := d) σV σA
  have hmul : ∀ p, Measurable fun x : ℝ => (s p : ℝ) * x := fun p => measurable_const_mul _
  have hγ : (Measure.pi fun p => gaussianReal 0 (s p ^ 2)) =
      (Measure.pi fun _ => gaussianReal 0 1).map fun z p => (s p : ℝ) * z p := by
    rw [Measure.pi_map_pi fun p => (hmul p).aemeasurable]
    congr 1; funext p
    rw [gaussianReal_map_const_mul, mul_zero, mul_one]
    congr 1
  have hD : Measurable fun (z : Fin 3 × Fin d × Fin d → ℝ) p => (s p : ℝ) * z p :=
    Measurable.of_eval fun p => (hmul p).comp (measurable_pi_apply p)
  have hB : Measurable (blockRot O) := (blockRot O).continuous_of_finiteDimensional.measurable
  have hcomm : blockRot O ∘ (fun z p => (s p : ℝ) * z p) =
      (fun z p => (s p : ℝ) * z p) ∘ blockRot O := by
    funext z p
    by_cases h : p.1 = 0
    · simp [blockRot, s, blockSigma, h]
    · simp only [Function.comp_apply, blockRot, LinearMap.coe_mk, AddHom.coe_mk, h, ite_false,
        s, blockSigma]
      rw [show (WithLp.toLp 2 fun k => (σA : ℝ) * z (p.1, k, p.2.2)) =
        (σA : ℝ) • WithLp.toLp 2 (fun k => z (p.1, k, p.2.2)) from rfl, map_smul]
      rfl
  rw [hγ, Measure.map_map hB hD, hcomm, ← Measure.map_map hD hB, map_blockRot_pi_std]

/-- `(V, A) ↦ (V, Oᵀ A O)`, as a measurable equivalence of head parameters. -/
noncomputable def conjHead {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    HeadParam d ≃ᵐ HeadParam d :=
  MeasurableEquiv.prodCongr (MeasurableEquiv.refl _)
    (conjMat O).toContinuousLinearEquiv.toHomeomorph.toMeasurableEquiv

theorem conjHead_apply {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (θ : HeadParam d) :
    conjHead O θ = (θ.1, conjMat O θ.2) := rfl

/-- The unpacking of `(V, W, W')` into the head `(V, W W'ᵀ)`. -/
def headOfBlocks {d : ℕ} (z : Fin 3 × Fin d × Fin d → ℝ) : HeadParam d :=
  (Matrix.of fun i j => z (0, i, j),
    Matrix.of (fun i j => z (1, i, j)) * (Matrix.of fun i j => z (2, i, j)).transpose)

theorem measurable_headOfBlocks {d : ℕ} : Measurable (headOfBlocks (d := d)) := by
  unfold headOfBlocks
  exact (Continuous.measurable (by fun_prop)).prodMk (Continuous.measurable (by fun_prop))

/-- Rotating the blocks is conjugating the head: `Oᵀ (W W'ᵀ) O = (Oᵀ W)(Oᵀ W')ᵀ`. -/
theorem conjHead_headOfBlocks {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (z : Fin 3 × Fin d × Fin d → ℝ) :
    conjHead O (headOfBlocks z) = headOfBlocks (blockRot O z) := by
  rw [conjHead_apply, headOfBlocks, headOfBlocks, conjMat_mul_transpose]
  refine Prod.ext ?_ ?_
  · ext i j; simp [blockRot]
  · dsimp only
    congr 1
    · ext i j
      simp [colMap_apply, blockRot, show (1 : Fin 3) ≠ 0 by decide]
    · congr 1
      ext i j
      simp [colMap_apply, blockRot, show (2 : Fin 3) ≠ 0 by decide]

/-- **The head law is invariant under `A ↦ Oᵀ A O`.**  Under assumption (G),
`(W, W')` and `(Oᵀ W, Oᵀ W')` have the same law, since the i.i.d. Gaussian law
of `(V, W, W')` is invariant under the rotation of the columns of `W` and
`W'`; and `Oᵀ W (Oᵀ W')ᵀ = Oᵀ (W W'ᵀ) O`. -/
theorem measurePreserving_conjHead {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    MeasurePreserving (conjHead O) ρ ρ := by
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', hind, hmV, hmW, hmW', rfl⟩ := hρ
  set F : Ω → Fin 3 × Fin d × Fin d → ℝ := fun ω p =>
    if p.1 = 0 then Vr ω p.2.1 p.2.2 else if p.1 = 1 then Wr ω p.2.1 p.2.2 else Wr' ω p.2.1 p.2.2
  have hFp : ∀ p, Measurable fun ω => F ω p := fun p => by
    simp only [F]
    split_ifs
    exacts [(hV.eval).eval, (hW.eval).eval, (hW'.eval).eval]
  have hlaw : P.map F = Measure.pi fun p => gaussianReal 0 (blockSigma σV σA p ^ 2) := by
    rw [show F = fun ω p => F ω p from rfl,
      hind.map_fun_eq_pi_map fun p => (hFp p).aemeasurable]
    congr 1; funext p
    obtain ⟨a, i, j⟩ := p
    fin_cases a
    · simpa [F, blockSigma] using hmV i j
    · simpa [F, blockSigma] using hmW i j
    · simpa [F, blockSigma] using hmW' i j
  have hF : Measurable F := Measurable.of_eval hFp
  have hg : Measurable (headOfBlocks (d := d)) := measurable_headOfBlocks
  have hB : Measurable (blockRot O) := (blockRot O).continuous_of_finiteDimensional.measurable
  have hρ : P.map (fun ω => ((Vr ω, Wr ω * (Wr' ω).transpose) : HeadParam d)) =
      (P.map F).map headOfBlocks := by
    rw [Measure.map_map hg hF]
    congr 1
  refine ⟨(conjHead O).measurable, ?_⟩
  rw [hρ, hlaw, Measure.map_map (conjHead O).measurable hg]
  rw [show conjHead O ∘ headOfBlocks = headOfBlocks ∘ blockRot O from
    funext (conjHead_headOfBlocks O), ← Measure.map_map hg hB, map_blockRot_pi]


/-- The hypotheses of `measurePreserving_conjHead` are satisfiable. -/
example (d : ℕ) (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧ O = O :=
  ⟨isGaussianHeadLaw_dirac_zero d, rfl⟩

end Homogenized
end Transformer
