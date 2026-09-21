/-
# Homogenized Transformers — the Gaussian head ensemble

The head law of `eq: tformers.at.initialization` (assumption (G)) of
arXiv:2604.01978v1, *Homogenized Transformers*, built explicitly: `d²` value
entries and `2d²` query/key entries, independent and centered Gaussian with
variances `σ_V²` and `σ_A²`, and the head `(V, W W'ᵀ)` read off them.

`IsGaussianHeadLaw` had one witness, the degenerate `σ_V = σ_A = 0`, at which
every Gaussian is a Dirac mass and the law of the head is `δ_0`.  That witness
cannot carry the standard scaling `σ_V² = 1/d` of §2.5, under which the whole
Gaussian section is written; `gaussHeadLaw` carries every pair `(σ_V, σ_A)`,
and `exists_isGaussianHeadLaw_stdScaling` is that scaling.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`, §2.5.
-/

import Transformer.Homogenized.GaussianInit

open scoped BigOperators NNReal ENNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- The entries of the three matrices `V`, `W`, `W'` of one head, indexed by
the block and the two matrix indices. -/
abbrev GaussIdx (d : ℕ) : Type := Fin 3 × Fin d × Fin d

/-- The variance of one entry: `σ_V²` on the value block `0`, `σ_A²` on the two
query/key blocks `1` and `2`. -/
def entryVar {d : ℕ} (σV σA : ℝ≥0) (p : GaussIdx d) : ℝ≥0 :=
  if p.1 = 0 then σV ^ 2 else σA ^ 2

@[simp] theorem entryVar_zero {d : ℕ} (σV σA : ℝ≥0) (i j : Fin d) :
    entryVar σV σA ((0 : Fin 3), i, j) = σV ^ 2 := by
  simp [entryVar]

@[simp] theorem entryVar_one {d : ℕ} (σV σA : ℝ≥0) (i j : Fin d) :
    entryVar σV σA ((1 : Fin 3), i, j) = σA ^ 2 := by
  simp [entryVar]

@[simp] theorem entryVar_two {d : ℕ} (σV σA : ℝ≥0) (i j : Fin d) :
    entryVar σV σA ((2 : Fin 3), i, j) = σA ^ 2 := by
  simp [entryVar]

/-- The `d²`-fold product of independent centered Gaussians, one per entry. -/
noncomputable def gaussEnsemble (d : ℕ) (σV σA : ℝ≥0) : Measure (GaussIdx d → ℝ) :=
  Measure.pi fun p => gaussianReal 0 (entryVar σV σA p)

instance instIsProbabilityMeasureGaussEnsemble (d : ℕ) (σV σA : ℝ≥0) :
    IsProbabilityMeasure (gaussEnsemble d σV σA) := by
  unfold gaussEnsemble
  infer_instance

/-- The matrix sitting on block `k` of a sample. -/
def gaussBlock {d : ℕ} (k : Fin 3) (ω : GaussIdx d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => ω (k, i, j)

@[simp] theorem gaussBlock_apply {d : ℕ} (k : Fin 3) (ω : GaussIdx d → ℝ) (i j : Fin d) :
    gaussBlock k ω i j = ω (k, i, j) := rfl

theorem measurable_gaussBlock {d : ℕ} (k : Fin 3) :
    Measurable (gaussBlock (d := d) k) :=
  (Matrix.measurable_of _ _ _).comp
    (Measurable.of_eval fun _ => Measurable.of_eval fun _ => measurable_pi_apply _)

/-- Each entry of each block is a centered Gaussian of the variance its block
carries. -/
theorem map_gaussBlock {d : ℕ} (σV σA : ℝ≥0) (k : Fin 3) (i j : Fin d) :
    Measure.map (fun ω => gaussBlock k ω i j) (gaussEnsemble d σV σA)
      = gaussianReal 0 (entryVar σV σA (k, i, j)) :=
  (measurePreserving_eval (fun p => gaussianReal 0 (entryVar σV σA p)) (k, i, j)).map_eq

/-- The law of the head `(V, A) = (V, W W'ᵀ)` under the Gaussian ensemble. -/
noncomputable def gaussHeadLaw (d : ℕ) (σV σA : ℝ≥0) : Measure (HeadParam d) :=
  Measure.map
    (fun ω => ((gaussBlock 0 ω, gaussBlock 1 ω * (gaussBlock 2 ω).transpose) : HeadParam d))
    (gaussEnsemble d σV σA)

theorem measurable_gaussHead (d : ℕ) :
    Measurable (fun ω : GaussIdx d → ℝ =>
      ((gaussBlock 0 ω, gaussBlock 1 ω * (gaussBlock 2 ω).transpose) : HeadParam d)) := by
  refine (measurable_gaussBlock 0).prodMk ?_
  refine (Matrix.measurable_of _ _ _).comp
    (Measurable.of_eval fun i => Measurable.of_eval fun j => ?_)
  show Measurable fun ω : GaussIdx d → ℝ => ∑ k, ω (1, i, k) * ω (2, j, k)
  exact Finset.measurable_sum _ fun k _ => (measurable_pi_apply _).mul (measurable_pi_apply _)

instance instIsProbabilityMeasureGaussHeadLaw (d : ℕ) (σV σA : ℝ≥0) :
    IsProbabilityMeasure (gaussHeadLaw d σV σA) :=
  (Measure.isProbabilityMeasure_map_iff (measurable_gaussHead d).aemeasurable).2 inferInstance

/-- **At `σ_A = 0` the attention matrix vanishes almost surely**: every entry of
`W` is `𝒩(0,0) = δ_0`, so `A = W W'ᵀ = 0`, whatever `σ_V` is. -/
theorem ae_snd_eq_zero_gaussHeadLaw (d : ℕ) (σV : ℝ≥0) :
    ∀ᵐ θ ∂(gaussHeadLaw d σV 0), θ.2 = 0 := by
  have hs : MeasurableSet {θ : HeadParam d | θ.2 = 0} :=
    measurable_snd (measurableSet_singleton 0)
  refine (ae_map_iff (measurable_gaussHead d).aemeasurable hs).2 ?_
  have h1 : ∀ i j : Fin d, ∀ᵐ ω ∂(gaussEnsemble d σV 0), gaussBlock 1 ω i j = 0 := by
    intro i j
    have := map_gaussBlock (d := d) σV 0 1 i j
    simp only [entryVar_one, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
      gaussianReal_zero_var] at this
    have hm : Measurable fun ω : GaussIdx d → ℝ => gaussBlock 1 ω i j := measurable_pi_apply _
    exact (ae_map_iff hm.aemeasurable (measurableSet_singleton (0 : ℝ))).1
      (by rw [this]; exact (ae_dirac_iff (measurableSet_singleton 0)).2 rfl)
  filter_upwards [ae_all_iff.2 fun i => ae_all_iff.2 (h1 i)] with ω hω
  simp only [gaussBlock_apply] at hω
  ext i j
  simp [Matrix.mul_apply, hω]

/-- At `σ_A = 0` an integrand that reads only the attention matrix integrates
to its value at `A = 0`. -/
theorem integral_snd_gaussHeadLaw_zero (d : ℕ) (σV : ℝ≥0)
    (F : Matrix (Fin d) (Fin d) ℝ → ℝ) :
    ∫ θ, F θ.2 ∂(gaussHeadLaw d σV 0) = F 0 := by
  rw [integral_congr_ae ((ae_snd_eq_zero_gaussHeadLaw d σV).mono
    fun θ h => (congrArg F h : F θ.2 = (fun _ => F 0) θ))]
  simp

theorem integrable_snd_gaussHeadLaw_zero (d : ℕ) (σV : ℝ≥0)
    (F : Matrix (Fin d) (Fin d) ℝ → ℝ) :
    Integrable (fun θ : HeadParam d => F θ.2) (gaussHeadLaw d σV 0) :=
  (integrable_const (F 0)).congr ((ae_snd_eq_zero_gaussHeadLaw d σV).mono
    fun _ h => (congrArg F h).symm)

/-- **Assumption (G) is satisfiable at every pair of variances.**  The Gaussian
ensemble realizes `IsGaussianHeadLaw d σ_V σ_A` for every `σ_V, σ_A`, in every
dimension — not only at the degenerate `σ_V = σ_A = 0` of
`isGaussianHeadLaw_dirac_zero`.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem isGaussianHeadLaw_gaussHeadLaw (d : ℕ) (σV σA : ℝ≥0) :
    IsGaussianHeadLaw d σV σA (gaussHeadLaw d σV σA) := by
  refine ⟨GaussIdx d → ℝ, inferInstance, gaussEnsemble d σV σA,
    gaussBlock 0, gaussBlock 1, gaussBlock 2, inferInstance,
    measurable_gaussBlock 0, measurable_gaussBlock 1, measurable_gaussBlock 2, ?_,
    fun i j => ?_, fun i j => ?_, fun i j => ?_, rfl⟩
  · have hfun : (fun p : GaussIdx d => fun ω : GaussIdx d → ℝ =>
        if p.1 = 0 then gaussBlock 0 ω p.2.1 p.2.2
        else if p.1 = 1 then gaussBlock 1 ω p.2.1 p.2.2
        else gaussBlock 2 ω p.2.1 p.2.2) = fun p ω => ω p := by
      funext p ω
      obtain ⟨k, i, j⟩ := p
      fin_cases k <;> rfl
    rw [hfun]
    exact iIndepFun_pi (X := fun _ : GaussIdx d => (id : ℝ → ℝ)) fun _ => aemeasurable_id
  · rw [map_gaussBlock, entryVar_zero]
  · rw [map_gaussBlock, entryVar_one]
  · rw [map_gaussBlock, entryVar_two]

/-- The standard scaling `σ_V² = 1/d` of §2.5. -/
noncomputable def stdSigmaV (d : ℕ) : ℝ≥0 := NNReal.sqrt (d : ℝ≥0)⁻¹

@[simp] theorem stdSigmaV_sq (d : ℕ) : ((stdSigmaV d : ℝ)) ^ 2 = 1 / d := by
  rw [stdSigmaV, ← NNReal.coe_pow, NNReal.sq_sqrt, NNReal.coe_inv, NNReal.coe_natCast,
    one_div]

/-- **Assumption (G) is satisfiable at the standard scaling.**  The source
adopts `σ_V² = 1/d` throughout §2.5 — "the standard scaling, which we adopt
henceforth" — and every theorem of that section is written under it.  The
degenerate witness `isGaussianHeadLaw_dirac_zero` has `σ_V = 0` and cannot
carry it; this one does, at every `σ_A` and every `d ≥ 1`.

Source: arXiv:2604.01978v1, §2.5. -/
theorem isGaussianHeadLaw_stdScaling (d : ℕ) (σA : ℝ≥0) :
    ((stdSigmaV d : ℝ)) ^ 2 = 1 / d ∧
      IsGaussianHeadLaw d (stdSigmaV d) σA (gaussHeadLaw d (stdSigmaV d) σA) :=
  ⟨stdSigmaV_sq d, isGaussianHeadLaw_gaussHeadLaw d _ _⟩

end Homogenized
end Transformer
