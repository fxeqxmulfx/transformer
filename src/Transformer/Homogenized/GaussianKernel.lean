/-
# Homogenized Transformers — the noise kernel at Gaussian initialization

Formalization of `lem:lemma_app` of arXiv:2604.01978v1, *Homogenized
Transformers*, §5: under `eq: tformers.at.initialization` the drift vanishes
and the covariance kernel `eq:covariance_kernel` collapses to a scalar,

  `K[μ](x,y) = (1/d) E_A⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ I_d`.

The computation has two halves.  The first is that the value matrix comes out
of the field: `B_θ[μ](x) = V m_{β,A}[μ](x)`, which is `attnFieldOf_eq_valueMap_softBary`
and is proved here — it is where the softmax barycenter of `Barycenter.lean`
enters the dynamics at all.  The second is `E_V[V a bᵀ Vᵀ] = σ_V²⟨a,b⟩ I_d`
together with `σ_V² = 1/d`, which is the lemma itself.
-/

import Transformer.Homogenized.Barycenter
import Transformer.Homogenized.GaussianEnsemble

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The value matrix comes out of the field -/

/-- The value matrix is linear, so it commutes with a scalar. -/
theorem valueMap_smul {d : ℕ} (θ : HeadParam d) (c : ℝ) (y : EucSpace d) :
    valueMap θ (c • y) = c • valueMap θ y := by
  simp [valueMap]

/-- The value matrix is a continuous linear map on a finite-dimensional space,
so it comes out of a Bochner integral. -/
theorem valueMap_integral {d : ℕ} (θ : HeadParam d) (μ : Measure (EucSpace d))
    (f : EucSpace d → EucSpace d) (hf : Integrable f μ) :
    valueMap θ (∫ y, f y ∂μ) = ∫ y, valueMap θ (f y) ∂μ :=
  ((LinearMap.toContinuousLinearMap
    (Matrix.toEuclideanLin θ.1)).integral_comp_comm hf).symm

/-- **`ξ_θ[μ](x) = V m_{β,A}[μ](x)`**, the identity that lets `lem:lemma_app`
average over `V` and `A` separately: the attention field is the value matrix
applied to the softmax barycenter, the normalizer being a scalar and `V`
linear.

The source writes this for `ξ_θ` rather than `B_θ`, which is the same thing
once `b_{ρ*}[μ] ≡ 0` — the first clause of `lem:lemma_app`.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem attnFieldOf_eq_valueMap_softBary {d : ℕ} (β : ℝ) (θ : HeadParam d)
    (μ : Measure (EucSpace d)) (x : EucSpace d)
    (hint : Integrable (fun y => attnWeight β θ x y • y) μ) :
    attnFieldOf β θ μ x = valueMap θ (softBary β θ.2 μ x) := by
  simp only [attnFieldOf, softBary, softWeight_eq_attnWeight]
  rw [valueMap_smul, valueMap_integral θ μ _ hint]
  congr 1
  exact integral_congr_ae
    (Filter.Eventually.of_forall fun y => (valueMap_smul θ (attnWeight β θ x y) y).symm)

/-- The hypothesis of `attnFieldOf_eq_valueMap_softBary` is satisfiable: at a
Dirac token measure every function is integrable. -/
example {d : ℕ} (β : ℝ) (θ : HeadParam d) (x z : EucSpace d) :
    Integrable (fun y => attnWeight β θ x y • y) (Measure.dirac z) :=
  integrable_dirac enorm_lt_top

/-! ### The kernel -/

/-- **Lemma (lem:lemma_app).**  Under `eq: tformers.at.initialization`,
`b_{ρ*}[μ] ≡ 0` and

  `K[μ](x,y) = (1/d) E_A⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ I_d`.

**What the source says and what is written here.**

* The source states the lemma at the empirical measure `μ_X`; it is carried
  here at a general measure on `ℝ^d`, which is where `covKernel` of
  `eq:covariance_kernel` lives and where `thm:large_beta_meta` uses it.  At
  `μ = μ_X` the first clause is `bField_gaussian`, §2.3.3 item (ii); it is not
  a consequence of that theorem at a general `μ`, so it is restated and the
  two clauses stay one lemma, as in the source.
* The source's display carries the two projections,
  `K[μ](x_i,x_j) = (1/d) Proj_{x_i} E⟨m,m⟩ I_d Proj_{x_j}`, because its `K` is
  the kernel as `eq:cross_variation_kernel` sandwiches it.  Written without
  them the identity is the stronger one — `K[μ](x,y)` is a multiple of the
  identity matrix — and the sandwiched display follows by applying `Proj_x` to
  both sides at `v = Proj_y v'`.
* `σ_V² = 1/d` is the standard scaling the source substitutes silently at the
  last line of the proof; it is a hypothesis here.

Not proved here.

Source: arXiv:2604.01978v1, `lem:lemma_app`. -/
theorem gaussian_drift_and_kernel {d : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : ((σV : ℝ)) ^ 2 = 1 / (d : ℝ)) (μ : Measure (EucSpace d)) :
    (∀ z : EucSpace d, meanFieldOf β ρ μ z = 0) ∧
      ∀ x y v : EucSpace d,
        covKernel β ρ μ x y v = ((1 / (d : ℝ)) * baryCorr β ρ μ x y) • v := by
  sorry

/-- The hypotheses of `gaussian_drift_and_kernel` are satisfiable in every
dimension: the Gaussian ensemble at the standard scaling `σ_V² = 1/d`. -/
example (d : ℕ) (σA : ℝ≥0) :
    IsGaussianHeadLaw d (stdSigmaV d) σA (gaussHeadLaw d (stdSigmaV d) σA) ∧
      ((stdSigmaV d : ℝ)) ^ 2 = 1 / (d : ℝ) :=
  ⟨isGaussianHeadLaw_gaussHeadLaw d _ _, stdSigmaV_sq d⟩

end Homogenized
end Transformer
