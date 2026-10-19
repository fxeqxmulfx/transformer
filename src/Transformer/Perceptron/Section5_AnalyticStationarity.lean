module

public import Transformer.Perceptron.Section5_ReLUObstruction

/-!
# The actual analytic first variation in arbitrary dimension

ArXiv:2601.21366v2, §5.3, the analytic alternative in the proof of
`thm: any.d` (i). The actual stationary field is analytic when the
original activation is analytic. Positive original uniform measure
of the original support then makes that field zero on the whole
sphere. Its actual first variation is therefore constant there.

All original neurons, the original probability and the original
primitive of twice the activation are retained. The interaction's
actual ambient analyticity and the spherical uniqueness theorem
justify the continuation step, before any Hessian argument.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original perceptron drift is ambient analytic for an analytic
activation. Source: arXiv:2601.21366v2, §5.3, analytic `g=∇H` in `thm: any.d` (i). -/
theorem analyticOnNhd_drift {d : ℕ} (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ univ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    AnalyticOnNhd ℝ (drift σ ω a) univ := by
  let D : EucSpace d → EucSpace d := fun x =>
    ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j
  have hD : AnalyticOnNhd ℝ D univ := by
    intro x _
    apply Finset.analyticAt_fun_sum
    intro j _
    have hj : AnalyticAt ℝ (fun z => σ (inner (𝕜 := ℝ) (a j) z)) x :=
      (hσ _ (mem_univ _)).comp (f := fun z => inner (𝕜 := ℝ) (a j) z)
        ((innerSL ℝ (a j)).analyticAt x)
    exact (analyticAt_const.fun_mul hj).fun_smul analyticAt_const
  intro x _
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (D z)) x :=
    (L.analyticAt_bilinear (x, D x)).comp (f := fun z => (z, D z))
      (analyticAt_id.prod (hD x (mem_univ x)))
  exact (hD x (mem_univ x)).fun_sub (hi.fun_smul analyticAt_id)

example : AnalyticOnNhd ℝ (fun _ : ℝ => (0 : ℝ)) univ := fun _ _ => analyticAt_const

/-- The original attention-plus-perceptron stationary field is analytic
in the actual ambient coordinates. Source: arXiv:2601.21366v2, §5.3,
analyticity of `g=∇H` in the second alternative of `thm: any.d` (i). -/
theorem analyticOnNhd_energyGrad {d : ℕ} (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (energyGrad β σ ω a μ) univ := by
  have heq : energyGrad β σ ω a μ = fun x =>
      β⁻¹ • attentionGradMap (β • LinearMap.id) μ x + drift σ ω a x := by
    funext x
    rw [attentionGradMap_smul_id, smul_smul, inv_mul_cancel₀ hβ, one_smul]
    rfl
  rw [heq]
  intro x hx
  exact (analyticAt_const.fun_smul (analyticOnNhd_attentionGradMap (β • LinearMap.id) μ x hx)).fun_add
    (analyticOnNhd_drift σ hσ ω a x hx)

example : (1 : ℝ) ≠ 0 ∧ AnalyticOnNhd ℝ (fun _ : ℝ => (0 : ℝ)) univ :=
  ⟨one_ne_zero, fun _ _ => analyticAt_const⟩

/-- Positive actual uniform measure of the original stationary support
forces the actual analytic stationary field to vanish everywhere.
Source: arXiv:2601.21366v2, §5.3, the analytic continuation step in `thm: any.d` (i). -/
theorem energyGrad_zero_of_stationary_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ univ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationary β σ ω a μ) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∀ x : SSphere d, energyGrad β σ ω a μ (x : EucSpace d) = 0 := by
  apply vector_eq_zero_on_sphere_of_positive_zero_measure hd (energyGrad β σ ω a μ)
    (analyticOnNhd_energyGrad β hβ σ hσ ω a μ) ν hν
  exact hpos.trans_le (measure_mono (by intro x hx; exact hμ x hx))

/-- The actual first variation is constant on the original sphere under
the positive-uniform-support hypothesis. Source: arXiv:2601.21366v2,
§5.3, `H` constant in the analytic alternative of `thm: any.d` (i).
The covector used below is the actual derivative of the original energy. -/
theorem firstVariation_eq_of_stationary_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∀ x y : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) =
      firstVariation β φ ω a μ (y : EucSpace d) := by
  have hzero := energyGrad_zero_of_stationary_positive_uniform_support hd β hβ σ hσ ω a μ hμ ν hν hpos
  let D := fun x : EucSpace d =>
    (∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d) ∂(μ : Measure (SSphere d))) +
      ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j
  have hD (x : EucSpace d) : HasFDerivAt (firstVariation β φ ω a μ) (innerSL ℝ (D x)) x := by
    have h := hasGradientAt_iff_hasFDerivAt.mp (hasGradientAt_firstVariation hβ φ σ hφ ω a μ x)
    apply h.congr_fderiv
    ext v
    rfl
  have hp (z : EucSpace d) (hz : ‖z‖ = 1) : proj d z (D z) = 0 := by
    rw [← energyGrad_eq_proj]
    exact hzero ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
  exact eq_on_sphere_of_projected_derivative_zero hd (firstVariation β φ ω a μ) D hD hp

/-- The original constant first variation has an actual real level,
chosen from an actual sphere point of the original probability space.
Source: arXiv:2601.21366v2, §5.3, `H` constant in `thm: any.d` (i). -/
theorem exists_constant_firstVariation_of_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ C : ℝ, ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C := by
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  refine ⟨firstVariation β φ ω a μ (x₀ : EucSpace d), ?_⟩
  exact fun x => firstVariation_eq_of_stationary_positive_uniform_support
    hd β hβ φ σ hφ hσ ω a μ hμ ν hν hpos x x₀

/-- Along every actual sphere-valued curve, the actual first variation
has zero second derivative after the global continuation step.
Source: arXiv:2601.21366v2, §5.3, the global diagonal Hessian cancellation
in the analytic alternative of `thm: any.d` (i). The original scalar
function is retained; constancy was proved from the original stationarity. -/
theorem hasDerivAt_deriv_firstVariation_on_sphere_curve {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) (γ : ℝ → SSphere d) (t : ℝ) :
    HasDerivAt (deriv (fun r => firstVariation β φ ω a μ (γ r : EucSpace d))) 0 t := by
  obtain ⟨C, hC⟩ := exists_constant_firstVariation_of_positive_uniform_support
    hd β hβ φ σ hφ hσ ω a μ hμ ν hν hpos
  have heq : (fun r => firstVariation β φ ω a μ (γ r : EucSpace d)) = fun _ => C :=
    funext fun r => hC (γ r)
  rw [heq]
  simpa using hasDerivAt_const t (0 : ℝ)

/-- The actual uniform probability with zero original weights witnesses
all analytic, stationary, primitive and positive-support hypotheses.
Source: arXiv:2601.21366v2, §5.3, the constant first-variation case.
The witness uses the original probability and its actual full support. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (1 : ℝ) ≠ 0 ∧
    AnalyticOnNhd ℝ (fun _ : ℝ => (0 : ℝ)) univ ∧
    (∀ s, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    IsStationary 1 (0 : ℝ → ℝ) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) (μ : Measure (SSphere 2)).support := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  refine ⟨μ, le_rfl, one_ne_zero, fun _ _ => analyticAt_const,
    fun s => by simpa using hasDerivAt_const s (0 : ℝ),
    isStationary_zero_weights_of_uniform 1 one_ne_zero 0 0 μ hν, hν, ?_⟩
  rw [support_eq_univ_of_uniform μ hν]
  simp

end Transformer.Perceptron
