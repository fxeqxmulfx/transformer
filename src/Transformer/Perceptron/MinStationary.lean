/-
# Perceptrons and attention's mean-field landscape — the minimizer is stationary

The first half of "it is clear that the minimizer `μ⋆` is SOPD"
(`rem:strictSOPD-perceptron`): a global minimizer of the coupled energy
satisfies `eq: steady.state` on its whole support (`isStationary_of_isMin`).

* Moving mass `ε` to a point `z` cannot lower the energy, so the slope
  `δE/δμ[μ](z) - ∫ δE/δμ[μ] dμ` of `energy_mixDirac` is nonnegative for every
  `z` on the sphere (`integral_firstVariation_le`).

* The first variation is continuous, so it equals its minimum over the sphere
  everywhere on the support (`firstVariation_le_of_mem_support`), and at a
  minimum on the sphere the tangential gradient vanishes
  (`proj_eq_zero_of_isMin_sphere`).  That tangential gradient is `energyGrad`
  (`energyGrad_eq_proj`).

The source states `eq: steady.state` for continuous `σ`.  The minimizer needs
no continuity of `σ`: the first variation is differentiable as soon as `σ` has
the primitive `φ`, its gradient being pointwise in `σ`.  Nor does it need
`β > 0`: `β ≠ 0` suffices here.

Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`, and
`rem:strictSOPD-perceptron`.
-/

import Transformer.Perceptron.MixDirac
import Transformer.Perceptron.GeodesicCurve

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- **At a minimizer, moving mass to a point does not help**: every `z` on the
sphere has `∫ δE/δμ[μ] dμ ≤ δE/δμ[μ](z)`.

`E[ν_ε] - E[μ] = ε L + ε² M` with `L` that difference (`energy_mixDirac`), and
it is `≥ 0`; so `L + ε M ≥ 0` for `ε ∈ (0, 1)`, and `L ≥ 0` in the limit. -/
theorem integral_firstVariation_le (β : ℝ) {φ : ℝ → ℝ} (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) {μ : Perspective.ProbSphere d}
    (hμ : ∀ ν : Perspective.ProbSphere d, energy β φ ω a μ ≤ energy β φ ω a ν) (z : SSphere d) :
    ∫ x, firstVariation β φ ω a μ (x : EucSpace d) ∂(μ : Measure (SSphere d)) ≤
      firstVariation β φ ω a μ (z : EucSpace d) := by
  set L := firstVariation β φ ω a μ (z : EucSpace d) -
    ∫ x, firstVariation β φ ω a μ (x : EucSpace d) ∂(μ : Measure (SSphere d)) with hL
  set M := (2 * β)⁻¹ * ((∫ x, Perspective.partitionMu d β μ (x : EucSpace d)
      ∂(μ : Measure (SSphere d))) - 2 * Perspective.partitionMu d β μ (z : EucSpace d) +
        Real.exp β) with hM
  have hpos : ∀ ε ∈ Set.Ioo (0 : ℝ) 1, 0 ≤ L + ε * M := fun ε hε => by
    have h := hμ (mixDirac μ z ε hε.1.le hε.2.le)
    rw [energy_mixDirac β hφ ω a μ z hε.1.le hε.2.le, ← hL, ← hM] at h
    exact (mul_nonneg_iff_of_pos_left hε.1).mp (by linarith)
  have hlim : Tendsto (fun ε : ℝ => L + ε * M) (𝓝[>] 0) (𝓝 L) := by
    have h : Tendsto (fun ε : ℝ => L + ε * M) (𝓝 0) (𝓝 (L + 0 * M)) :=
      (continuous_const.add (continuous_id.mul continuous_const)).tendsto (0 : ℝ)
    rw [zero_mul, add_zero] at h
    exact h.mono_left nhdsWithin_le_nhds
  exact sub_nonneg.mp (ge_of_tendsto hlim (mem_of_superset (Ioo_mem_nhdsGT one_pos) hpos))

/-- The hypotheses of `integral_firstVariation_le` are satisfiable: `d = 1`,
`β = 1`, `φ = 0`, and the minimizer of `existsUnique_min_energy`. -/
example (ω : Idx 1 → ℝ) (a : Idx 1 → EucSpace 1) : Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    ∃ μ : Perspective.ProbSphere 1, ∀ ν : Perspective.ProbSphere 1,
      energy 1 (fun _ => 0) ω a μ ≤ energy 1 (fun _ => 0) ω a ν :=
  ⟨continuous_const, (existsUnique_min_energy le_rfl 1 one_pos _ (fun _ => 0)
    (fun s => by simpa using hasDerivAt_const s (0 : ℝ)) ω a).1.exists⟩

/-- **At a minimizer, the first variation is minimal on the support**: for
`β ≠ 0`, every point `x` of `supp μ` minimizes `δE/δμ[μ]` over the sphere.

`δE/δμ[μ] - ∫ δE/δμ[μ] dμ` is continuous, `≥ 0` on the sphere
(`integral_firstVariation_le`) and of integral `0`, so it vanishes
`μ`-almost everywhere; the set where it does not is open and `μ`-null, hence
outside the support. -/
theorem firstVariation_le_of_mem_support {β : ℝ} (hβ : β ≠ 0) {φ σ : ℝ → ℝ}
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    {μ : Perspective.ProbSphere d}
    (hμ : ∀ ν : Perspective.ProbSphere d, energy β φ ω a μ ≤ energy β φ ω a ν)
    {x : SSphere d} (hx : x ∈ (μ : Measure (SSphere d)).support) (y : SSphere d) :
    firstVariation β φ ω a μ (x : EucSpace d) ≤ firstVariation β φ ω a μ (y : EucSpace d) := by
  set c := ∫ x, firstVariation β φ ω a μ (x : EucSpace d) ∂(μ : Measure (SSphere d))
  have hle := integral_firstVariation_le β (continuous_of_hasDerivAt hφ) ω a hμ
  have hF : Continuous fun x : SSphere d => firstVariation β φ ω a μ (x : EucSpace d) :=
    (continuous_firstVariation hβ φ σ hφ ω a μ).comp continuous_subtype_val
  have hFc : Continuous fun x : SSphere d => firstVariation β φ ω a μ (x : EucSpace d) - c :=
    hF.sub continuous_const
  have hint : ∫ x, (firstVariation β φ ω a μ (x : EucSpace d) - c)
      ∂(μ : Measure (SSphere d)) = 0 := by
    rw [integral_sub (Perspective.integrable_of_continuous_compact hF _) (integrable_const c),
      integral_const, probReal_univ, one_smul, sub_self]
  have hae := (integral_eq_zero_iff_of_nonneg (fun z => sub_nonneg.mpr (hle z))
    (Perspective.integrable_of_continuous_compact hFc _)).mp hint
  have hnull : (μ : Measure (SSphere d)) {z | firstVariation β φ ω a μ (z : EucSpace d) ≠ c} = 0 :=
    measure_mono_null (fun z hz => by simpa [sub_eq_zero] using hz) (ae_iff.mp hae)
  have hxc : firstVariation β φ ω a μ (x : EucSpace d) = c := by
    by_contra hne
    exact Measure.subset_compl_support_of_isOpen (isOpen_ne_fun hF continuous_const) hnull hne hx
  exact hxc ▸ hle y

/-- The hypotheses of `firstVariation_le_of_mem_support` are satisfiable: `d = 1`,
`β = 1`, `σ = 0` with the primitive `φ = 0`, the minimizer of
`existsUnique_min_energy`, and a point of its support, which is nonempty. -/
example (ω : Idx 1 → ℝ) (a : Idx 1 → EucSpace 1) : (1 : ℝ) ≠ 0 ∧
    (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    ∃ μ : Perspective.ProbSphere 1, (∀ ν : Perspective.ProbSphere 1,
      energy 1 (fun _ => 0) ω a μ ≤ energy 1 (fun _ => 0) ω a ν) ∧
      (μ : Measure (SSphere 1)).support.Nonempty := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
    fun s => by simpa using hasDerivAt_const s (0 : ℝ)
  obtain ⟨μ, hμ, -⟩ := (existsUnique_min_energy le_rfl 1 one_pos _ _ hφ ω a).1
  exact ⟨one_ne_zero, hφ, μ, hμ, Measure.nonempty_support (IsProbabilityMeasure.ne_zero _)⟩

/-- **The minimizer is stationary**: for `β ≠ 0`, a global minimizer of the
coupled energy satisfies `eq: steady.state` at every point of its support.

At `x ∈ supp μ` the first variation is minimal over the sphere
(`firstVariation_le_of_mem_support`), so its tangential gradient vanishes
(`proj_eq_zero_of_isMin_sphere`), and that is `energyGrad` at `x`
(`energyGrad_eq_proj`).

Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`; see the module docstring
for the hypotheses. -/
theorem isStationary_of_isMin {β : ℝ} (hβ : β ≠ 0) {φ σ : ℝ → ℝ}
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    {μ : Perspective.ProbSphere d}
    (hμ : ∀ ν : Perspective.ProbSphere d, energy β φ ω a μ ≤ energy β φ ω a ν) :
    IsStationary β σ ω a μ := fun x hx => by
  rw [energyGrad_eq_proj]
  exact proj_eq_zero_of_isMin_sphere (norm_eq_of_mem_sphere x)
    (hasGradientAt_firstVariation hβ φ σ hφ ω a μ x) fun y hy =>
      firstVariation_le_of_mem_support hβ hφ ω a hμ hx ⟨y, mem_sphere_zero_iff_norm.mpr hy⟩

/-- The hypotheses of `isStationary_of_isMin` are satisfiable: `d = 1`, `β = 1`,
`σ = 0` with the primitive `φ = 0`, and the minimizer of
`existsUnique_min_energy`. -/
example (ω : Idx 1 → ℝ) (a : Idx 1 → EucSpace 1) : (1 : ℝ) ≠ 0 ∧
    (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    ∃ μ : Perspective.ProbSphere 1, ∀ ν : Perspective.ProbSphere 1,
      energy 1 (fun _ => 0) ω a μ ≤ energy 1 (fun _ => 0) ω a ν := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
    fun s => by simpa using hasDerivAt_const s (0 : ℝ)
  exact ⟨one_ne_zero, hφ, (existsUnique_min_energy le_rfl 1 one_pos _ _ hφ ω a).1.exists⟩

end Perceptron
end Transformer
