/-
# Perceptrons and attention's mean-field landscape — the minimizer

Formalization of `prop: min.max` (ii) of arXiv:2601.21366v2: the coupled
energy `E_{β,ϑ}[μ] = 𝖤_β[μ] + ½ ∫ v_ϑ dμ` has a unique global minimizer `μ⋆`,
and `μ⋆` is invariant under the isometries that fix the weights.  Part (i),
the maximizers, is `MinMax`.

**What the source says and what is carried here.**

* Existence is the compactness of `𝒫(𝕊^{d-1})`, as in the source.  Uniqueness
  is the strict convexity of `𝖤_β`, to which the linear term `½ ∫ v_ϑ dμ` adds
  nothing: the source reads it off the spherical-harmonic expansion of
  `e^{β x·y}` (through Bilyk et al.), this proof off its monomial expansion, at
  the midpoint `(μ₀ + μ₁)/2` (`Perspective.Section2_EnergyConvex`).  Both are
  `Perspective.existsUnique_isMin_interactionEnergy_add` with `V = ½ v_ϑ`.

* "Rotations" is read as all of `O(d)`, the linear isometries of `ℝ^d`, as in
  the source's proof ("`R ∈ O(d)`"); with the reflections included the
  statement is the stronger one.  The proof asks `R a_j = a_j` of every `j`,
  the statement only of those with `ω_j ≠ 0`, and it is the statement that is
  carried: a neuron with `ω_j = 0` does not enter `v_ϑ`.

* The source's standing `d ≥ 2` is relaxed to `d ≥ 1`: on `𝕊^0 = {±1}` the
  argument is the same.  At `d = 0` the sphere is empty, `𝒫(𝕊^{-1}) = ∅`, and
  there is no minimizer.

Source: arXiv:2601.21366v2, `prop: min.max` (ii) and its proof.
-/

import Transformer.Perceptron.MinMax
import Transformer.Perspective.Section2_EnergyMin

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- `E_{β,ϑ}[μ] = 𝖤_β[μ] + ∫ ½ v_ϑ dμ`, the form
`Perspective.Section2_EnergyConvex` works with.

Source: arXiv:2601.21366v2, proof of `prop: min.max`, first display. -/
theorem energy_eq_interactionEnergy_add (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    energy β φ ω a μ = Perspective.interactionEnergy d β μ +
      ∫ x, (2 : ℝ)⁻¹ * potential φ ω a (x : EucSpace d) ∂(μ : Measure (SSphere d)) := by
  rw [energy, integral_const_mul]

/-- **`v_ϑ(Ux) = v_ϑ(x)`** for every linear isometry `U` fixing every `a_j`
with `ω_j ≠ 0`: `a_j · Ux = Ua_j · Ux = a_j · x`, and a neuron with `ω_j = 0`
does not contribute.

Source: arXiv:2601.21366v2, proof of `prop: min.max` (ii). -/
theorem potential_map_eq (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (hU : ∀ j : Idx d, ω j ≠ 0 → U (a j) = a j)
    (x : EucSpace d) : potential φ ω a (U x) = potential φ ω a x := by
  unfold potential
  refine Finset.sum_congr rfl fun j _ => ?_
  by_cases hj : ω j = 0
  · simp [hj]
  · rw [← U.inner_map_map (a j) x, hU j hj]

/-- The hypothesis of `potential_map_eq` is satisfiable: `U = id`. -/
example (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    ∀ j : Idx d, ω j ≠ 0 → LinearIsometryEquiv.refl ℝ (EucSpace d) (a j) = a j :=
  fun _ _ => rfl

/-- **Proposition (prop: min.max) (ii).**  For `d ≥ 1` and `β > 0`, the coupled
energy has a unique global minimizer `μ⋆` over `𝒫(𝕊^{d-1})`, and `μ⋆` is
invariant under every linear isometry of `ℝ^d` fixing every `a_j` with
`ω_j ≠ 0`.

`E_{β,ϑ} = 𝖤_β + ∫ ½ v_ϑ dμ` with `½ v_ϑ` continuous
(`energy_eq_interactionEnergy_add`), so existence and uniqueness are
`Perspective.existsUnique_isMin_interactionEnergy_add`; `½ v_ϑ` is invariant
under those isometries (`potential_map_eq`), so the invariance is
`Perspective.map_sphereMap_eq_of_isMin`.

The source assumes `d ≥ 2` and says "rotations"; see the module docstring for
`d ≥ 1` and for `O(d)`.

Source: arXiv:2601.21366v2, `prop: min.max` (ii). -/
theorem existsUnique_min_energy (hd : 1 ≤ d) (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    (∃! μ₀ : Perspective.ProbSphere d,
      ∀ μ : Perspective.ProbSphere d, energy β φ ω a μ₀ ≤ energy β φ ω a μ) ∧
    (∀ μ₀ : Perspective.ProbSphere d,
      (∀ μ : Perspective.ProbSphere d, energy β φ ω a μ₀ ≤ energy β φ ω a μ) →
      ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, (∀ j : Idx d, ω j ≠ 0 → U (a j) = a j) →
        (μ₀ : Measure (SSphere d)).map (Perspective.sphereMap d U)
          = (μ₀ : Measure (SSphere d))) := by
  have hV : Continuous fun x : SSphere d => (2 : ℝ)⁻¹ * potential φ ω a (x : EucSpace d) :=
    continuous_const.mul ((continuous_potential (continuous_of_hasDerivAt hφ) ω a).comp
      continuous_subtype_val)
  simp_rw [energy_eq_interactionEnergy_add]
  exact ⟨Perspective.existsUnique_isMin_interactionEnergy_add β hβ hd hV,
    fun μ₀ h₀ U hU => Perspective.map_sphereMap_eq_of_isMin β hβ hV h₀ U fun x =>
      congrArg (fun t => (2 : ℝ)⁻¹ * t) (potential_map_eq φ ω a U hU (x : EucSpace d))⟩

/-- The hypotheses of `existsUnique_min_energy` are satisfiable: `d = 1`,
`β = 1`, and `σ = 0` with the primitive `φ = 0`. -/
example : 1 ≤ 1 ∧ (0 : ℝ) < 1 ∧
    ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
  ⟨le_rfl, one_pos, fun s => by simpa using hasDerivAt_const s (0 : ℝ)⟩

end Perceptron
end Transformer
