/-
# Perceptrons and attention's mean-field landscape — the extremal points

Formalization of `prop: min.max` of arXiv:2601.21366v2: the global maximizers
of the coupled energy are exactly the Dirac masses at the maximizers of the
perceptron potential, and the global minimizer is unique and inherits the
symmetries of the weights.

**What the source says and what is carried here.**

* "the global maximizers are exactly those `μ = δ_x` with
  `x ∈ argmax_y Σ_j ω_j φ(a_j · y)`" is carried as two theorems, one per
  inclusion, both proved.  `argmax` over `𝕊^{d-1}` is written out as
  `∀ y : 𝕊^{d-1}, v_ϑ(y) ≤ v_ϑ(x)`; that the maximum is attained at all is
  part of the proof, not a hypothesis — `v_ϑ` is continuous, because `φ` is
  differentiable, and the sphere is compact.

* The maximizer half needs only `β > 0`, not `d ≥ 2`: the sphere being
  nonempty is what the argument uses, and a probability measure on it
  witnesses that.  The standing `d ≥ 2` of the source is therefore dropped
  here, which strengthens the statement.

* The interaction half of both directions is the theorem of arXiv:2312.10794v5
  already in this tree — `Perspective.interactionEnergy_le` and
  `Perspective.exists_eq_dirac_of_isMaxEnergy`.  Coupling adds `∫ v_ϑ dμ`,
  which is maximized by the same Dirac masses, so the two maximizations do not
  compete.

Source: arXiv:2601.21366v2, `prop: min.max`.
-/

import Transformer.Perceptron.Basic
import Transformer.Perspective.Section2_EnergyMax

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Continuity and integrability of the potential -/

/-- A primitive of `2σ` is continuous: it is differentiable everywhere. -/
theorem continuous_of_hasDerivAt {φ σ : ℝ → ℝ} (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) :
    Continuous φ :=
  continuous_iff_continuousAt.mpr fun s => (hφ s).continuousAt

/-- `v_ϑ` is continuous whenever `φ` is. -/
theorem continuous_potential {φ : ℝ → ℝ} (hφc : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) : Continuous (potential φ ω a) :=
  continuous_finsetSum _ fun _ _ =>
    continuous_const.mul (hφc.comp (continuous_const.inner continuous_id))

/-- `v_ϑ` is integrable against every probability measure on the sphere: it is
continuous, and the sphere is compact. -/
theorem integrable_potential {φ : ℝ → ℝ} (hφc : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    Integrable (fun y : SSphere d => potential φ ω a (y : EucSpace d))
      (μ : Measure (SSphere d)) :=
  ((continuous_potential hφc ω a).comp continuous_subtype_val).integrable_of_hasCompactSupport
    (HasCompactSupport.of_compactSpace _)

/-- The coupled energy of a Dirac mass:
`E_{β,ϑ}[δ_x] = (2β)⁻¹ e^β + (1/2) v_ϑ(x)`. -/
theorem energy_diracProb (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : SSphere d) :
    energy β φ ω a (Perspective.diracProb d x)
      = (2 * β)⁻¹ * Real.exp β + (2 : ℝ)⁻¹ * potential φ ω a (x : EucSpace d) := by
  rw [energy, Perspective.interactionEnergy_diracProb]
  congr 1
  rw [show ((Perspective.diracProb d x : Perspective.ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x from rfl, integral_dirac]

/-! ### The maximizers -/

/-- **Proposition (prop: min.max) (i), the easy inclusion.**  If `x` maximizes
the perceptron potential on the sphere, then `δ_x` maximizes the coupled
energy: it maximizes the interaction term, as every Dirac mass does, and the
potential term at the same time.

Source: arXiv:2601.21366v2, `prop: min.max` (i). -/
theorem isMaxEnergy_diracProb_of_isMaxOn (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : SSphere d)
    (hx : ∀ y : SSphere d, potential φ ω a (y : EucSpace d) ≤ potential φ ω a (x : EucSpace d))
    (μ : Perspective.ProbSphere d) :
    energy β φ ω a μ ≤ energy β φ ω a (Perspective.diracProb d x) := by
  have hφc : Continuous φ := continuous_of_hasDerivAt hφ
  have hI : Perspective.interactionEnergy d β μ ≤ (2 * β)⁻¹ * Real.exp β :=
    Perspective.interactionEnergy_le d β hβ μ
  have hv : ∫ y, potential φ ω a (y : EucSpace d) ∂(μ : Measure (SSphere d))
      ≤ potential φ ω a (x : EucSpace d) := by
    calc ∫ y, potential φ ω a (y : EucSpace d) ∂(μ : Measure (SSphere d))
        ≤ ∫ _ : SSphere d, potential φ ω a (x : EucSpace d) ∂(μ : Measure (SSphere d)) :=
          integral_mono (integrable_potential hφc ω a μ) (integrable_const _) hx
      _ = potential φ ω a (x : EucSpace d) := by simp
  rw [energy_diracProb, energy]
  linarith

/-- The hypotheses of `isMaxEnergy_diracProb_of_isMaxOn` are satisfiable: the
vanishing perceptron `ω = 0`, whose potential is constant, with `φ = 0` a
primitive of `2σ` for `σ = 0`. -/
example (a : Idx d → EucSpace d) (x : SSphere d) :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
      ∀ y : SSphere d, potential (fun _ : ℝ => (0 : ℝ)) (0 : Idx d → ℝ) a (y : EucSpace d)
        ≤ potential (fun _ : ℝ => (0 : ℝ)) (0 : Idx d → ℝ) a (x : EucSpace d) :=
  ⟨one_pos, fun s => by simpa using hasDerivAt_const s (0 : ℝ), fun _ => by simp [potential]⟩

/-- **Proposition (prop: min.max) (i), the converse inclusion.**  Every global
maximizer of the coupled energy is a Dirac mass at a maximizer of the
perceptron potential.

Both terms of `E_{β,ϑ}` are bounded by their values at `δ_{x⋆}`, `x⋆` a
maximizer of `v_ϑ`, so a maximizer attains both bounds at once: the first
forces a Dirac mass by `Perspective.exists_eq_dirac_of_isMaxEnergy`, and the
second then says its atom maximizes `v_ϑ`.

Source: arXiv:2601.21366v2, `prop: min.max` (i). -/
theorem exists_isMaxOn_potential_of_isMaxEnergy (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d)
    (hmax : ∀ ν : Perspective.ProbSphere d, energy β φ ω a ν ≤ energy β φ ω a μ) :
    ∃ x : SSphere d,
      (∀ y : SSphere d,
        potential φ ω a (y : EucSpace d) ≤ potential φ ω a (x : EucSpace d)) ∧
      (μ : Measure (SSphere d)) = Measure.dirac x := by
  have hφc : Continuous φ := continuous_of_hasDerivAt hφ
  -- The sphere is nonempty: it carries the probability measure `μ`.
  obtain ⟨z⟩ : Nonempty (SSphere d) := by
    by_contra hcon
    rw [not_nonempty_iff] at hcon
    have h1 : (μ : Measure (SSphere d)) Set.univ = 1 := measure_univ
    rw [Set.univ_eq_empty_iff.mpr hcon, measure_empty] at h1
    exact zero_ne_one h1
  -- The potential attains its maximum on the compact sphere.
  obtain ⟨xs, -, hxs⟩ :=
    isCompact_univ.exists_isMaxOn ⟨z, Set.mem_univ z⟩
      (((continuous_potential hφc ω a).comp continuous_subtype_val).continuousOn)
  have hxs' : ∀ y : SSphere d,
      potential φ ω a (y : EucSpace d) ≤ potential φ ω a (xs : EucSpace d) :=
    fun y => hxs (Set.mem_univ y)
  -- Both terms of the energy are bounded by their values at `δ_{x⋆}`.
  have hI : Perspective.interactionEnergy d β μ ≤ (2 * β)⁻¹ * Real.exp β :=
    Perspective.interactionEnergy_le d β hβ μ
  have hv : ∫ y, potential φ ω a (y : EucSpace d) ∂(μ : Measure (SSphere d))
      ≤ potential φ ω a (xs : EucSpace d) := by
    calc ∫ y, potential φ ω a (y : EucSpace d) ∂(μ : Measure (SSphere d))
        ≤ ∫ _ : SSphere d, potential φ ω a (xs : EucSpace d) ∂(μ : Measure (SSphere d)) :=
          integral_mono (integrable_potential hφc ω a μ) (integrable_const _) hxs'
      _ = potential φ ω a (xs : EucSpace d) := by simp
  have hge := hmax (Perspective.diracProb d xs)
  rw [energy_diracProb, energy] at hge
  -- A maximizer attains both bounds.
  have hItop : Perspective.interactionEnergy d β μ = (2 * β)⁻¹ * Real.exp β := by linarith
  have hvtop : ∫ y, potential φ ω a (y : EucSpace d) ∂(μ : Measure (SSphere d))
      = potential φ ω a (xs : EucSpace d) := by linarith
  -- The first makes `μ` a maximizer of the interaction energy, hence a Dirac mass.
  obtain ⟨x₀, hx₀⟩ := Perspective.exists_eq_dirac_of_isMaxEnergy d β hβ μ (fun ν => by
    rw [hItop]; exact Perspective.interactionEnergy_le d β hβ ν)
  -- The second says its atom maximizes the potential.
  refine ⟨x₀, fun y => ?_, hx₀⟩
  rw [hx₀, integral_dirac] at hvtop
  rw [hvtop]
  exact hxs' y

/-- The hypotheses of `exists_isMaxOn_potential_of_isMaxEnergy` are
satisfiable: for the vanishing perceptron the Dirac mass at any point of the
sphere is a global maximizer, by `isMaxEnergy_diracProb_of_isMaxOn`. -/
example (a : Idx d → EucSpace d) (x : SSphere d) :
    ∀ ν : Perspective.ProbSphere d,
      energy 1 (fun _ : ℝ => (0 : ℝ)) (0 : Idx d → ℝ) a ν
        ≤ energy 1 (fun _ : ℝ => (0 : ℝ)) (0 : Idx d → ℝ) a (Perspective.diracProb d x) :=
  isMaxEnergy_diracProb_of_isMaxOn 1 one_pos _ 0
    (fun s => by simpa using hasDerivAt_const s (0 : ℝ)) 0 a x (fun _ => by simp [potential])

/-! ### The minimizer -/

/-- **Proposition (prop: min.max) (ii).**  For `d ≥ 2` and `β > 0`, the coupled
energy has a unique global minimizer `μ⋆`, and `μ⋆` is invariant under every
rotation fixing every `a_j` with `ω_j ≠ 0`.

Not proved here.  Even for `ω = 0` this is
`Perspective.existence_uniqueness_energy_min`, which rests on the positive
definiteness of the kernel `e^{β⟨x,y⟫}` on the sphere — its Funk–Hecke
expansion in Gegenbauer polynomials — and Mathlib has neither.

Source: arXiv:2601.21366v2, `prop: min.max` (ii). -/
theorem existsUnique_min_energy (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    (∃! μ₀ : Perspective.ProbSphere d,
      ∀ μ : Perspective.ProbSphere d, energy β φ ω a μ₀ ≤ energy β φ ω a μ) ∧
    (∀ μ₀ : Perspective.ProbSphere d,
      (∀ μ : Perspective.ProbSphere d, energy β φ ω a μ₀ ≤ energy β φ ω a μ) →
      ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, (∀ j : Idx d, ω j ≠ 0 → U (a j) = a j) →
        (μ₀ : Measure (SSphere d)).map (Perspective.sphereMap d U)
          = (μ₀ : Measure (SSphere d))) := by
  sorry

/-- The hypotheses of `existsUnique_min_energy` are satisfiable: `d = 2`,
`β = 1`, and `σ = 0` with the primitive `φ = 0`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧
    ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
  ⟨le_rfl, one_pos, fun s => by simpa using hasDerivAt_const s (0 : ℝ)⟩

end Perceptron
end Transformer
