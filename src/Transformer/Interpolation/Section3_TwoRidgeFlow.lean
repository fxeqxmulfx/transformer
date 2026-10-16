/-
# Actual complete flow of two original ReLU neurons

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: neural.ode.sphere`,
and Appendix, the constant-control construction toward `lem: two.balls`.

Choose the actual spherical trajectory after proving its global existence.
It solves the original network at the two explicit hidden coordinates.
Uniqueness gives the autonomous flow law and inverse at every real time,
so each original flow map is bijective.

An actual equilibrium stays fixed, including the explicitly prescribed
target of the two-cap construction. Every point where both activations
are inactive is fixed as well. These properties use the original particle
equation; no flow or fixed-point premise is assumed.
-/

import Transformer.Interpolation.Section3_TwoRidgeExtension

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual complete two-neuron flow, chosen after proving existence.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
noncomputable def twoRidgeFlow (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) :
    ℝ → SSphere d := Classical.choose (exists_twoRidgeTrajectory p u b q v c x₀)

/-- The actual original flow starts at its prescribed initial point.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem twoRidgeFlow_zero (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) :
    twoRidgeFlow p u b q v c x₀ 0 = x₀ := (Classical.choose_spec (exists_twoRidgeTrajectory p u b q v c x₀)).1

/-- The selected spherical curve solves the original two-neuron field.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem hasDerivAt_twoRidgeFlow (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (twoRidgeFlow p u b q v c x₀ s : EucSpace d))
      (twoRidgeField p u b q v c (twoRidgeFlow p u b q v c x₀ t)) t :=
  (Classical.choose_spec (exists_twoRidgeTrajectory p u b q v c x₀)).2 t

/-- The actual curve solves the exact original network matrices.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem hasDerivAt_twoRidgeFlow_fullVF (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) (ρ : Perspective.ProbSphere d) (t : ℝ) :
    HasDerivAt (fun s => (twoRidgeFlow p u b q v c x₀ s : EucSpace d))
      (fullVF d (fun _ => twoRidgeParams i j p u b q v c) ρ t (twoRidgeFlow p u b q v c x₀ t)) t := by
  rw [fullVF_twoRidgeParams i j hij]
  exact hasDerivAt_twoRidgeFlow p u b q v c x₀ t

/-- The same actual path solves the complete ambient extension.
Auxiliary for arXiv:2411.04551v3, §1, the particle well-posedness argument. -/
theorem hasDerivAt_twoRidgeFlow_extension (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (twoRidgeFlow p u b q v c x₀ s : EucSpace d))
      (twoRidgeGlobalField p u b q v c (twoRidgeFlow p u b q v c x₀ t)) t := by
  rw [twoRidgeGlobalField_eq_of_unit _ _ _ _ _ _ _
    (mem_sphere_zero_iff_norm.mp (twoRidgeFlow p u b q v c x₀ t).2)]
  exact hasDerivAt_twoRidgeFlow p u b q v c x₀ t

/-- The original particle trajectory is continuous in time.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem continuous_twoRidgeFlow_time (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) :
    Continuous (twoRidgeFlow p u b q v c x₀) :=
  Continuous.subtype_mk (continuous_iff_continuousAt.mpr fun t =>
    (hasDerivAt_twoRidgeFlow p u b q v c x₀ t).continuousAt) _

/-- The actual original two-neuron flow composes in time.
Source: arXiv:2411.04551v3, §1, the autonomous characteristic flow. -/
theorem twoRidgeFlow_add (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) (s t : ℝ) :
    twoRidgeFlow p u b q v c x₀ (s + t) =
      twoRidgeFlow p u b q v c (twoRidgeFlow p u b q v c x₀ s) t := by
  have he := twoRidgeTrajectory_unique p u b q v c (fun r => twoRidgeFlow p u b q v c x₀ (s + r))
    (twoRidgeFlow p u b q v c (twoRidgeFlow p u b q v c x₀ s))
    (fun r => by simpa only [Function.comp_def, one_smul] using
      (hasDerivAt_twoRidgeFlow p u b q v c x₀ (s + r)).scomp r ((hasDerivAt_id r).const_add s))
    (hasDerivAt_twoRidgeFlow p u b q v c (twoRidgeFlow p u b q v c x₀ s))
    (by simp only [add_zero, twoRidgeFlow_zero])
  exact congrFun he t

/-- Reversing time inverts the actual original two-neuron flow.
Source: arXiv:2411.04551v3, §1, the invertible characteristic flow. -/
theorem twoRidgeFlow_inverse (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) (t : ℝ) :
    twoRidgeFlow p u b q v c (twoRidgeFlow p u b q v c x₀ t) (-t) = x₀ := by
  rw [← twoRidgeFlow_add, add_neg_cancel, twoRidgeFlow_zero]

/-- Each actual original two-neuron flow map is bijective.
Source: arXiv:2411.04551v3, §1, the invertible characteristic flow. -/
theorem bijective_twoRidgeFlow (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c t : ℝ) :
    Function.Bijective (fun x : SSphere d => twoRidgeFlow p u b q v c x t) := by
  have hl : Function.LeftInverse (fun x => twoRidgeFlow p u b q v c x (-t))
      (fun x => twoRidgeFlow p u b q v c x t) := fun x => twoRidgeFlow_inverse p u b q v c x t
  refine ⟨hl.injective, fun x => ⟨twoRidgeFlow p u b q v c x (-t), ?_⟩⟩
  simpa only [neg_neg] using twoRidgeFlow_inverse p u b q v c x (-t)

/-- An actual equilibrium is fixed by the complete original flow.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem twoRidgeFlow_eq_of_field_eq_zero (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x₀ : SSphere d) (hx₀ : twoRidgeField p u b q v c x₀ = 0) (t : ℝ) :
    twoRidgeFlow p u b q v c x₀ t = x₀ := by
  have hd (s : ℝ) : HasDerivAt (fun _ : ℝ => (x₀ : EucSpace d)) (twoRidgeField p u b q v c x₀) s := by
    rw [hx₀]
    exact hasDerivAt_const s _
  have he := twoRidgeTrajectory_unique p u b q v c (twoRidgeFlow p u b q v c x₀) (fun _ => x₀)
    (hasDerivAt_twoRidgeFlow p u b q v c x₀) hd (twoRidgeFlow_zero p u b q v c x₀)
  exact congrFun he t

/-- The actual original flow fixes every point with both neurons inactive.
Source: arXiv:2411.04551v3, Appendix, the fixed region in `lem: two.balls`. -/
theorem twoRidgeFlow_inactive (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x₀ : SSphere d) (hu : b - inner ℝ (x₀ : EucSpace d) u ≤ 0) (hv : c - inner ℝ (x₀ : EucSpace d) v ≤ 0)
    (t : ℝ) : twoRidgeFlow p u b q v c x₀ t = x₀ :=
  twoRidgeFlow_eq_of_field_eq_zero p u b q v c x₀ (twoRidgeField_inactive p u b q v c x₀ hu hv) t

/-- A set containing both activation regions is invariant under the
actual complete flow. Source: arXiv:2411.04551v3, Appendix,
the fixed-outside flow clause in `lem: two.balls`. -/
theorem twoRidgeFlow_mem_iff (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : Set (SSphere d)) (hS : ∀ x : SSphere d, x ∉ S →
      b - inner ℝ (x : EucSpace d) u ≤ 0 ∧ c - inner ℝ (x : EucSpace d) v ≤ 0)
    (x₀ : SSphere d) (t : ℝ) : twoRidgeFlow p u b q v c x₀ t ∈ S ↔ x₀ ∈ S := by
  constructor
  · intro ht
    by_contra hx
    rw [twoRidgeFlow_inactive p u b q v c x₀ (hS x₀ hx).1 (hS x₀ hx).2 t] at ht
    exact hx ht
  · intro hx
    by_contra ht
    have hh := hS (twoRidgeFlow p u b q v c x₀ t) ht
    have he := twoRidgeFlow_inactive p u b q v c (twoRidgeFlow p u b q v c x₀ t) hh.1 hh.2 (-t)
    rw [twoRidgeFlow_inverse] at he
    exact ht (he ▸ hx)

/-- The prescribed target is fixed by the actual constant two-neuron flow.
Auxiliary toward arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidgeFlow_target (z w ω : SSphere d) (c h ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) (t : ℝ) :
    twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c)
      ((inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)⁻¹ •
        (ℓ • (ω : EucSpace d) - (inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) • (z : EucSpace d)))
      (-(w : EucSpace d)) (-h) ω t = ω :=
  twoRidgeFlow_eq_of_field_eq_zero _ _ _ _ _ _ ω (twoRidgeField_target z w ω c h ℓ hα hβ) t

/-- The two actual coordinates, a nonzero equilibrium at the common
output axis, its inactive antipode and the target margins jointly realize
all coordinate, equilibrium, inactivity and target hypotheses. -/
example : ∃ a y : SSphere 2, (0 : Fin 2) ≠ 1 ∧
    twoRidgeField (a : EucSpace 2) (-(a : EucSpace 2)) 0 (a : EucSpace 2) (-(a : EucSpace 2)) 0 a = 0 ∧
    0 - inner ℝ (y : EucSpace 2) (-(a : EucSpace 2)) ≤ 0 ∧
    0 ≤ inner ℝ (a : EucSpace 2) (a : EucSpace 2) - 0 ∧
    0 < inner ℝ (a : EucSpace 2) (a : EucSpace 2) - 0 := by
  let a : SSphere 2 := basePoint 1
  refine ⟨a, antipode 2 a, by norm_num, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [a, antipode, basePoint, twoRidgeField, proj, EuclideanSpace.inner_single_left]

/-- The inactive-complement premise holds for the actual union of the
two activation regions, with arbitrary genuine neuron data. -/
example (u v : EucSpace 2) (b c : ℝ) : ∀ x : SSphere 2,
    x ∉ {y : SSphere 2 | 0 < b - inner ℝ (y : EucSpace 2) u ∨ 0 < c - inner ℝ (y : EucSpace 2) v} →
      b - inner ℝ (x : EucSpace 2) u ≤ 0 ∧ c - inner ℝ (x : EucSpace 2) v ≤ 0 := by
  intro x hx
  exact ⟨not_lt.mp (fun h => hx (Or.inl h)), not_lt.mp (fun h => hx (Or.inr h))⟩

end Transformer.Interpolation
