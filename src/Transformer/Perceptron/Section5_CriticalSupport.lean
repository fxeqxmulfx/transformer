module

public import Mathlib.Analysis.Calculus.Deriv.Inverse
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Topology.DiscreteSubset
public import Mathlib.Topology.Compactness.Compact
public import Mathlib.Topology.Order.Compact

/-!
# Isolated critical angles on a compact set

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron` and
`eq:strictSOPD-sufficient`. Its curvature condition makes the
second angular derivative of the first variation positive at
stationary support points. A positive derivative of the angular
stationarity function isolates its zeros, without requiring
continuity of its derivative on a neighbourhood.

This module proves that calculus and compactness step separately
from the interaction integral. It assumes the actual zero and
derivative conditions on a supplied function and set, and does
not define atomicity by those conditions. Nonempty examples at
the quadratic function realize every hypothesis.
-/

@[expose] public section

open Set Filter Topology

namespace Transformer.Perceptron

/-- **A zero with nonzero derivative is isolated among the zeros.**
Source: arXiv:2601.21366v2, §5.5, `eq:strictSOPD-sufficient`,
applied to the angular derivative of the first variation.
Only differentiability at the point is needed. -/
theorem zero_isolated_of_hasDerivAt (g : ℝ → ℝ) (x d : ℝ)
    (hd : HasDerivAt g d x) (hd0 : d ≠ 0) :
    ∃ U : Set ℝ, IsOpen U ∧ x ∈ U ∧ ∀ y ∈ U, g y = 0 → y = x := by
  have hne : ∀ᶠ y in 𝓝[≠] x, g y ≠ 0 := hd.eventually_ne hd0
  obtain ⟨U, hU, hneU⟩ := mem_nhdsWithin_iff_exists_mem_nhds_inter.mp hne
  obtain ⟨V, hVU, hVopen, hxV⟩ := mem_nhds_iff.mp hU
  refine ⟨V, hVopen, hxV, fun y hy hgy => ?_⟩
  by_contra hxy
  have hyn : y ∈ U ∩ ({x}ᶜ : Set ℝ) := ⟨hVU hy, by simpa using hxy⟩
  exact hneU hyn hgy

/-- The local-isolation inputs hold at the zero of `g(x)=2x`:
the actual derivative is two and is nonzero. -/
example : HasDerivAt (fun x : ℝ => 2 * x) 2 0 ∧ (2 : ℝ) ≠ 0 := by
  refine ⟨?_, by norm_num⟩
  simpa using (hasDerivAt_id (0 : ℝ)).const_mul 2

/-- **A compact set of zeros with nonzero derivatives is finite.**
Source: arXiv:2601.21366v2, §5.5, the angular stationarity condition
and `eq:strictSOPD-sufficient`. The derivative witnesses are supplied
pointwise; no stronger uniform or global smoothness is assumed. -/
theorem finite_zero_set_of_nonzero_derivative (g : ℝ → ℝ) (S : Set ℝ)
    (hS : IsCompact S) (hzero : ∀ x ∈ S, g x = 0)
    (hderiv : ∀ x ∈ S, ∃ d : ℝ, HasDerivAt g d x ∧ d ≠ 0) : S.Finite := by
  apply hS.finite
  apply isDiscrete_iff_forall_mem_exists_isOpen.mpr
  intro x hx
  obtain ⟨d, hd, hd0⟩ := hderiv x hx
  obtain ⟨U, hUopen, hxU, hU⟩ := zero_isolated_of_hasDerivAt g x d hd hd0
  refine ⟨U, hUopen, ?_⟩
  ext y
  constructor
  · rintro ⟨hyU, hyS⟩
    exact mem_singleton_iff.mpr (hU y hyU (hzero y hyS))
  · intro hy
    have hxy : y = x := mem_singleton_iff.mp hy
    subst y
    exact ⟨hxU, hx⟩

/-- Every input of the compact-zero theorem is realized by `S={0}`
and the actual affine function `g(x)=2x`. -/
example : IsCompact ({0} : Set ℝ) ∧
    (∀ x ∈ ({0} : Set ℝ), (fun y : ℝ => 2 * y) x = 0) ∧
    (∀ x ∈ ({0} : Set ℝ), ∃ d : ℝ,
      HasDerivAt (fun y : ℝ => 2 * y) d x ∧ d ≠ 0) := by
  refine ⟨isCompact_singleton, ?_, ?_⟩
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    norm_num
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    refine ⟨2, ?_, by norm_num⟩
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul 2

/-- **A compact set of critical angles with positive second derivative
is finite.** Source: arXiv:2601.21366v2, §5.5,
`rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
The second derivative is Mathlib's actual iterated derivative.
Its nonzero value entails differentiability of the first derivative,
since the totalized derivative is zero at a nondifferentiable point. -/
theorem finite_critical_set_of_positive_second_derivative (f : ℝ → ℝ) (S : Set ℝ)
    (hS : IsCompact S) (hcrit : ∀ x ∈ S, deriv f x = 0)
    (hcurv : ∀ x ∈ S, 0 < deriv^[2] f x) : S.Finite := by
  apply finite_zero_set_of_nonzero_derivative (deriv f) S hS hcrit
  intro x hx
  have hp := hcurv x hx
  change 0 < deriv (deriv f) x at hp
  exact ⟨deriv (deriv f) x, (differentiableAt_of_deriv_ne_zero hp.ne').hasDerivAt, hp.ne'⟩

/-- The actual second derivative in the quadratic witnesses is two.
This is the elementary calculus example for the curvature condition
of arXiv:2601.21366v2, §5.5, `eq:strictSOPD-sufficient`. -/
theorem critical_support_quadratic_second_derivative (x : ℝ) :
    deriv^[2] (fun y : ℝ => y ^ 2) x = 2 := by
  have hfirst : deriv (fun y : ℝ => y ^ 2) = fun y => 2 * y := by
    funext y
    simp
  change deriv (deriv (fun y : ℝ => y ^ 2)) x = 2
  rw [hfirst]
  simp

/-- The positive-curvature hypotheses are simultaneously realized
on the nonempty compact critical set of `f(x)=x²`. -/
example : IsCompact ({0} : Set ℝ) ∧
    (∀ x ∈ ({0} : Set ℝ), deriv (fun y : ℝ => y ^ 2) x = 0) ∧
    (∀ x ∈ ({0} : Set ℝ), 0 < deriv^[2] (fun y : ℝ => y ^ 2) x) := by
  refine ⟨isCompact_singleton, ?_, ?_⟩
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    simp
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    rw [critical_support_quadratic_second_derivative]
    norm_num

/-- **The same finiteness conclusion for closed, bounded angular sets.**
Source: arXiv:2601.21366v2, §5.5, `eq:strictSOPD-sufficient`.
An angular preimage of stationary support is closed and is restricted
to a compact interval of representatives. This packages exactly that
topological input without assuming that the support is already finite. -/
theorem finite_bounded_critical_set (f : ℝ → ℝ) (S : Set ℝ) (a b : ℝ)
    (hclosed : IsClosed S) (hbounded : S ⊆ Icc a b)
    (hcrit : ∀ x ∈ S, deriv f x = 0)
    (hcurv : ∀ x ∈ S, 0 < deriv^[2] f x) : S.Finite := by
  have hcompact : IsCompact S := isCompact_Icc.of_isClosed_subset hclosed hbounded
  exact finite_critical_set_of_positive_second_derivative f S hcompact hcrit hcurv

/-- The bounded-set variant has all its inputs at `S={0}` inside
`[-1,1]` for the quadratic function. -/
example : IsClosed ({0} : Set ℝ) ∧ ({0} : Set ℝ) ⊆ Icc (-1 : ℝ) 1 ∧
    (∀ x ∈ ({0} : Set ℝ), deriv (fun y : ℝ => y ^ 2) x = 0) ∧
    (∀ x ∈ ({0} : Set ℝ), 0 < deriv^[2] (fun y : ℝ => y ^ 2) x) := by
  refine ⟨isClosed_singleton, ?_, ?_, ?_⟩
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    norm_num
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    simp
  · intro x hx
    rcases mem_singleton_iff.mp hx with rfl
    rw [critical_support_quadratic_second_derivative]
    norm_num

end Transformer.Perceptron
