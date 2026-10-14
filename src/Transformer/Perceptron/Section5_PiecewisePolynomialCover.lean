/-
# Finite original polynomial branches, including their boundary values

ArXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
The source's finitely many activation breakpoints divide the real line
into finitely many open intervals. Every actual interval is specified by
its strict comparisons with the original breakpoints. Empty comparison
patterns cause no difficulty.

The closures of these actual intervals cover the entire original line:
they are a finite closed cover of the dense breakpoint complement.
Continuity extends the actual polynomial identities to these closures.
Thus finitely many actual polynomial values cover the original activation
at every real input, including every original breakpoint. No global
polynomial or analyticity assertion about the activation is assumed.
-/

import Transformer.Perceptron.TransformPoly
import Mathlib.Topology.Algebra.Polynomial
import Mathlib.Analysis.Convex.PathConnected

open Real Set
open scoped NNReal

namespace Transformer.Perceptron

/-- The original activation is polynomial on each open interval avoiding
its finitely many original breakpoints. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`. This is a predicate of the actual activation. -/
def IsPiecewisePolynomial (σ : ℝ → ℝ) : Prop :=
  ∃ B : Finset ℝ, ∀ I : Set ℝ, IsOpen I → IsPreconnected I → Disjoint I (B : Set ℝ) →
    ∃ P : Polynomial ℝ, ∀ s ∈ I, σ s = P.eval s

/-- The original ReLU has the single actual breakpoint zero.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem isPiecewisePolynomial_relu : IsPiecewisePolynomial (fun s : ℝ => max s 0) := by
  refine ⟨{0}, fun I _ hconn hdisj => ?_⟩
  have h0 : (0 : ℝ) ∉ I := fun h => (Set.disjoint_left.mp hdisj h) (by simp)
  rcases Set.eq_empty_or_nonempty I with hI | ⟨s₀, hs₀⟩
  · exact ⟨0, fun s hs => absurd (hI ▸ hs) (Set.notMem_empty s)⟩
  rcases lt_or_gt_of_ne (show s₀ ≠ 0 from fun h => h0 (h ▸ hs₀)) with hneg | hpos
  · refine ⟨0, fun s hs => ?_⟩
    have hs0 : s < 0 := by
      by_contra hge
      exact h0 (hconn.Icc_subset hs₀ hs ⟨hneg.le, not_lt.mp hge⟩)
    simp [max_eq_right hs0.le]
  · refine ⟨Polynomial.X, fun s hs => ?_⟩
    have hs0 : 0 < s := by
      by_contra hle
      exact h0 (hconn.Icc_subset hs hs₀ ⟨not_lt.mp hle, hpos.le⟩)
    simp [max_eq_left hs0.le]

/-- The original ReLU is globally Lipschitz with constant one.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem lipschitzWith_relu : LipschitzWith 1 (fun s : ℝ => max s 0) := LipschitzWith.id.max_const 0

/-- The actual open comparison interval of one original breakpoint
pattern. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
True selects the original left interval, false the original right one. -/
def polynomialIntervalCell (B : Finset ℝ) (p : B → Bool) : Set ℝ :=
  ⋂ b : B, if p b = true then Iio (b : ℝ) else Ioi (b : ℝ)

/-- Each original comparison interval is open.
Source: arXiv:2601.21366v2, §5.2, the components in `rem:piecewise-poly`. -/
theorem isOpen_polynomialIntervalCell (B : Finset ℝ) (p : B → Bool) :
    IsOpen (polynomialIntervalCell B p) := by
  apply isOpen_iInter_of_finite
  intro b
  split_ifs
  · exact isOpen_Iio
  · exact isOpen_Ioi

/-- Each original comparison interval is preconnected, including an
empty one. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem isPreconnected_polynomialIntervalCell (B : Finset ℝ) (p : B → Bool) :
    IsPreconnected (polynomialIntervalCell B p) := by
  have hc : Convex ℝ (polynomialIntervalCell B p) := by
    apply convex_iInter
    intro b
    split_ifs
    · exact convex_Iio _
    · exact convex_Ioi _
  exact hc.isPreconnected

/-- Each original strict comparison interval avoids every original
breakpoint. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem disjoint_polynomialIntervalCell (B : Finset ℝ) (p : B → Bool) :
    Disjoint (polynomialIntervalCell B p) (B : Set ℝ) := by
  apply Set.disjoint_left.mpr
  intro x hx hxB
  have hb := mem_iInter.mp hx (⟨x, hxB⟩ : B)
  split_ifs at hb
  · exact lt_irrefl x hb
  · exact lt_irrefl x hb

/-- Every original non-breakpoint belongs to its actual strict
comparison interval. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem exists_mem_polynomialIntervalCell (B : Finset ℝ) {x : ℝ} (hx : x ∉ B) :
    ∃ p : B → Bool, x ∈ polynomialIntervalCell B p := by
  classical
  refine ⟨fun b => decide (x < (b : ℝ)), mem_iInter.mpr fun b => ?_⟩
  by_cases hlt : x < (b : ℝ)
  · rw [ite_eq_left (by simp [hlt])]
    exact hlt
  · rw [ite_eq_right (by simp [hlt])]
    have hne : x ≠ (b : ℝ) := fun h => hx (h.symm ▸ b.2)
    exact (lt_or_gt_of_ne hne).resolve_left hlt

/-- The closures of the finitely many actual comparison intervals cover
the entire original line, including the original breakpoints.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Their finite closed union contains the dense breakpoint complement. -/
theorem exists_mem_polynomialIntervalCell_closure (B : Finset ℝ) (x : ℝ) :
    ∃ p : B → Bool, x ∈ closure (polynomialIntervalCell B p) := by
  classical
  have hd : Dense (B : Set ℝ)ᶜ := by
    have heq : univ \ (B : Set ℝ) = (B : Set ℝ)ᶜ := by ext y; simp
    rw [← heq]
    exact (dense_univ : Dense (univ : Set ℝ)).sdiff_finite B.finite_toSet
  have hsub : (B : Set ℝ)ᶜ ⊆ ⋃ p : B → Bool, closure (polynomialIntervalCell B p) := by
    intro y hy
    obtain ⟨p, hp⟩ := exists_mem_polynomialIntervalCell B hy
    exact mem_iUnion.mpr ⟨p, subset_closure hp⟩
  have hcl := closure_minimal hsub (isClosed_iUnion_of_finite (fun _ => isClosed_closure))
  rw [hd.closure_eq] at hcl
  exact mem_iUnion.mp (hcl (mem_univ x))

/-- A continuous original piecewise-polynomial activation has a finite
set of actual polynomial candidates whose values cover every original
input. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Continuity proves agreement at the original boundaries; the source's
global Lipschitz condition suffices for this hypothesis. -/
theorem exists_finset_polynomial_cover (σ : ℝ → ℝ) (hσ : Continuous σ)
    (hpp : IsPiecewisePolynomial σ) :
    ∃ S : Finset (Polynomial ℝ), ∀ x : ℝ, ∃ P ∈ S, σ x = P.eval x := by
  classical
  obtain ⟨B, hB⟩ := hpp
  choose P hP using fun p : B → Bool => hB (polynomialIntervalCell B p)
    (isOpen_polynomialIntervalCell B p) (isPreconnected_polynomialIntervalCell B p)
    (disjoint_polynomialIntervalCell B p)
  refine ⟨Finset.univ.image P, fun x => ?_⟩
  obtain ⟨p, hp⟩ := exists_mem_polynomialIntervalCell_closure B x
  refine ⟨P p, Finset.mem_image.mpr ⟨p, Finset.mem_univ p, rfl⟩, ?_⟩
  exact closure_minimal (hP p) (isClosed_eq hσ (P p).continuous) hp

/-- All actual continuity, piecewise-polynomial and global Lipschitz
inputs hold jointly for the original ReLU activation, including at its
actual breakpoint zero. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
example : Continuous (fun s : ℝ => max s 0) ∧ IsPiecewisePolynomial (fun s : ℝ => max s 0) ∧
    LipschitzWith 1 (fun s : ℝ => max s 0) ∧ (1 : ℝ) ∉ ({0} : Finset ℝ) :=
  ⟨lipschitzWith_relu.continuous, isPiecewisePolynomial_relu, lipschitzWith_relu, by norm_num⟩

end Transformer.Perceptron
