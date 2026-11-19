module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Basic

/-!
# Exact zero rows in complete complex component contractions

Source: `review/convex/moments.md`, CM94. The native complete gradient has one
real entry for every original output/query/context coordinate and the exact
imaginary partner from CM88. A row whose entries are all exactly zero makes
no contribution to any complex component contraction. All factors and gradient
coordinates remain stored; this is an execution identity, not rank pruning or
an approximate minimization guarantee.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Support

open scoped BigOperators

variable {I O : Type*} [Fintype I] [Fintype O]

/-- Complete rows with at least one nonzero original-class entry, CM94.
There is no magnitude threshold or selected subset of classes. -/
noncomputable def active (g : I → O → ℝ) : Finset I := by
  classical
  exact Finset.univ.filter (fun i => ∃ c, g i c ≠ 0)

/-- Skipping only exactly zero complete rows preserves every complex
contraction, CM94. The arbitrary complex multiplier covers output, query and
context contractions and the component objective, including imaginary entries. -/
theorem complex_sum (g : I → O → ℝ) (v : I → O → ℂ) :
    (∑ i ∈ active g, ∑ c, (g i c : ℂ) * v i c) =
      ∑ i, ∑ c, (g i c : ℂ) * v i c := by
  classical
  unfold active
  apply Finset.sum_filter_of_ne
  intro i hi hn
  by_contra hz
  have hg : ∀ c, g i c = 0 := by simpa only [not_exists, not_not] using hz
  exact hn (by simp [hg])

/-- The actual learned complex joint objective is unchanged by exact-row
execution, CM94. The complete factor domain and every class stay unchanged. -/
theorem component_value {T : ℕ}
    (g : QueryCoordinate O × ContextCoordinate O T → O → ℝ) (f : Atom O T) :
    (∑ i ∈ active g, ∑ c, (g i c : ℂ) *
      ((1 + Complex.I) * f.output c * f.query i.1 * f.context i.2)).re =
    (∑ i, ∑ c, (g i c : ℂ) *
      ((1 + Complex.I) * f.output c * f.query i.1 * f.context i.2)).re := by
  exact congrArg Complex.re (complex_sum g
    (fun i c => (1 + Complex.I) * f.output c * f.query i.1 * f.context i.2))

/-- An inhabited nonzero complex control with one absent and one active
row, CM94; it is an execution check rather than stress-test evidence. -/
theorem nonzero_control :
    (∑ i ∈ active (fun i c : Bool => if i then if c then (2 : ℝ) else -1 else 0),
      ∑ c : Bool, Complex.ofReal (if i then if c then (2 : ℝ) else -1 else 0) *
        (if c then Complex.I else 1)) = -1 + 2 * Complex.I := by
  rw [complex_sum]
  norm_num [Fintype.sum_bool]
  ring

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Support
