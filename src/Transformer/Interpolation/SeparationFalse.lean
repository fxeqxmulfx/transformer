/-
# Measure-to-measure interpolation — `prop: separation` fails for equal data

`prop: separation` of arXiv:2411.04551v3 asks for the solutions started at the
`μ_0^i` to end with pairwise disjoint geodesic hulls of their supports.  Two
equal initial measures have the same solutions, and a hull disjoint from
itself is empty, which the support of a probability measure is not.  So the
proposition as printed is false in every dimension `d ≥ 1`, for every
parameter curve and every number of switches; the statement kept in
`Transformer.Interpolation.Disentanglement` adds that the `μ_0^i` are pairwise
distinct.

Source: arXiv:2411.04551v3, §3, `prop: separation`.
-/

import Transformer.Interpolation.Disentanglement
import Transformer.Interpolation.MassConcentrationSqrt

open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d : ℕ)

/-- **Proposition (prop: separation) is false as written.**

Take `N = 2` and `μ_0^1 = μ_0^2 = δ_q`, `q = (1, …, 1)/√d ∈ ℚ_1^{d-1}`.  For
any parameter curve `θ` under which the data has a solution `μ`, the
proposition's conclusion applied to `μ` for both indices would make
`conv_g supp μ(T)` disjoint from itself, hence empty; but `supp μ(T)` lies in
its hull and has full `μ(T)`-measure `1`.  No hypothesis on `θ`, `T` or the
number of switches is used, so no choice of the `O(d · N)` constant helps.

Source: arXiv:2411.04551v3, §3, `prop: separation`. -/
theorem not_separation (hd : 0 < d) :
    ∃ μ₀ : Idx 2 → ProbSphere d,
      (∀ i : Idx 2, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) ∧
      ∀ (θ : TimeParams d) (T : ℝ),
        (∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ 0 ∧ cauchyPB d θ μ) →
        ¬ ∀ i j : Idx 2, i ≠ j → ∀ μ ν : ℝ → ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ → ν 0 = μ₀ j → cauchyPB d θ ν →
          Disjoint (convG d (μ T : Measure (SSphere d)).support)
            (convG d (ν T : Measure (SSphere d)).support) := by
  refine ⟨fun _ => diracProb d (diagPoint d hd), fun _ x hx => ?_, ?_⟩
  · have hx' : x ∈ (Measure.dirac (diagPoint d hd)).support := hx
    rw [eq_of_mem_support_dirac hx']
    exact diagPoint_mem_positiveQuadrant d hd
  · rintro θ T ⟨μ, h0, hμ⟩ hdisj
    have h := disjoint_self.mp (hdisj 0 1 (by decide) μ μ h0 hμ h0 hμ)
    have hs : (μ T : Measure (SSphere d)).support = ∅ :=
      Set.subset_empty_iff.mp (by
        have := subset_convG d (μ T : Measure (SSphere d)).support
        rwa [h] at this)
    have h1 := Measure.measure_compl_support (μ := (μ T : Measure (SSphere d)))
    rw [hs, Set.compl_empty, measure_univ] at h1
    exact one_ne_zero h1

/-- The data of `not_separation` is admissible for `prop: separation`: the
two measures are supported in `ℚ_1^{d-1}`, here for `d = 1`. -/
example : ∀ i : Idx 2,
    (((fun _ : Idx 2 => diracProb 1 (diagPoint 1 one_pos)) i :
      Measure (SSphere 1))).support ⊆ positiveQuadrant 1 :=
  fun _ x hx => by
    have hx' : x ∈ (Measure.dirac (diagPoint 1 one_pos)).support := hx
    rw [eq_of_mem_support_dirac hx']
    exact diagPoint_mem_positiveQuadrant 1 one_pos

end Interpolation
end Transformer
