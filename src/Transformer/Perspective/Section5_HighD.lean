/-
# §6 — The high-dimensional case

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §6 of the survey:

* `Theorem thm: boumal`            — clustering for `d ≥ 3`, any `β ≥ 0`,
* `Theorem thm: d.infty`, `eq: expconvtocons` — the exponential rate when
  `d ≥ n` — are `Perspective.Section5_ExpRate`,
* `eq: therighthandside`,
* `e:dotalpha.step2`, `e:mineqalpha.step2`, `e:diffineqalpha.step2` — in
  `Perspective.Section5_HemisphereRate`, for the limit `x⋆` of step 1,
* `Theorem r:wendel` — Wendel's hemisphere probability.

The remark following `thm: boumal` — no smooth invariant measure — is
`Perspective.Section5_InvariantMeasure`.

`Lemma lem: hemisphere.clustering` — *cone collapse* — is `cone_collapse`, in
`Perspective.Section5_ExpRate`; its step 1, `eq: qual.conv`, is the
qualitative half of its conclusion and is not stated apart.  The last step of
its proof, integrating `e:diffineqalpha.step2` into the rate, is proved in
`Perspective.Section5_ConeCollapse`.  Steps 1 and 2 — that `min_i ⟨x_i(t), w⟩` does not decrease, and
the decomposition `e:decompox*.step2` of `x⋆` along the particles — are
`Perspective.Section5_Hemisphere`, which needs the one-sided calculus of
`Perspective.MinCurve`.

§6.2 (`thm: orthogonal`, `eq: ybeta`, `thm: phase.transition.curve`) and §6.3
(the phase-transition curve) are in `Perspective.Section5_HighDCurve`; the
auxiliary calculus lemma `lem: ez.lemma`, which the section uses but which
belongs to no dynamics, is proved in `Perspective.Section5_Vanishing`.

Both theorems of §6.1 are almost-everywhere statements, and the initial
sequences they have to exclude are exhibited in
`Perspective.Section5_Exceptional`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.Section5_Exceptional
import Transformer.Perspective.Section5_Vanishing
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **Theorem (thm: boumal).** *Clustering in dimension `d ≥ 3`.*

For `n ≥ 2`, `d ≥ 3` and `β ≥ 0`, the conclusion of `thm: beta.tiny` holds for
both `SA` and `USA`: Lebesgue-almost every initial sequence converges to a
single cluster `x⋆`, that is, lies in `𝒮_β = clusteringSet` and in its `USA`
counterpart `clusteringSetUSA`.

*Almost every* is not *every*, and the difference is not decoration: at every
`β` and every `d`, `antipodalPair_not_mem_clusteringSet` exhibits an initial
sequence outside `𝒮_β`, so the `∀ X₀` reading of this theorem is false rather
than unproved.  The exceptional set is null but non-empty, which is why the
conclusion is read against the uniform law `UniformTuple` of §4, exactly as in
`beta_interval`.

Not proved here.

Source: arXiv:2312.10794v5, §6.1, `thm: boumal`. -/
theorem boumal_clustering
    (hd : 3 ≤ d) (hn : 2 ≤ n) (β : ℝ) (hβ : 0 ≤ β) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β ∧ X₀ ∈ clusteringSetUSA d n β := by
  sorry

/-- The hypotheses of `boumal_clustering` are satisfiable: `d = 3`, `n = 2`,
`β = 0`. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- `r t = min_i ⟨x_i(t), w⟩`, the smallest coordinate of the configuration
along a fixed direction `w`, written as a specification: `r t` is a lower
bound for every `i`, and it is attained.

Spelling the minimum out this way rather than as `Finset.inf'` keeps the
nonemptiness proof of `[n]` out of the statements that use it — here
`Perspective.Section5_Hemisphere.hemisphere_step1_monotone`, and in
`Perspective.AppendixD_Alpha` the function `α` of `e:dotalpha`.

Source: arXiv:2312.10794v5, §6.1, step 1 of `lem: hemisphere.clustering`. -/
def IsMinInner (X : ℝ → SphereTuple d n) (w : SSphere d) (r : ℝ → ℝ) : Prop :=
  ∀ t : ℝ,
    (∀ i : Idx n,
      r t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))) ∧
    ∃ i : Idx n,
      r t = inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))

/-- **Equation (eq: therighthandside).**  The right-hand side of `SA` tested
against a fixed direction `w`:

  `d/dt ⟨x_i(t), w⟩
     = Z_{β,i}(t)⁻¹ Σ_j e^{β ⟨x_i,x_j⟩} ( ⟨x_j, w⟩ - ⟨x_i, x_j⟩ ⟨x_i, w⟩ )`.

The projection `Proj_{x_i}` is what produces the second summand, and with it
the sign that makes `r(t) = min_i ⟨x_i(t), w⟩` non-decreasing in
`Perspective.Section5_Hemisphere`: at a minimising `i` every bracket is `≥ 0`.

Source: arXiv:2312.10794v5, §6.1, `eq: therighthandside`. -/
theorem step1_rhs
    (β : ℝ) (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (w : EucSpace d) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => inner (𝕜 := ℝ) ((X s i : EucSpace d)) w)
      ((partitionSA d n β X t i)⁻¹ *
        ∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))) *
            (inner (𝕜 := ℝ) ((X t j : EucSpace d)) w
              - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
                  * inner (𝕜 := ℝ) ((X t i : EucSpace d)) w)) t := by
  have key : ∀ (x : EucSpace d) (c : ℝ) (S : Idx n → ℝ) (v : Idx n → EucSpace d),
      inner (𝕜 := ℝ) (proj d x (c • ∑ j : Idx n, S j • v j)) w
        = c * ∑ j : Idx n, S j *
            (inner (𝕜 := ℝ) (v j) w - inner (𝕜 := ℝ) x (v j) * inner (𝕜 := ℝ) x w) := by
    intro x c S v
    have h1 : inner (𝕜 := ℝ) (c • ∑ j : Idx n, S j • v j) w
        = c * ∑ j : Idx n, S j * inner (𝕜 := ℝ) (v j) w := by
      rw [real_inner_smul_left, sum_inner]
      simp [real_inner_smul_left]
    have h2 : inner (𝕜 := ℝ) x (c • ∑ j : Idx n, S j • v j)
        = c * ∑ j : Idx n, S j * inner (𝕜 := ℝ) x (v j) := by
      rw [real_inner_smul_right, inner_sum]
      simp [real_inner_smul_right]
    have h3 : ∑ j : Idx n, S j *
          (inner (𝕜 := ℝ) (v j) w - inner (𝕜 := ℝ) x (v j) * inner (𝕜 := ℝ) x w)
        = (∑ j : Idx n, S j * inner (𝕜 := ℝ) (v j) w)
            - (∑ j : Idx n, S j * inner (𝕜 := ℝ) x (v j)) * inner (𝕜 := ℝ) x w := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [proj, inner_sub_left, h1, real_inner_smul_left, h2, h3]
    ring
  refine (HasDerivAt.inner ℝ (hX t i) (hasDerivAt_const t w)).congr_deriv ?_
  rw [inner_zero_right, zero_add, key]

/-- **Theorem (r:wendel) — Wendel's theorem.**

Let `1 ≤ d ≤ n` and let `x_1,…,x_n` be i.i.d. uniformly distributed points on
`𝕊^{d-1}`. The probability that they all lie in the same open hemisphere
equals

  `2^{-(n-1)} · Σ_{k=0}^{d-1} C(n-1, k)`.

The event is the one `lem: hemisphere.clustering` needs, written on the
initial sequence itself; the law is the uniform `UniformTuple` of §4.

Not proved here: Wendel's counting argument is not formalized.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem wendel (hd : 1 ≤ d) (hdn : d ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      (P { X₀ : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
            0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d)) }).toReal
        = (∑ k ∈ Finset.range d, ((n - 1).choose k : ℝ)) / 2 ^ (n - 1) := by
  sorry

/-- The hypotheses of `wendel` are satisfiable: `d = n = 1`. -/
example : 1 ≤ 1 ∧ 1 ≤ 1 := ⟨le_rfl, le_rfl⟩

end Perspective
end Transformer
