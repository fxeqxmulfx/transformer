/-
# §6 — The high-dimensional case

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §6 of the survey:

* `Theorem thm: boumal`            — clustering for `d ≥ 3`, any `β ≥ 0`,
* `Theorem thm: d.infty`           — exponential rate when `d ≥ n`,
* `eq: expconvtocons`              — explicit convergence rate,
* `eq: therighthandside`, `eq: qual.conv`,
* `e:dotalpha.step2`,
* `e:mineqalpha.step2`, `e:diffineqalpha.step2`,
* `Theorem r:wendel` — Wendel's hemisphere probability,
* `Theorem r:wendel` — Wendel's hemisphere probability.

`Lemma lem: hemisphere.clustering` — *cone collapse* — is
`Perspective.Section5_ConeCollapse`, where it is proved from the two steps of
its own proof.  Steps 1 and 2 — that `min_i ⟨x_i(t), w⟩` does not decrease, and
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
single cluster `x⋆`, that is, lies in `𝒮_β = clusteringSet`.

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
      ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β := by
  sorry

/-- The hypotheses of `boumal_clustering` are satisfiable: `d = 3`, `n = 2`,
`β = 0`. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- *Invariant measures remark.* When `thm: beta.tiny` applies (e.g. always
for `d ≥ 3`), neither `SA` nor `USA` admits a smooth invariant measure.

"Smooth" is read as *having a density*: `μ ≪ P`, with `P` the uniform law
`UniformTuple` of §4.  That already rules out the Dirac masses on consensus
configurations, which *are* invariant but sit on a `P`-null set, and it is what
makes the remark a consequence of `boumal_clustering`: everything is swept into
that null set, so no measure with a density can be preserved.  The flow is
presented as any map `Φ` that transports initial data along solutions of `SA`.

Not proved here: it rests on `boumal_clustering`, which is a `sorry`.

Source: arXiv:2312.10794v5, §6.1 (remark after `thm: boumal`). -/
theorem no_smooth_invariant_measure (β : ℝ) (hd : 3 ≤ d) (hn : 2 ≤ n)
    (hβ : 0 ≤ β) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
    ∀ μ : Measure (SphereTuple d n), IsProbabilityMeasure μ → μ ≪ P →
      ¬ ∃ Φ : ℝ → SphereTuple d n → SphereTuple d n,
          (∀ (X : ℝ → SphereTuple d n), Perspective.SA d n β X →
              ∀ t : ℝ, Φ t (X 0) = X t) ∧
          ∀ t : ℝ, μ.map (Φ t) = μ := by
  sorry

/-- The hypotheses of `no_smooth_invariant_measure` are satisfiable: `d = 3`,
`n = 2`, `β = 0`. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-! ### §6.1 — Exponential rate when `d ≥ n` -/

/-- **Theorem (thm: d.infty), eq: expconvtocons.** *Exponential clustering when
`d ≥ n`.*

For `n ≥ 1`, `β > 0` and `d ≥ n`, if the initial points
`(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` are uniformly distributed, then almost
surely there exist `x⋆ ∈ 𝕊^{d-1}` and constants `C, lam > 0` such that

  `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`  for all `i` and `t ≥ 0`.

*Almost surely* is part of the statement and not a turn of phrase: for `n = 2`
the antipodal pair admits no such rate at any `β`
(`antipodalPair_not_exponential`), so the `∀ X₀` reading is false.  As in
`boumal_clustering`, the initial sequence is drawn from the uniform law
`UniformTuple` of §4.

Not proved here.

Source: arXiv:2312.10794v5, §6.1, `thm: d.infty`, `eq: expconvtocons`. -/
theorem d_infty_exponential
    (hn : 1 ≤ n) (β : ℝ) (hβ : 0 < β) (hdn : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X₀ ∂P,
        ∃ (x_star : SSphere d) (C lam : ℝ),
          0 < C ∧ 0 < lam ∧
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
            ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
              ‖((X t i : EucSpace d)) - x_star‖ ≤ C * Real.exp (-(lam * t)) := by
  sorry

/-- The hypotheses of `d_infty_exponential` are satisfiable: `d = n = 1`,
`β = 1`. -/
example : 1 ≤ 1 ∧ (0 : ℝ) < 1 ∧ 1 ≤ 1 := ⟨le_rfl, one_pos, le_rfl⟩

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

/-- **Equation (eq: qual.conv).** *Qualitative convergence at step 1.*

  `lim_{t→∞} x_i(t) = x⋆`  for all `i`. -/
theorem hemisphere_step1_qual_conv
    (β : ℝ) (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ∃ x_star : SSphere d,
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
        ∀ i : Idx n,
          Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
            Filter.atTop (nhds 0) := by
  sorry

/-- **Equation (e:diffineqalpha.step2).** Differential inequality for
`α(t) = min_i ⟨x_i(t), x⋆⟩`:

  `α̇(t) ≥ 1/(2 n e^{2β}) · (1 - α(t))`,  for `t ≥ t₀`,

which integrates to the exponential rate of `lem: hemisphere.clustering`.

The minimum is presented by its two defining properties (`α t` is a lower
bound, and is attained) rather than as a `Finset.inf'`, so that no nonemptiness
witness for `Idx n` has to be carried through the statement.

Not proved here: the inequality is step 2 of the proof of
`lem: hemisphere.clustering`, and it is what
`Perspective.Section5_ConeCollapse.hemisphere_clustering` carries as a
hypothesis.

Source: arXiv:2312.10794v5, §6.1, `e:diffineqalpha.step2`. -/
theorem step2_alpha_diff_ineq
    (β : ℝ) (X : ℝ → SphereTuple d n) (x_star : SSphere d)
    (α : ℝ → ℝ) (t₀ : ℝ) (hβ : 0 ≤ β) (hX : Perspective.SA d n β X)
    (hlb : ∀ t : ℝ, ∀ i : Idx n,
      α t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)))
    (hmin : ∀ t : ℝ, ∃ i : Idx n,
      α t = inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) :
    ∀ t : ℝ, t₀ ≤ t →
      ∃ α' : ℝ, HasDerivAt α α' t ∧
        (1 - α t) / (2 * (n : ℝ) * Real.exp (2 * β)) ≤ α' := by
  sorry

/-- The hypotheses of `step2_alpha_diff_ineq` are satisfiable, and by a genuine
solution: one token sitting at `x⋆` is a consensus equilibrium of `eq: SA`, and
`α ≡ 1` is then the minimum `min_i ⟨x_i(t), x⋆⟩`, attained at the only token
there is. -/
example :
    (0 : ℝ) ≤ 1 ∧
      Perspective.SA 1 1 1 (fun _ _ => basePoint 0) ∧
      (∀ t : ℝ, ∀ i : Idx 1,
        (1 : ℝ) ≤ inner (𝕜 := ℝ)
          (((fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) t i : EucSpace 1))
          ((basePoint 0 : EucSpace 1))) ∧
      (∀ t : ℝ, ∃ i : Idx 1,
        (1 : ℝ) = inner (𝕜 := ℝ)
          (((fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) t i : EucSpace 1))
          ((basePoint 0 : EucSpace 1))) := by
  have hb : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1)) ((basePoint 0 : EucSpace 1))
      = 1 := by
    rw [real_inner_self_eq_norm_mul_norm,
      mem_sphere_zero_iff_norm.mp (basePoint 0 : SSphere 1).2]
    ring
  exact ⟨zero_le_one, Perspective.SA_const_consensus 1 1 one_pos 1 (basePoint 0),
    fun _ _ => hb.ge, fun _ => ⟨0, hb.symm⟩⟩

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
