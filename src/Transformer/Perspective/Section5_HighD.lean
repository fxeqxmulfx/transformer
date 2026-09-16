/-
# §6 — The high-dimensional case

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §6 of the survey:

* `Theorem thm: boumal`            — clustering for `d ≥ 3`, any `β ≥ 0`,
* `Theorem thm: d.infty`           — exponential rate when `d ≥ n`,
* `eq: expconvtocons`              — explicit convergence rate,
* `Lemma lem: hemisphere.clustering`  — *cone collapse*,
* `eq: therighthandside`, `eq: qual.conv`,
* `e:decompox*.step2`, `e:dotalpha.step2`,
* `e:mineqalpha.step2`, `e:diffineqalpha.step2`,
* `Theorem r:wendel` — Wendel's hemisphere probability,
* `Theorem r:wendel` — Wendel's hemisphere probability.

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

A `Prop`-valued definition and not a theorem: it rests on
`boumal_clustering`, which is a `sorry` here.

Source: arXiv:2312.10794v5, §6.1 (remark after `thm: boumal`). -/
def NoSmoothInvariantMeasure (β : ℝ) : Prop :=
  3 ≤ d → 2 ≤ n → 0 ≤ β →
  ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
  ∀ μ : Measure (SphereTuple d n), IsProbabilityMeasure μ → μ ≪ P →
    ¬ ∃ Φ : ℝ → SphereTuple d n → SphereTuple d n,
        (∀ (X : ℝ → SphereTuple d n), Perspective.SA d n β X →
            ∀ t : ℝ, Φ t (X 0) = X t) ∧
        ∀ t : ℝ, μ.map (Φ t) = μ

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

/-- **Lemma (lem: hemisphere.clustering) — *Cone collapse.*

Let `β > 0` and `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` be such that there exists
`w ∈ 𝕊^{d-1}` with `⟨x_i(0), w⟩ > 0` for all `i`.  Then the unique solution
of `SA` (or `USA`) converges exponentially to a common point `x⋆`:

  `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`.

The same conclusion holds for `eq: transformerSd.QKV` with `V = I_d` and
arbitrary `d × d` matrices `Q, K`. -/
lemma hemisphere_clustering
    (β : ℝ) (hβ : 0 < β)
    (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ∃ (x_star : SSphere d) (C lam : ℝ),
      0 < C ∧ 0 < lam ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
        ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
          ‖((X t i : EucSpace d)) - x_star‖ ≤ C * Real.exp (-(lam * t)) := by
  sorry

/-- `r t = min_i ⟨x_i(t), w⟩`, the smallest coordinate of the configuration
along a fixed direction `w`, written as a specification: `r t` is a lower
bound for every `i`, and it is attained.

Spelling the minimum out this way rather than as `Finset.inf'` keeps the
nonemptiness proof of `[n]` out of the statements that use it — here
`hemisphere_step1_monotone`, and in `Perspective.AppendixD_Alpha` the
function `α` of `e:dotalpha`.

Source: arXiv:2312.10794v5, §6.1, step 1 of `lem: hemisphere.clustering`. -/
def IsMinInner (X : ℝ → SphereTuple d n) (w : SSphere d) (r : ℝ → ℝ) : Prop :=
  ∀ t : ℝ,
    (∀ i : Idx n,
      r t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))) ∧
    ∃ i : Idx n,
      r t = inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))

/-- *Step 1 inequalities in the proof of `lem: hemisphere.clustering`:*

`r(t) := min_i ⟨x_i(t), w⟩` is non-decreasing on `ℝ_{≥0}`.

Not proved here: it follows from `step1_rhs`, whose brackets are all `≥ 0` at
a minimising index.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, step 1. -/
theorem hemisphere_step1_monotone
    (β : ℝ) (w : SSphere d) (X : ℝ → SphereTuple d n) (r : ℝ → ℝ)
    (hX : Perspective.SA d n β X) (hr : IsMinInner d n X w r)
    (hinit : ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d))) :
    MonotoneOn r (Set.Ici (0 : ℝ)) := by
  sorry

/-- The hypotheses of `hemisphere_step1_monotone` are satisfiable: the
consensus solution, with `w` the common position and `r ≡ ⟨x, x⟩ = 1`. -/
example :
    Perspective.SA 1 1 0 (fun _ _ => basePoint 0) ∧
      IsMinInner 1 1 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) ∧
      ∀ i : Idx 1,
        0 < inner (𝕜 := ℝ)
              (((fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) 0 i : EucSpace 1))
              (((basePoint 0 : SSphere 1)) : EucSpace 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact ⟨Perspective.SA_const_consensus 1 1 one_pos 0 (basePoint 0),
    fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩, fun _ => by norm_num [hxx]⟩

/-- **Equation (eq: therighthandside).**  The right-hand side of `SA` tested
against a fixed direction `w`:

  `d/dt ⟨x_i(t), w⟩
     = Z_{β,i}(t)⁻¹ Σ_j e^{β ⟨x_i,x_j⟩} ( ⟨x_j, w⟩ - ⟨x_i, x_j⟩ ⟨x_i, w⟩ )`.

The projection `Proj_{x_i}` is what produces the second summand, and with it
the sign that makes `r(t) = min_i ⟨x_i(t), w⟩` non-decreasing in
`hemisphere_step1_monotone`: at a minimising `i` every bracket is `≥ 0`.

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

/-- **Equation (e:decompox*.step2).**

  `x⋆ = Σ_k θ_k(t) x_k(t)`,  with `Σ_k θ_k(t) ≥ 1`, `θ_k(t) ≥ 0`. -/
theorem step2_decomposition
    (β : ℝ) (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (x_star : SSphere d) :
    ∀ t : ℝ, 0 < t → ∃ θ : Idx n → ℝ,
      (∀ k, 0 ≤ θ k) ∧ (1 ≤ ∑ k : Idx n, θ k) ∧
        ((x_star : EucSpace d) = ∑ k : Idx n, (θ k) • ((X t k : EucSpace d))) := by
  sorry

/-- **Equation (e:diffineqalpha.step2).** Differential inequality for
`α(t) = min_i ⟨x_i(t), x⋆⟩`:

  `α̇(t) ≥ 1/(2 n e^{2β}) · (1 - α(t))`,  for `t ≥ t₀`,

which integrates to the exponential rate of `lem: hemisphere.clustering`.

The minimum is presented by its two defining properties (`α t` is a lower
bound, and is attained) rather than as a `Finset.inf'`, so that no nonemptiness
witness for `Idx n` has to be carried through the statement.

A `Prop`-valued definition and not a theorem: the inequality is step 2 of the
proof of `lem: hemisphere.clustering`, which is a `sorry` here.

Source: arXiv:2312.10794v5, §6.1, `e:diffineqalpha.step2`. -/
def Step2AlphaDiffIneq
    (β : ℝ) (X : ℝ → SphereTuple d n) (x_star : SSphere d)
    (α : ℝ → ℝ) (t₀ : ℝ) : Prop :=
  0 ≤ β → Perspective.SA d n β X →
  (∀ t : ℝ, ∀ i : Idx n,
      α t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) →
  (∀ t : ℝ, ∃ i : Idx n,
      α t = inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) →
  ∀ t : ℝ, t₀ ≤ t →
    ∃ α' : ℝ, HasDerivAt α α' t ∧
      (1 - α t) / (2 * (n : ℝ) * Real.exp (2 * β)) ≤ α'

/-- **Theorem (r:wendel) — Wendel's theorem.**

Let `1 ≤ d ≤ n` and let `x_1,…,x_n` be i.i.d. uniformly distributed points on
`𝕊^{d-1}`. The probability that they all lie in the same open hemisphere
equals

  `2^{-(n-1)} · Σ_{k=0}^{d-1} C(n-1, k)`.

The event is the one `lem: hemisphere.clustering` needs, written on the
initial sequence itself; the law is the uniform `UniformTuple` of §4.

A `Prop`-valued definition and not a theorem: Wendel's counting argument is
not formalized here.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
def Wendel : Prop :=
  1 ≤ d → d ≤ n →
  ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
    (P { X₀ : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d)) }).toReal
      = (∑ k ∈ Finset.range d, ((n - 1).choose k : ℝ)) / 2 ^ (n - 1)

end Perspective
end Transformer
