/-
# Measure-to-measure interpolation — Clustering of input data

Formalization of §2 of arXiv:2411.04551v3:

* `Proposition prop: targets.atoms`  — clustering to a single point mass,
* `Proposition prop: compression`     — clustering to discrete measures,
* `Remark rem: nb.disc.clustering`    — bound on the number of switches.

Both propositions assert that *some* parameter curve drives `eq: cauchy.pb`
to a prescribed target, and measure the error in a Wasserstein distance, which
Mathlib does not have; the geodesic convex hull `conv_g` is not available here
either.  They therefore carry the distance as a parameter, with the geodesic
hull replaced by the support of the measure — weaker, and stated as such — and
neither is proved.  `rem: nb.disc.clustering` counts the switches of
the parameter curve; the count is here reduced to "finitely many", since the
packing and homotopy quantities `𝖭_k^i(δ)` and `L_{k,n}^i` it is expressed in
are not defined in this development.

What is proved is the invariance that both proofs start from: the barycenter
of a measure supported in a cap `⟨·, w⟩ ≥ c` lies in that same cap, so a
measure supported in an open hemisphere has a nonzero barycenter — which is
what makes the drift of `eq: average.vf` point into the hemisphere.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Measure.Support

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d N M : ℕ)

/-- The barycenter `∫ x dμ(x) ∈ ℝ^d` of a measure on the sphere — the
attention output of `eq: average.vf`, whose `𝐁 ≡ 0` turns the softmax average
into the plain one. -/
noncomputable def barycenter (μ : ProbSphere d) : EucSpace d :=
  ∫ x, (x : EucSpace d) ∂(μ : Measure (SSphere d))

/-- **A cap is barycenter-invariant.**

If `μ` gives full mass to the cap `{ x : ⟨x, w⟩ ≥ c }`, then its barycenter
satisfies `⟨∫ x dμ, w⟩ ≥ c` as well: the inner product with `w` commutes with
the integral, and the integral of a function bounded below by `c` against a
probability measure is bounded below by `c`.

This is the step the proof of `prop: targets.atoms` opens with, for the cap
`c > 0` cut out by an open hemisphere.  Source: arXiv:2411.04551v3, §2. -/
theorem le_inner_barycenter (μ : ProbSphere d) (w : EucSpace d) (c : ℝ)
    (hint : Integrable (fun x : SSphere d => (x : EucSpace d))
      (μ : Measure (SSphere d)))
    (hcap : ∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)),
      c ≤ inner (𝕜 := ℝ) (x : EucSpace d) w) :
    c ≤ inner (𝕜 := ℝ) (barycenter d μ) w := by
  have hmono : ∫ _ : SSphere d, c ∂(μ : Measure (SSphere d))
      ≤ ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
          ∂(μ : Measure (SSphere d)) :=
    integral_mono_ae (integrable_const c) (hint.inner_const w) hcap
  rw [integral_const, probReal_univ, one_smul] at hmono
  have hswap : ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
      ∂(μ : Measure (SSphere d)) = inner (𝕜 := ℝ) (barycenter d μ) w := by
    calc ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
          ∂(μ : Measure (SSphere d))
        = ∫ x : SSphere d, inner (𝕜 := ℝ) w (x : EucSpace d)
            ∂(μ : Measure (SSphere d)) :=
          integral_congr_ae (Filter.Eventually.of_forall fun x => real_inner_comm _ _)
      _ = inner (𝕜 := ℝ) w (∫ x : SSphere d, (x : EucSpace d)
            ∂(μ : Measure (SSphere d))) := integral_inner hint w
      _ = inner (𝕜 := ℝ) (barycenter d μ) w := real_inner_comm _ _
  rwa [hswap] at hmono

/-- **A measure supported in an open hemisphere has a nonzero barycenter.**

The hemisphere hypothesis of `prop: targets.atoms` is `⟨x, w⟩ ≥ c` for some
`c > 0`; by `le_inner_barycenter` the barycenter then has positive inner
product with `w`, so it cannot vanish, and `eq: average.vf` has a nonzero
drift.  Source: arXiv:2411.04551v3, §2. -/
theorem barycenter_ne_zero (μ : ProbSphere d) (w : EucSpace d) (c : ℝ) (hc : 0 < c)
    (hint : Integrable (fun x : SSphere d => (x : EucSpace d))
      (μ : Measure (SSphere d)))
    (hcap : ∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)),
      c ≤ inner (𝕜 := ℝ) (x : EucSpace d) w) :
    barycenter d μ ≠ 0 := by
  intro h
  have hle := le_inner_barycenter d μ w c hint hcap
  rw [h, inner_zero_left] at hle
  linarith

/-- The hypotheses of both are satisfiable, and not by an empty cap: the Dirac
mass at a point `x₀` of the sphere is a probability measure, the coordinate
map is integrable against it, and all of its mass sits in the cap
`⟨·, x₀⟩ ≥ 1`. -/
example (x₀ : SSphere 1) :
    (0 : ℝ) < 1 ∧
      Integrable (fun x : SSphere 1 => (x : EucSpace 1))
        ((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1)) ∧
      ∀ᵐ x : SSphere 1
        ∂((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1)),
          (1 : ℝ) ≤ inner (𝕜 := ℝ) (x : EucSpace 1) ((x₀ : EucSpace 1)) := by
  refine ⟨one_pos, integrable_dirac (by simp), ?_⟩
  rw [show ((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1))
      = Measure.dirac x₀ from rfl, MeasureTheory.ae_dirac_eq, Filter.eventually_pure]
  have hx : ‖(x₀ : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x₀.2
  rw [real_inner_self_eq_norm_mul_norm, hx, mul_one]

/-- **Proposition (prop: targets.atoms).**  *Clustering to a single point.*

If `𝐁 ∈ M_{d×d}(ℝ)` and `supp μ_0` is contained in an open hemisphere, then
the solution to `eq: cauchy.pb`–`eq: vf` with `(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)`
satisfies `diam(conv_g supp μ(t)) → 0` as `t → ∞`, and for any `ε > 0` there
are `z ∈ conv_g supp μ_0` and `T > 0` with `W_∞(μ(T), δ_z) ≤ ε`.

Three weakenings: `conv_g` is replaced by the support itself (the diameters
agree inside an open hemisphere, which is not proved here), the membership
`z ∈ conv_g supp μ_0` is dropped, and `W_∞` (`Winf`) is a parameter.  The paper's
companion rate `inf{ t : W_2(μ(t), δ_z) ≤ ε } = O(log(1/ε))` is not
formalized.

Not proved here.

Source: arXiv:2411.04551v3, §2. -/
theorem clustering_to_atom
    (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (B : ParamMatrix d) (μ₀ : ProbSphere d)
    (hhemi : ∃ w : SSphere d, ∀ x ∈ (μ₀ : Measure (SSphere d)).support,
      0 < inner (𝕜 := ℝ) (x : EucSpace d) ((w : EucSpace d))) :
    ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ →
      cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ →
        Filter.Tendsto
            (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support))
            Filter.atTop (nhds 0) ∧
          ∀ ε : ℝ, 0 < ε → ∃ (z : SSphere d) (T : ℝ), 0 < T ∧
            Winf (μ T : Measure (SSphere d)) (Measure.dirac z) ≤ ε := by
  sorry

/-- The hypothesis of `clustering_to_atom` is satisfiable: a Dirac mass is
supported at its own point, which lies in the open hemisphere around itself. -/
example (x₀ : SSphere 1) :
    ∃ w : SSphere 1,
      ∀ x ∈ ((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) :
          Measure (SSphere 1)).support,
        0 < inner (𝕜 := ℝ) (x : EucSpace 1) ((w : EucSpace 1)) := by
  refine ⟨x₀, fun x hx => ?_⟩
  have hxx : x = x₀ := eq_of_mem_support_dirac hx
  subst hxx
  have hx1 : ‖(x : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [real_inner_self_eq_norm_mul_norm, hx1, mul_one]
  exact one_pos

/-- `θ` is piecewise constant on `[0, T]` with at most `K` pieces — hence at
most `K - 1` switches: there are times `0 = t_0 ≤ … ≤ t_K = T` such that `θ`
is constant on each `[t_k, t_{k+1})`. -/
def PiecewiseConstant (θ : TimeParams d) (T : ℝ) (K : ℕ) : Prop :=
  ∃ t : Fin (K + 1) → ℝ, Monotone t ∧ t 0 = 0 ∧ t (Fin.last K) = T ∧
    ∀ k : Fin K, ∀ s s' : ℝ,
      s ∈ Set.Ico (t k.castSucc) (t k.succ) →
        s' ∈ Set.Ico (t k.castSucc) (t k.succ) → θ s = θ s'

/-- A constant parameter curve is piecewise constant with one piece, so the
definition is not vacuous. -/
example (θ₀ : Params d) : PiecewiseConstant d (fun _ => θ₀) 1 1 := by
  refine ⟨fun i => (i.val : ℝ), fun a b hab => ?_, by simp, by simp,
    fun k s s' _ _ => rfl⟩
  exact Nat.cast_le.mpr hab

/-- **Proposition (prop: compression) with Remark (rem: nb.disc.clustering).**
*Clustering to discrete measures.*

Let the `μ_0^i` have no atoms and pairwise disjoint supports.  For any
`M ≥ 1`, targets `x_k^i` and weights `α_k^i ≥ 0` with `Σ_k α_k^i = 1`, there
are piecewise-constant `(𝐖, 𝐔, b) : [0, T] → M_{d×d}(ℝ)² × ℝ^d` — finitely
many switches, which is the content of `rem: nb.disc.clustering` — such that
the solutions `μ^i` of `eq: cauchy.pb`–`eq: vf` with `𝐕 ≡ 0` satisfy

  `W_2(μ^i(T), Σ_k α_k^i δ_{x_k^i}) ≤ ε`

and have pairwise disjoint supports.

`W_2` is a parameter, the geodesic convex hulls of the hypothesis and the
requirement `x_k^i ∈ conv_g supp μ_0^i` are replaced by supports, and the
remark's explicit switch count `N · M · max_{(i,k)} 𝖭_k^i(δ) · max_n L_{k,n}^i`
is weakened to the existence of a finite count, its ingredients not being
defined here.

Not proved here.

Source: arXiv:2411.04551v3, §2. -/
theorem compression
    (W₂ : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (μ₀ : Idx N → ProbSphere d) (x : Idx N → Idx M → SSphere d)
    (α : Idx N → Idx M → ℝ) (ε : ℝ) (hM : 1 ≤ M) (hε : 0 < ε)
    (hatom : ∀ i : Idx N, ∀ y : SSphere d, (μ₀ i : Measure (SSphere d)) {y} = 0)
    (hdisj : ∀ i j : Idx N, i ≠ j → Disjoint ((μ₀ i : Measure (SSphere d)).support)
      ((μ₀ j : Measure (SSphere d)).support))
    (hαnonneg : ∀ i : Idx N, ∀ k : Idx M, 0 ≤ α i k)
    (hαsum : ∀ i : Idx N, ∑ k : Idx M, α i k = 1) :
    ∃ (θ : TimeParams d) (T : ℝ) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
      0 < T ∧ (∀ s : ℝ, (θ s).V = 0) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      (∀ i : Idx N, W₂ (μ i T : Measure (SSphere d))
          (∑ k : Idx M, (α i k).toNNReal • Measure.dirac (x i k)) ≤ ε) ∧
      (∀ i j : Idx N, i ≠ j →
        Disjoint ((μ i T : Measure (SSphere d)).support)
          ((μ j T : Measure (SSphere d)).support)) := by
  sorry

/-- The hypotheses of `compression` are satisfiable — but only on an empty
family of initial measures, `N = 0`, where the atomlessness and disjointness
conditions have nothing to check.  A witness with `N ≥ 1` would need an
atomless probability measure on the sphere, and this development constructs
none: the uniform measure is exactly what it carries as a parameter. -/
example (μ₀ : Idx 0 → ProbSphere 1) (α : Idx 0 → Idx 1 → ℝ) :
    1 ≤ 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 0, ∀ y : SSphere 1, (μ₀ i : Measure (SSphere 1)) {y} = 0) ∧
      (∀ i j : Idx 0, i ≠ j →
        Disjoint ((μ₀ i : Measure (SSphere 1)).support)
          ((μ₀ j : Measure (SSphere 1)).support)) ∧
      (∀ i : Idx 0, ∀ k : Idx 1, 0 ≤ α i k) ∧
      (∀ i : Idx 0, ∑ k : Idx 1, α i k = 1) :=
  ⟨le_rfl, one_pos, fun i => i.elim0, fun i => i.elim0, fun i => i.elim0, fun i => i.elim0⟩

end Interpolation
end Transformer
