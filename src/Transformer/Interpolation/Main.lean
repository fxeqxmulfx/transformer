/-
# Measure-to-measure interpolation — Main theorems

Formalization of the main theorems of arXiv:2411.04551v3:

* `Theorem thm: targets.atoms`  — interpolation when targets are point masses,
* `Theorem thm: main.result`    — general interpolation,
* `Lemma lem: hyp.propagation`  — propagation of transport maps: false as
                                  written, refuted in
                                  `Interpolation.HypPropagationFalse`,
* `Lemma lem: monge`            — Monge-style optimal-transport identity, proved in
                                  `Transformer.Interpolation.Monge`,
* `Lemma lem: univ.approx`      — universal `L²`-map approximation.

All but `lem: monge` assert the existence of a parameter curve, or of a map, with no
construction available here, so they are not proved: each is a theorem closed by `sorry`.

The two theorems carry the manuscript's standing assumption that the input measures are pairwise
distinct and so are the targets; without it two equal inputs, which the well-posed Cauchy problem
sends to the same solution, could not be matched to different targets.  The `O(d · N)` switch bound
belongs to `thm: targets.atoms` only: for general targets the source says the number of switches
can be exponential in `d`, and `thm: main.result` states piecewise-constant parameters and nothing
more.  `lem: univ.approx` is about the perceptron part `eq: neural.pde.sphere` alone.

The transport maps of `thm: main.result` are required to be measurable rather
than to lie in `L²(𝕊^{d-1}; 𝕊^{d-1})`; on a sphere of finite measure and with
values in a bounded set the two agree, which is not proved here.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.BallTransport
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.Disentanglement
import Transformer.Interpolation.NeuralODE
import Transformer.Interpolation.Wasserstein

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d N : ℕ)

/-- `w` misses every input support — the "hole" of `eq: assumption.hole`. -/
def IsHole (μ : Idx N → ProbSphere d) (w : SSphere d) : Prop :=
  ∀ i : Idx N, w ∉ (μ i : Measure (SSphere d)).support

/-- A constant family of Dirac masses at `x` has a hole, namely the antipode of
`x`: every support is `{x}`, and `-x ≠ x`.  This is the witness the
satisfiability examples below use for `eq: assumption.hole`. -/
theorem isHole_antipode_diracProb (x : SSphere d) :
    IsHole d N (fun _ => diracProb d x) (antipode d x) := by
  intro i hmem
  have hmem' : antipode d x ∈ (Measure.dirac x).support := hmem
  exact antipode_ne d x (eq_of_mem_support_dirac hmem')

/-- **Theorem (thm: targets.atoms).**  *Interpolation to point-mass targets.*

For `d ≥ 3` and data `(μ_0^i, δ_{x^i})_{i=1}^N` with a hole
`w_0 ∈ 𝕊^{d-1} \ ⋃_i supp μ_0^i`, and for any `T, ε > 0`, there is
`θ ∈ L^∞((0, T); Θ)` such that the solution `μ^i` of `eq: cauchy.pb` with data
`μ_0^i` and parameters `θ` satisfies `W_2(μ^i(T), δ_{x^i}) ≤ ε`.  Moreover `θ`
can be chosen piecewise constant with `O(d · N)` switches and

  `‖θ‖_{L^∞((0,T); Θ)} = O((d · N) / T + log(1/ε))`.

The `L^∞` norm of `θ` is measured coordinate by coordinate, as the sum of the
operator norms of the four matrices and of `‖b‖`; `Params d` carries no norm of
its own.

**What the source says and what is changed here.**  Four repairs, none of
them a weakening of what the paper proves.

*`W_2` is `Interpolation.W2`.*  It was a free function of two measures, of
which nothing was assumed — see `not_forall_monge` for what that costs.

*The two `O(·)` constants are quantified before the data.*  That is what
`O(d · N)` and `O((d·N)/T + log(1/ε))` mean: one pair of constants serving
every dimension, every `N` and every datum.  Written as a universally
quantified binder, as `C` was, the switch bound said instead that *every*
constant works, so `C = 0` forced `K = 0`; written as an existential inside,
as `Cnorm` was, it allowed a constant chosen after the data, which is weaker
than the paper.  Both now stand outside every other quantifier, so `d` and `N`
are bound here rather than taken from the section.

*The standing assumption of §1 is written in.*  Before the theorem the source
fixes `μ_0^i ≢ μ_0^j` and `μ_1^i ≢ μ_1^j` for `i ≠ j`, "for simplicity"; the
targets `δ_{x^i}` are pairwise distinct exactly when the `x^i` are.  The
statement had neither.  Without distinct inputs it fails: the Cauchy problem has
a unique solution, so equal inputs `μ_0^1 = μ_0^2` end at the same measure and
cannot both come within `ε` of different targets.  The footnote which says the
assumption can be removed (`appendix: technical`) concerns the targets and the
hole, not the inputs.

*The norm bound is asked for `ε < 1`.*  The `O(·)` in `log(1/ε)` is an
asymptotic as `ε ↓ 0`.  As an inequality for every `ε > 0` it cannot hold: at
`ε > e^{d N / T}` the right-hand side `Cnorm · ((d N)/T + log(1/ε))` is negative
for `Cnorm > 0`, and a norm is not.  The other conclusions hold for every
`ε > 0`.

*`N ≥ 1`.*  The index set of the source is `⟦1, N⟧`.  `PiecewiseConstant` counts
pieces, and `[0, T]` has at least one, so at `N = 0` the bound `K ≤ C · d · N = 0`
was false whatever `C`.

Not proved here.

Source: arXiv:2411.04551v3, §1, `thm: targets.atoms`. -/
theorem targets_atoms :
    ∃ C Cnorm : ℝ, ∀ (d N : ℕ) (μ₀ : Idx N → ProbSphere d) (xtarget : Idx N → SSphere d)
      (T ε : ℝ), 3 ≤ d → 1 ≤ N → 0 < T → 0 < ε → Function.Injective μ₀ →
      Function.Injective xtarget → (∃ w₀ : SSphere d, IsHole d N μ₀ w₀) →
      ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
        (K : ℝ) ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
        (ε < 1 → ∀ s ∈ Set.Icc (0 : ℝ) T,
          ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
            ≤ Cnorm * ((d * N : ℝ) / T + Real.log (1 / ε))) ∧
        (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
        ∀ i : Idx N,
          W2 d (μ i T : Measure (SSphere d)) (Measure.dirac (xtarget i)) ≤ ε := by
  sorry

/-- The conditions inside `targets_atoms` are satisfiable, so its conclusion is
asked of a nonempty class of data: `d = 3`, `T = ε = 1`, and a one-element
family of Dirac masses on `𝕊^2`, whose hole is the antipode. -/
example :
    3 ≤ 3 ∧ 1 ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      Function.Injective (fun _ : Idx 1 => diracProb 3 (basePoint 2)) ∧
      Function.Injective (fun _ : Idx 1 => basePoint 2 : Idx 1 → SSphere 3) ∧
      ∃ w₀ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₀ :=
  ⟨le_rfl, le_rfl, one_pos, one_pos, Function.injective_of_subsingleton _,
    Function.injective_of_subsingleton _, _, isHole_antipode_diracProb 3 1 (basePoint 2)⟩

/-- **Theorem (thm: main.result).**  *General interpolation.*

For `d ≥ 3` and data `(μ_0^i, μ_1^i)_{i=1}^N` such that

* (eq: assumption.hole) some `w_0` misses every input support and some `w_1`
  misses every target support,
* each `μ_1^i` is the pushforward of `μ_0^i` under a measurable
  `𝖳^i : 𝕊^{d-1} → 𝕊^{d-1}`,

for any `T, ε > 0` there is a piecewise-constant `θ` such that
`W_2(μ^i(T), μ_1^i) ≤ ε` for every `i`.

**What the source says and what is changed here.**  Three repairs, none of
them a weakening of what the paper proves.

*`W_2` is `Interpolation.W2`* rather than a free function; see `targets_atoms`.

*There is no bound on the number of switches.*  The theorem says only that `θ`
"can be chosen piecewise constant".  The `O(d · N)` bound is that of
`thm: targets.atoms`, where the targets are atoms; the source states for the
general case that the number of switches "can be exponential in `d`" (§1.3,
`rem: nb.disc.clustering`), because it depends on the packing numbers of the
supports and, through the simple function approximating each `𝖳^i`, on `ε`.  The
former statement asked `K ≤ C · d · N` with `C` before the data — a bound the
paper proves for another theorem and disclaims for this one — and it is removed
with `C`.

*The standing assumption of §1 is written in:* `μ_0^i ≢ μ_0^j` and
`μ_1^i ≢ μ_1^j` for `i ≠ j`, as in `targets_atoms`, where the reason is given.
The source's footnote says the assumption on the targets, like the two holes,
can be removed at the price of technicalities (`appendix: technical`); the one
on the inputs cannot.

Not proved here.

Source: arXiv:2411.04551v3, §1, `thm: main.result`. -/
theorem main_result (μ₀ μ₁ : Idx N → ProbSphere d) (T ε : ℝ)
    (hd : 3 ≤ d) (hT : 0 < T) (hε : 0 < ε)
    (hinj₀ : Function.Injective μ₀) (hinj₁ : Function.Injective μ₁)
    (hhole₀ : ∃ w₀ : SSphere d, IsHole d N μ₀ w₀) (hhole₁ : ∃ w₁ : SSphere d, IsHole d N μ₁ w₁)
    (htrans : ∀ i : Idx N, ∃ Tr : SSphere d → SSphere d, Measurable Tr ∧
      Measure.map Tr (μ₀ i : Measure (SSphere d)) = (μ₁ i : Measure (SSphere d))) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
      PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ∀ i : Idx N, W2 d (μ i T : Measure (SSphere d)) (μ₁ i : Measure (SSphere d)) ≤ ε := by
  sorry

/-- The conditions inside `main_result` are satisfiable: `d = 3`, `T = ε = 1`,
and the same one-element family of Dirac masses on both sides, which is its own
pushforward under the identity and has the antipode as a hole. -/
example :
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      Function.Injective (fun _ : Idx 1 => diracProb 3 (basePoint 2)) ∧
      Function.Injective (fun _ : Idx 1 => diracProb 3 (basePoint 2)) ∧
      (∃ w₀ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₀) ∧
      (∃ w₁ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₁) ∧
      ∀ i : Idx 1, ∃ Tr : SSphere 3 → SSphere 3, Measurable Tr ∧
        Measure.map Tr ((fun _ : Idx 1 => diracProb 3 (basePoint 2)) i : Measure (SSphere 3))
          = ((fun _ : Idx 1 => diracProb 3 (basePoint 2)) i : Measure (SSphere 3)) :=
  ⟨le_rfl, one_pos, one_pos, Function.injective_of_subsingleton _,
    Function.injective_of_subsingleton _, ⟨_, isHole_antipode_diracProb 3 1 (basePoint 2)⟩,
    ⟨_, isHole_antipode_diracProb 3 1 (basePoint 2)⟩,
    fun _ => ⟨id, measurable_id, Measure.map_id⟩⟩

/-- **Lemma (lem: univ.approx).** *Universal approximation of `L²` maps.*

Let `d ≥ 3`, `T, ε > 0` and `μ ∈ 𝒫(𝕊^{d-1})`.  For every measurable
`ψ : 𝕊^{d-1} → 𝕊^{d-1}` there are piecewise-constant `(𝐖, 𝐔, b)` on `[0, T]`
with finitely many switches such that the time-`T` map `φ^T` of the flow of
`eq: neural.ode.sphere` is Lipschitz-continuous and invertible, the solution of
`eq: neural.pde.sphere` from `μ` satisfies `μ(T) = φ^T_# μ`, and

  `‖ψ - φ^T‖_{L²(μ)} ≤ ε`.

**What the source says and what is changed here.**  The lemma is about the
perceptron part `eq: neural.pde.sphere` alone — `𝐕 ≡ 𝐁 ≡ 0`, `neuralParams` —
whose flow map is the one the proof of `thm: main.result` composes and inverts.
The former statement allowed the full vector field of `eq: vf`, attention
included, and asked neither for a Lipschitz nor an invertible map nor for
`μ(T) = φ^T_# μ`: a weaker claim under the source's name.  Three further points.
`d ≥ 3` is the standing dimension of every result the lemma serves, and it is
needed: on the circle a flow map is a homeomorphism, and no homeomorphism
approximates in `L²` of the uniform measure a map that wraps the circle three
times, `ψ(θ) = 3θ`; on `𝕊^0` every tangent projection vanishes and the flow is
the identity.  `T` is the time of `Φ^T_{θ_ε}` in the source, which does not
quantify it; every `T > 0` is asked.  The conclusion is `‖·‖_{L²(μ)} ≤ ε`, not
its square.  "The solution" is read as in `two_balls`: one exists and every
solution satisfies `μ(T) = φ^T_# μ`.

Not proved here.

Source: arXiv:2411.04551v3, §5, `lem: univ.approx`. -/
theorem univ_approx (μ : ProbSphere d) (ψ : SSphere d → SSphere d) (T ε : ℝ)
    (hd : 3 ≤ d) (hT : 0 < T) (hε : 0 < ε) (hψ : Measurable ψ) :
    ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (K : ℕ) (φ : ℝ → SSphere d → SSphere d),
      PiecewiseConstant d (neuralParams d W U b) T K ∧ IsNeuralFlow d W U b φ ∧
      (∃ L : NNReal, LipschitzWith L (φ T)) ∧ Function.Bijective (φ T) ∧
      (∃ μt : ℝ → ProbSphere d, μt 0 = μ ∧ cauchyPB d (neuralParams d W U b) μt) ∧
      (∀ μt : ℝ → ProbSphere d, μt 0 = μ → cauchyPB d (neuralParams d W U b) μt →
        (μt T : Measure (SSphere d)) = Measure.map (φ T) (μ : Measure (SSphere d))) ∧
      Real.sqrt (∫ x, ‖(φ T x : EucSpace d) - (ψ x : EucSpace d)‖ ^ 2
        ∂(μ : Measure (SSphere d))) ≤ ε := by
  sorry

/-- The hypotheses of `univ_approx` are satisfiable: `d = 3`, `T = ε = 1` and the
identity map, which is measurable. -/
example : 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Measurable (id : SSphere 3 → SSphere 3) :=
  ⟨le_rfl, one_pos, one_pos, measurable_id⟩

end Interpolation
end Transformer
