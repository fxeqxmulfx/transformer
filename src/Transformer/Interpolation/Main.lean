/-
# Measure-to-measure interpolation — Main theorems

Formalization of the main theorems of arXiv:2411.04551v3:

* `Theorem thm: targets.atoms`  — interpolation when targets are point masses,
* `Theorem thm: main.result`    — general interpolation,
* `Lemma lem: hyp.propagation`  — propagation of transport maps,
* `Lemma lem: monge`            — Monge-style optimal-transport identity,
* `Lemma lem: univ.approx`      — universal `L²`-map approximation.

Each of these asserts the existence of a parameter curve, or of a map, with no
construction available here, so none is proved: each is a theorem closed by
`sorry`.  The weakenings are the ones already used in
`Transformer.Interpolation.Clustering` and `…Disentanglement`: `W_2` is a
parameter (Mathlib has no Wasserstein
distance), `conv_g` is replaced by the support, and the switch and norm bounds
written `O(·)` in the paper carry an explicit constant as a parameter, whose
uniformity in `d`, `N` and the data is not expressible one statement at a time.

The transport maps of `thm: main.result` are required to be measurable rather
than to lie in `L²(𝕊^{d-1}; 𝕊^{d-1})`; on a sphere of finite measure and with
values in a bounded set the two agree, which is not proved here.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.Disentanglement
import Transformer.Interpolation.NeuralODE

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

Not proved here.

Source: arXiv:2411.04551v3, §1. -/
theorem targets_atoms (W₂ : Measure (SSphere d) → Measure (SSphere d) → ℝ) (C : ℕ)
    (μ₀ : Idx N → ProbSphere d) (xtarget : Idx N → SSphere d) (T ε : ℝ)
    (hd : 3 ≤ d) (hT : 0 < T) (hε : 0 < ε)
    (hhole : ∃ w₀ : SSphere d, IsHole d N μ₀ w₀) :
    ∃ (θ : TimeParams d) (K : ℕ) (Cnorm : ℝ) (μ : Idx N → ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
      (∀ s ∈ Set.Icc (0 : ℝ) T,
        ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
          ≤ Cnorm * ((d * N : ℝ) / T + Real.log (1 / ε))) ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ∀ i : Idx N, W₂ (μ i T : Measure (SSphere d)) (Measure.dirac (xtarget i)) ≤ ε := by
  sorry

/-- The hypotheses of `targets_atoms` are satisfiable: `d = 3`, `T = ε = 1`, and
a one-element family of Dirac masses on `𝕊^2`, whose hole is the antipode. -/
example :
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      ∃ w₀ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₀ :=
  ⟨le_rfl, one_pos, one_pos, _, isHole_antipode_diracProb 3 1 (basePoint 2)⟩

/-- **Theorem (thm: main.result).**  *General interpolation.*

For `d ≥ 3` and data `(μ_0^i, μ_1^i)_{i=1}^N` such that

* (eq: assumption.hole) some `w_0` misses every input support and some `w_1`
  misses every target support,
* each `μ_1^i` is the pushforward of `μ_0^i` under a measurable
  `𝖳^i : 𝕊^{d-1} → 𝕊^{d-1}`,

for any `T, ε > 0` there is a piecewise-constant `θ` with `O(d · N)` switches
such that `W_2(μ^i(T), μ_1^i) ≤ ε` for every `i`.

Not proved here.

Source: arXiv:2411.04551v3, §1. -/
theorem main_result (W₂ : Measure (SSphere d) → Measure (SSphere d) → ℝ) (C : ℕ)
    (μ₀ μ₁ : Idx N → ProbSphere d) (T ε : ℝ) (hd : 3 ≤ d) (hT : 0 < T) (hε : 0 < ε)
    (hhole₀ : ∃ w₀ : SSphere d, IsHole d N μ₀ w₀)
    (hhole₁ : ∃ w₁ : SSphere d, IsHole d N μ₁ w₁)
    (hpush : ∀ i : Idx N, ∃ Tr : SSphere d → SSphere d, Measurable Tr ∧
      Measure.map Tr (μ₀ i : Measure (SSphere d)) = (μ₁ i : Measure (SSphere d))) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ∀ i : Idx N, W₂ (μ i T : Measure (SSphere d)) (μ₁ i : Measure (SSphere d)) ≤ ε := by
  sorry

/-- The hypotheses of `main_result` are satisfiable: `d = 3`, `T = ε = 1`, and
the same one-element family of Dirac masses on both sides, which is its own
pushforward under the identity and has the antipode as a hole. -/
example :
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∃ w₀ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₀) ∧
      (∃ w₁ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₁) ∧
      ∀ i : Idx 1, ∃ Tr : SSphere 3 → SSphere 3, Measurable Tr ∧
        Measure.map Tr ((fun _ : Idx 1 => diracProb 3 (basePoint 2)) i : Measure (SSphere 3))
          = ((fun _ : Idx 1 => diracProb 3 (basePoint 2)) i : Measure (SSphere 3)) :=
  ⟨le_rfl, one_pos, one_pos, ⟨_, isHole_antipode_diracProb 3 1 (basePoint 2)⟩,
    ⟨_, isHole_antipode_diracProb 3 1 (basePoint 2)⟩,
    fun _ => ⟨id, measurable_id, Measure.map_id⟩⟩

/-- *Three-step factorization*:

  `Φ^T_fin := (Φ^{T/3}_{θ_3})⁻¹ ∘ Φ^{T/3}_{θ_2} ∘ Φ^{T/3}_{θ_1}`.

The three flows are arguments, since `Interpolation.IsFlowMap` is a property of
a candidate solution operator and not a construction of one; `Φ₃` here stands
for the inverse of the third flow, which is again a flow map. -/
noncomputable def threeStepFlow
    (Φ₁ Φ₂ Φ₃ : ℝ → ProbSphere d → ProbSphere d) (T : ℝ) (μ : ProbSphere d) :
    ProbSphere d :=
  Φ₃ (T/3) (Φ₂ (T/3) (Φ₁ (T/3) μ))

/-- **Lemma (lem: hyp.propagation).**  Propagation of transport maps along
the first and last flows.

If every `μ_1^i` is a measurable pushforward of `μ_0^i`, then there is a
bijective Lipschitz `ψ : 𝕊^{d-1} → 𝕊^{d-1}` with
`ψ_# Φ_1(μ_0^i) = Φ_3(μ_1^i)` for every `i`.

Not proved here.

Source: arXiv:2411.04551v3, §5. -/
theorem hyp_propagation (Φ₁ Φ₃ : ℝ → ProbSphere d → ProbSphere d) (t : ℝ)
    (μ₀ μ₁ : Idx N → ProbSphere d)
    (hpush : ∀ i : Idx N, ∃ Tr : SSphere d → SSphere d, Measurable Tr ∧
      Measure.map Tr (μ₀ i : Measure (SSphere d)) = (μ₁ i : Measure (SSphere d))) :
    ∃ (ψ : SSphere d → SSphere d) (L : NNReal),
      LipschitzWith L ψ ∧ Function.Bijective ψ ∧
      ∀ i : Idx N,
        Measure.map ψ (Φ₁ t (μ₀ i) : Measure (SSphere d))
          = (Φ₃ t (μ₁ i) : Measure (SSphere d)) := by
  sorry

/-- The hypothesis of `hyp_propagation` is satisfiable: any family is its own
pushforward under the identity. -/
example (μ₀ : Idx N → ProbSphere d) :
    ∀ i : Idx N, ∃ Tr : SSphere d → SSphere d, Measurable Tr ∧
      Measure.map Tr (μ₀ i : Measure (SSphere d)) = (μ₀ i : Measure (SSphere d)) :=
  fun _ => ⟨id, measurable_id, Measure.map_id⟩

/-- **Lemma (lem: monge).**  Monge identity: the Wasserstein distance between
two pushforwards of the same measure is controlled by the `L²(μ)` distance of
the maps,

  `W_2(S_# μ, ψ_# μ) ≤ Cst · ‖S - ψ‖_{L²(μ)}`,

which is how `W_2((Φ^{2T/3}_{θ_2})_# Φ^{T/3}_{θ_1}(μ_0^i), Φ_3^{T/3}(μ_1^i))`
is bounded in the proof — `μ` being `Φ^{T/3}_{θ_1}(μ_0^i)` and `ψ` the map of
`lem: hyp.propagation`.

Not proved here.

Source: arXiv:2411.04551v3, §5. -/
theorem monge (W₂ : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (μ : ProbSphere d) (S ψ : SSphere d → SSphere d) (Cst : ℝ)
    (hS : Measurable S) (hψ : Measurable ψ) :
    W₂ (Measure.map S (μ : Measure (SSphere d))) (Measure.map ψ (μ : Measure (SSphere d)))
      ≤ Cst * Real.sqrt
          (∫ x, ‖(S x : EucSpace d) - (ψ x : EucSpace d)‖ ^ 2 ∂(μ : Measure (SSphere d))) := by
  sorry

/-- The hypotheses of `monge` are satisfiable: the identity is measurable. -/
example : Measurable (id : SSphere d → SSphere d) ∧ Measurable (id : SSphere d → SSphere d) :=
  ⟨measurable_id, measurable_id⟩

/-- **Lemma (lem: univ.approx).** *Universal approximation of `L²` maps.*

For any measurable `f : 𝕊^{d-1} → 𝕊^{d-1}` and any `ε > 0` there is a
piecewise-constant `θ` whose time-`T` flow map approximates `f` to within `ε`
in `L²(μ)`.  The flow map is the endpoint map of the characteristics of
`eq: cauchy.pb`: for each `x` a curve started at `x` and driven by `eq: vf`
along the solution `μ(·)`.

Not proved here.

Source: arXiv:2411.04551v3, §5. -/
theorem univ_approx (μ : ProbSphere d) (f : SSphere d → SSphere d) (T ε : ℝ)
    (hT : 0 < T) (hε : 0 < ε) (hf : Measurable f) :
    ∃ (θ : TimeParams d) (K : ℕ) (μt : ℝ → ProbSphere d) (Φ : SSphere d → SSphere d),
      PiecewiseConstant d θ T K ∧ μt 0 = μ ∧ cauchyPB d θ μt ∧ Measurable Φ ∧
      (∀ x : SSphere d, ∃ γ : ℝ → EucSpace d,
        γ 0 = (x : EucSpace d) ∧ γ T = (Φ x : EucSpace d) ∧
        ∀ t : ℝ, HasDerivAt γ (fullVF d θ (μt t) t (γ t)) t) ∧
      ∫ x, ‖(Φ x : EucSpace d) - (f x : EucSpace d)‖ ^ 2 ∂(μ : Measure (SSphere d)) ≤ ε := by
  sorry

/-- The hypotheses of `univ_approx` are satisfiable: `T = ε = 1` and the
identity map, which is measurable. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Measurable (id : SSphere d → SSphere d) :=
  ⟨one_pos, one_pos, measurable_id⟩

end Interpolation
end Transformer
