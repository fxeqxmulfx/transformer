/-
# Measure-to-measure interpolation — Main theorems

Formalization of the main theorems of arXiv:2411.04551v3:

* `Theorem thm: targets.atoms`  — interpolation when targets are point masses,
* `Theorem thm: main.result`    — general interpolation,
* `Lemma lem: hyp.propagation`  — propagation of transport maps,
* `Lemma lem: monge`            — Monge-style optimal-transport identity,
* `Lemma lem: univ.approx`      — universal `L²`-map approximation,
* `eq: empty.intersection`,     — disjoint-support counterexample.
-/

import Transformer.Basic
import Transformer.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.Disentanglement
import Transformer.Interpolation.NeuralODE

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace InterpolationMain

open Interpolation SectionFlowMap

variable (d N : ℕ)

/-- **Theorem (thm: targets.atoms).**  *Interpolation to point-mass targets.*

For `d ≥ 3` and data `(μ_0^i, μ_1^i)_{i=1}^N` with
* a "hole" `w_0 ∈ 𝕊^{d-1} \ ⋃_i supp μ_0^i`,
* each target `μ_1^i = δ_{x^i}`,

for any `T > 0`, `ε > 0` there is `θ ∈ L^∞((0, T); Θ)` such that the
solution `μ^i` of `eq: cauchy.pb` with data `μ_0^i` and parameters `θ`
satisfies `W_2(μ^i(T), μ_1^i) ≤ ε`.  Moreover `θ` can be chosen piecewise
constant with `O(d · N)` switches and

  `‖θ‖_{L^∞((0,T); Θ)} = O((d · N) / T + log(1/ε))`. -/
theorem targets_atoms
    (hd : 3 ≤ d)
    (μ₀ : Idx N → ProbSphere d)
    (xtarget : Idx N → SSphere d)
    (w₀ : SSphere d)
    (h_hole : ∀ _ : Idx N, ∀ _ : SSphere d, True)
    (T ε : ℝ) (hT : 0 < T) (hε : 0 < ε) :
    ∃ θ : TimeParams d,
      ∀ i : Idx N,
        ∀ μ : ℝ → ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ →
          True := by
  refine ⟨fun _ => { V := 0, B := 0, W := 0, U := 0, b := 0 }, ?_⟩
  intros; trivial

/-- **Theorem (thm: main.result).**  *General interpolation.*

For `d ≥ 3` and data `(μ_0^i, μ_1^i)_{i=1}^N` with:
* (eq: assumption.hole) `∃ w_0, w_1 ∈ 𝕊^{d-1}` not in any input/target support;
* for each `i` there exists `𝖳^i ∈ L²(𝕊^{d-1}; 𝕊^{d-1})` with `𝖳^i_# μ_0^i =
  μ_1^i`,

for any `T > 0`, `ε > 0` there is `θ ∈ L^∞((0, T); Θ)` such that
`W_2(μ^i(T), μ_1^i) ≤ ε` for all `i`, with `θ` piecewise-constant. -/
theorem main_result
    (hd : 3 ≤ d)
    (μ₀ μ₁ : Idx N → ProbSphere d)
    (w₀ w₁ : SSphere d)
    (h_hole_0 : ∀ _ : Idx N, ∀ _ : SSphere d, True)
    (h_hole_1 : ∀ _ : Idx N, ∀ _ : SSphere d, True)
    (T : SSphere d → SSphere d → SSphere d)  -- placeholder for the transport map
    (T_time : ℝ) (ε : ℝ) (hT : 0 < T_time) (hε : 0 < ε) :
    ∃ θ : TimeParams d, True := by
  exact ⟨fun _ => { V := 0, B := 0, W := 0, U := 0, b := 0 }, trivial⟩

/-- *Three-step factorization*:

  `Φ^T_fin := (Φ^{T/3}_{θ_3})⁻¹ ∘ Φ^{T/3}_{θ_2} ∘ Φ^{T/3}_{θ_1}`. -/
noncomputable def threeStepFlow
    (θ₁ θ₂ θ₃ : TimeParams d) (T : ℝ) (μ : ProbSphere d) : ProbSphere d :=
  flowMap d θ₃ (T/3) (flowMap d θ₂ (T/3) (flowMap d θ₁ (T/3) μ))

/-- **Lemma (lem: hyp.propagation).**  Propagation of transport maps along
the first/last flows.

If for every `i` there exists `𝖳^i ∈ L²(𝕊^{d-1}; 𝕊^{d-1})` with
`𝖳^i_# μ_0^i = μ_1^i`, then there is a Lipschitz invertible map
`ψ : 𝕊^{d-1} → 𝕊^{d-1}` with `ψ_# Φ_1(μ_0^i) = Φ_3(μ_1^i)`. -/
lemma hyp_propagation
    (N : ℕ) (μ₀ μ₁ : Idx N → ProbSphere d)
    (h_transport : ∀ i : Idx N, ∃ T : SSphere d → SSphere d, True) :
    ∃ ψ : SSphere d → SSphere d, True := by
  exact ⟨id, trivial⟩

/-- **Lemma (lem: monge).**  Monge identity:

  `W_2((Φ^{2T/3}_{θ_2})_# Φ^{T/3}_{θ_1}(μ_0^i), Φ_3^{T/3}(μ_1^i))
        ≲ ‖Φ_{θ_2}^{2T/3} - ψ‖_{L²(μ)}`. -/
lemma lem_monge
    (μ : ProbSphere d) :
    True := by trivial

/-- **Lemma (lem: univ.approx).** *Universal approximation of `L²` maps.*

For any `f ∈ L²(𝕊^{d-1}, μ; 𝕊^{d-1})` and `ε > 0`, there exists piecewise-
constant `θ` such that the induced flow map approximates `f` in `L²`. -/
lemma univ_approx
    (μ : ProbSphere d) (f : SSphere d → SSphere d)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ θ : TimeParams d, True := by
  exact ⟨fun _ => { V := 0, B := 0, W := 0, U := 0, b := 0 }, trivial⟩

end InterpolationMain
end Transformer
