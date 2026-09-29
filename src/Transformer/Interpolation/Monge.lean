/-
# Measure-to-measure interpolation — the Monge-style identity

`Lemma lem: monge` of arXiv:2411.04551v3, §5, proved against the `W_2` of
`Transformer.Interpolation.Wasserstein`, and the refutation of its former form with a free
distance function.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Wasserstein

open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Perspective

variable (d : ℕ)

/-- **Lemma (lem: monge).**  Monge identity: the Wasserstein distance between
two pushforwards of the same measure is controlled by the `L²(μ)` distance of
the maps,

  `W_2(S_# μ, ψ_# μ) ≤ ‖S - ψ‖_{L²(μ)}`,

which is how `W_2((Φ^{2T/3}_{θ_2})_# Φ^{T/3}_{θ_1}(μ_0^i), Φ_3^{T/3}(μ_1^i))`
is bounded in the proof — `μ` being `Φ^{T/3}_{θ_1}(μ_0^i)` and `ψ` the map of
`lem: hyp.propagation`.

**What the source says and what is changed here.**  Two things, and both are
what makes this a theorem rather than a request.

*`W_2` is `Interpolation.W2`, not a parameter.*  Carried as a free function,
as it was here, the inequality is false: `not_forall_monge` refutes it below.
That is not an accident of the constant — the statement *is* a defining
property of the 2-Wasserstein distance, so asking it of an arbitrary function
of two measures asks for something no hypothesis in the binders supplies.  The
distance is therefore defined, in `Interpolation.Wasserstein`, as the infimum
of the quadratic transport cost over couplings, and the lemma is proved: the
map `x ↦ (S x, ψ x)` pushes `μ` to a coupling of `S_# μ` and `ψ_# μ` whose
cost is exactly the `L²(μ)` distance of the two maps.

*The constant is `1`.*  The paper writes `≲`; the proof gives `1`, which is
the sharp value, so nothing is lost by writing it.  Carried as a parameter, as
`Cst` was here, it was again a free variable in the direction that makes the
claim false.

*The paper's bijectivity of `S` is dropped.*  `lem: monge` assumes the first
map invertible; the proof uses nothing of it, so the hypothesis goes and the
statement is the stronger one.

Source: arXiv:2411.04551v3, §5, `lem: monge`. -/
theorem monge (μ : ProbSphere d) (S ψ : SSphere d → SSphere d)
    (hS : Measurable S) (hψ : Measurable ψ) :
    W2 d (Measure.map S (μ : Measure (SSphere d))) (Measure.map ψ (μ : Measure (SSphere d)))
      ≤ Real.sqrt
          (∫ x, ‖(S x : EucSpace d) - (ψ x : EucSpace d)‖ ^ 2 ∂(μ : Measure (SSphere d))) := by
  have hpair : Measurable (fun x : SSphere d => (S x, ψ x)) := hS.prodMk hψ
  set γ : Measure (SSphere d × SSphere d) :=
    Measure.map (fun x : SSphere d => (S x, ψ x)) (μ : Measure (SSphere d)) with hγdef
  have hcoup : IsCoupling d (Measure.map S (μ : Measure (SSphere d)))
      (Measure.map ψ (μ : Measure (SSphere d))) γ := by
    constructor <;>
      rw [hγdef, Measure.map_map (by fun_prop) hpair] <;> rfl
  have hmeasf : AEStronglyMeasurable (fun p : SSphere d × SSphere d => dist p.1 p.2 ^ 2) γ := by
    fun_prop
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂γ
      = ∫ x, ‖(S x : EucSpace d) - (ψ x : EucSpace d)‖ ^ 2 ∂(μ : Measure (SSphere d)) := by
    rw [hγdef, integral_map hpair.aemeasurable hmeasf]
    exact integral_congr_ae (Filter.Eventually.of_forall fun x => by
      simp only [Subtype.dist_eq, dist_eq_norm])
  have := W2_le_of_coupling d _ _ γ hcoup
  rwa [hcost] at this

/-- The hypotheses of `monge` are satisfiable: the identity is measurable. -/
example : Measurable (id : SSphere d → SSphere d) ∧ Measurable (id : SSphere d → SSphere d) :=
  ⟨measurable_id, measurable_id⟩

/-- **`lem: monge` is false for a free `W_2` and a free constant.**

Read with `W₂` and `Cst` as universally quantified binders, as the statement
stood here before, the lemma claims an inequality about a function of two
measures of which nothing is assumed.  `W₂ ≡ 1` and `Cst = 0` with `S = ψ`
give `1 ≤ 0`.  This is what `monge` above repairs by defining `W_2`.

Source: arXiv:2411.04551v3, §5, `lem: monge`. -/
theorem not_forall_monge :
    ¬ ∀ (d : ℕ) (W₂ : Measure (SSphere d) → Measure (SSphere d) → ℝ)
        (μ : ProbSphere d) (S ψ : SSphere d → SSphere d) (Cst : ℝ),
        Measurable S → Measurable ψ →
        W₂ (Measure.map S (μ : Measure (SSphere d))) (Measure.map ψ (μ : Measure (SSphere d)))
          ≤ Cst * Real.sqrt
              (∫ x, ‖(S x : EucSpace d) - (ψ x : EucSpace d)‖ ^ 2
                ∂(μ : Measure (SSphere d))) := by
  intro h
  have := h 1 (fun _ _ => 1) (diracProb 1 (basePoint 0)) id id 0 measurable_id measurable_id
  norm_num at this

end Interpolation
end Transformer
