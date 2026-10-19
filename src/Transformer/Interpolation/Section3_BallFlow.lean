module

public import Transformer.Basic
public import Transformer.Interpolation.Basic
public import Transformer.Interpolation.Clustering
public import Transformer.Interpolation.NeuralODE

/-!
# Measure-to-measure interpolation — Transporting mass through balls

Formalization of Appendix "Transporting mass through overlapping balls" of
arXiv:2411.04551v3:

* `Lemma lem: two.balls`             — pushing the mass of one ball into its
                                       overlap with a neighbour,
* `Lemma lem: tubular.mass.movement` — iterating along a chain of balls.

The two-ball and tubular statements use the shared original flow and
collapse operations formalized here.

Both lemmas use only the perceptron part of `eq: vf`, which is `eq: vf` with
`𝐕 ≡ 0`; the `eq: neural.pde.sphere` of the source is therefore `cauchyPB`
at `neuralParams`, and its flow map is that of `eq: neural.ode.sphere`
(`IsNeuralFlow`). The source's geodesic balls are ambient balls here: on the
unit sphere the two families coincide, only the radius being reparametrized.

**Quantifier order.**  The source chooses the parameters first, "such that for
any `μ_0 ∈ 𝒫(𝕊^{d-1})`", and its flow map `φ^t` is then fixed with them. In
that order both lemmas are false whenever the mass must actually move
(`BallTransportFalse`): a bijection which is the identity off `𝓑_0` maps `𝓑_0`
onto itself, so Dirac data forces `𝓑_0 ⊆ 𝓑_1`. The proof chooses its
radius `R - δ` after `μ_0`, and the lemmas are stated here in that order:
`μ_0` first, then the parameters.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory Set

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- `φ` is the flow map of `eq: neural.ode.sphere` for the parameters
`(𝐖, 𝐔, b)`: from every `x`, `t ↦ φ^t(x)` solves the equation, in integrated
form as `neuralODESphere`.

Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
def IsNeuralFlow (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (φ : ℝ → SSphere d → SSphere d) : Prop :=
  ∀ (x : SSphere d) (t : ℝ),
    IntervalIntegrable (fun s => neuralVF d W U b s (φ s x)) volume 0 t ∧
      (φ t x : EucSpace d) = x + ∫ s in (0 : ℝ)..t, neuralVF d W U b s (φ s x)

/-- At every time of `[0, T]` the map `φ^t` is Lipschitz, invertible, and the
identity outside `S`: the "moreover" clause shared by `lem: two.balls` and
`lem: tubular.mass.movement`.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
def IsTransportOutside (φ : ℝ → SSphere d → SSphere d) (S : Set (SSphere d)) (T : ℝ) :
    Prop :=
  ∀ t ∈ Set.Icc (0 : ℝ) T,
    (∃ L : NNReal, LipschitzWith L (φ t)) ∧ Function.Bijective (φ t) ∧
      ∀ x : SSphere d, x ∉ S → φ t x = x

/-- Constant parameters `(𝐕, 𝐁, 𝐖, 𝐔, b) ≡ (0, 0, 𝐖, 𝐔, b)`. -/
def constNeural (W U : ParamMatrix d) (b : EucSpace d) : TimeParams d :=
  neuralParams d (fun _ => W) (fun _ => U) (fun _ => b)

/-- The target of the last clause of `lem: two.balls`,
`α(A) = μ_0(𝓑_0) δ_ω(A) + μ_0(A ∖ 𝓑_0)`: the mass of `S` collapsed onto `ω`.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
noncomputable def collapseInto (μ₀ : Measure (SSphere d)) (S : Set (SSphere d))
    (ω : SSphere d) : Measure (SSphere d) :=
  μ₀ S • Measure.dirac ω + μ₀.restrict Sᶜ

section CollapseMaps

variable {d : ℕ}

/-- The measurable collapse used in the source's measure `α`.
Source: arXiv:2411.04551v3, Appendix, last clause of `lem: two.balls`. -/
noncomputable def collapseMap (S : Set (SSphere d)) (ω x : SSphere d) : SSphere d := by
  classical
  exact if x ∈ S then ω else x

/-- Collapsing a measurable region gives a measurable map.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem measurable_collapseMap (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d) :
    Measurable (collapseMap S ω) := by
  classical
  exact Measurable.ite hS measurable_const measurable_id

/-- The collapse pushforward is exactly the source's measure `α`, including
the untouched restricted measure. Source: arXiv:2411.04551v3,
Appendix, the final clause of `lem: two.balls`. -/
theorem map_collapseMap (μ : Measure (SSphere d)) (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d) :
    Measure.map (collapseMap S ω) μ = collapseInto d μ S ω := by
  classical
  ext A hA
  rw [Measure.map_apply (measurable_collapseMap S hS ω) hA, collapseInto,
    Measure.add_apply, Measure.smul_apply, Measure.restrict_apply hA, Measure.dirac_apply' ω hA]
  by_cases hω : ω ∈ A
  · have he : collapseMap S ω ⁻¹' A = S ∪ (A ∩ Sᶜ) := by
      ext x
      by_cases hx : x ∈ S <;> simp [collapseMap, hx, hω]
    rw [he, measure_union (by exact disjoint_left.mpr (fun x hx hx' => hx'.2 hx)) (hA.inter hS.compl)]
    simp [hω]
  · have he : collapseMap S ω ⁻¹' A = A ∩ Sᶜ := by
      ext x
      by_cases hx : x ∈ S <;> simp [collapseMap, hx, hω]
    rw [he]
    simp [hω]

/-- A genuine spherical ball realizes the measurable-collapse hypothesis. -/
example : MeasurableSet (Metric.ball (basePoint 1) 1 : Set (SSphere 2)) :=
  Metric.isOpen_ball.measurableSet

end CollapseMaps

section FlowProperties

variable {d : ℕ}

/-- The original integrated particle equation has its stated initial
point. Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem IsNeuralFlow.zero (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (φ : ℝ → SSphere d → SSphere d) (hφ : IsNeuralFlow d W U b φ) (x : SSphere d) :
    φ 0 x = x := by
  apply Subtype.ext
  simpa only [intervalIntegral.integral_same, add_zero] using (hφ x 0).2

/-- Original particle velocities are integrable on every finite interval.
Auxiliary for arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem IsNeuralFlow.intervalIntegrable (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (φ : ℝ → SSphere d → SSphere d) (hφ : IsNeuralFlow d W U b φ)
    (x : SSphere d) (a c : ℝ) :
    IntervalIntegrable (fun s => neuralVF d W U b s (φ s x)) volume a c :=
  (hφ x a).1.symm.trans (hφ x c).1

/-- Enlarging the allowed transport region retains its fixed-complement
clause. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the growing ball union. -/
theorem IsTransportOutside.mono (φ : ℝ → SSphere d → SSphere d)
    (S A : Set (SSphere d)) (T : ℝ) (hφ : IsTransportOutside d φ S T) (hSA : S ⊆ A) :
    IsTransportOutside d φ A T := by
  intro t ht
  obtain ⟨hL, hb, hfix⟩ := hφ t ht
  exact ⟨hL, hb, fun x hx => hfix x (fun hs => hx (hSA hs))⟩

/-- A shorter nonnegative-time interval retains the same original
transport properties. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, its prefix controls. -/
theorem IsTransportOutside.restrictTime (φ : ℝ → SSphere d → SSphere d)
    (S : Set (SSphere d)) (T T' : ℝ) (hφ : IsTransportOutside d φ S T) (hT : T' ≤ T) :
    IsTransportOutside d φ S T' := by
  intro t ht
  exact hφ t ⟨ht.1, ht.2.trans hT⟩

/-- The true zero neural equation and identity particle map jointly
realize the flow, region-inclusion and shorter-horizon hypotheses. -/
example : IsNeuralFlow 2 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ x => x) ∧
    IsTransportOutside 2 (fun _ x => x) ∅ 2 ∧
    (∅ : Set (SSphere 2)) ⊆ Metric.ball (basePoint 1) 1 ∧ (1 : ℝ) ≤ 2 := by
  refine ⟨?_, fun _ _ => ⟨⟨1, LipschitzWith.id⟩, Function.bijective_id, fun _ _ => rfl⟩,
    empty_subset _, by norm_num⟩
  intro x t
  have hzero : (fun s => neuralVF 2 (fun _ => 0) (fun _ => 0) (fun _ => 0) s x) = fun _ => 0 := by
    funext s
    simp [neuralVF, proj]
  rw [hzero, intervalIntegral.integral_zero, add_zero]
  exact ⟨intervalIntegrable_const, rfl⟩

end FlowProperties

end Interpolation
end Transformer
