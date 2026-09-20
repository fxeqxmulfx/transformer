/-
# Homogenized Transformers — the random chain and the weight law

Formalization of `ass:high_order_short` and of the randomness carried by the
discrete chain `eq:update_tokens` of arXiv:2604.01978v1, *Homogenized
Transformers* (Geshkovski, Koubbi, Rigollet).

Two hypotheses are written out here, both as genuine predicates of their
arguments rather than as claims.

* `IsHighOrderLaw` is `ass:high_order_short`: the weight law `ρ*` is the law of
  `(V, A)` where `A - 𝔼A = W W'ᵀ` is independent of `V`, and the entries of
  `V - 𝔼V`, of `W` and of `W'` are independent subGaussian with variance
  proxies `C σ_V²` and `C σ_A²`.  The source space carrying `V, W, W'` is part
  of the data; `HasHighOrderLaw` existentially closes over it, which is how the
  assumption reads in the manuscript ("whenever `(A, V) ∼ ρ*`").

* `IsRandomChain` is the chain of `eq:update_tokens` driven by heads
  `θ^ℓ_h` drawn i.i.d. from `ρ*`: the independence is over the pair `(ℓ, h)` of
  layer and head, which is the only place randomness enters the discrete model.

Source: arXiv:2604.01978v1, §2.3, `ass:high_order_short`, `eq:update_tokens`.
-/

import Transformer.Homogenized.Basic
import Mathlib.Probability.Moments.SubGaussian

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- On a one-point probability space every family of functions is independent:
each preimage is either everything or nothing, so the measure of an
intersection is the product of the measures.

This is the witness that makes the independence hypotheses of
`ass:high_order_short` satisfiable at the degenerate weight law `δ_0`. -/
theorem iIndepFun_of_unit {ι : Type*} {β : ι → Type*}
    [∀ i, MeasurableSpace (β i)] (f : ∀ i, Unit → β i) :
    iIndepFun f (Measure.dirac ()) := by
  classical
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
  intro s g _
  have key : ∀ A : Set Unit, Measure.dirac () A = if () ∈ A then 1 else 0 := by
    intro A
    rw [Measure.dirac_apply]
    simp [Set.indicator_apply]
  rw [key, Finset.prod_congr rfl (fun i _ => key (f i ⁻¹' g i))]
  by_cases h : ∀ i ∈ s, () ∈ f i ⁻¹' g i
  · rw [ite_eq_left (by simpa using h)]
    exact (Finset.prod_eq_one fun i hi => ite_eq_left (h i hi)).symm
  · obtain ⟨i, hi, hni⟩ : ∃ i ∈ s, () ∉ f i ⁻¹' g i := by
      by_contra hc
      exact h fun i hi => by
        by_contra hni
        exact hc ⟨i, hi, hni⟩
    rw [ite_eq_right (fun hall => hni (Set.mem_iInter₂.mp hall i hi))]
    exact (Finset.prod_eq_zero hi (ite_eq_right hni)).symm

/-- **Assumption (ass:high_order_short)**, as a property of a concrete source
space `Ω` carrying the centered parts `V - 𝔼V`, `W`, `W'` and of the means
`mV = 𝔼V`, `mA = 𝔼A`.

`(A, V) ∼ ρ*` with `A - 𝔼A = W W'ᵀ` independent of `V`; the entries of
`V - 𝔼V` are independent subGaussian with variance proxy `C σ_V²`, and the
entries of `W` and `W'` together are independent subGaussian with variance
proxy `C σ_A²`.

Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
structure IsHighOrderLaw (d : ℕ) (C σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
    (mV mA : Matrix (Fin d) (Fin d) ℝ) : Prop where
  /-- `P` is a probability measure. -/
  prob : IsProbabilityMeasure P
  measV : Measurable Vr
  measW : Measurable Wr
  measW' : Measurable Wr'
  /-- `ρ*` is the law of `(𝔼V + (V - 𝔼V), 𝔼A + W W'ᵀ)`. -/
  pushforward :
    Measure.map (fun ω => ((mV + Vr ω, mA + Wr ω * (Wr' ω).transpose) : HeadParam d)) P = ρ
  /-- `V - 𝔼V` is centered, which is what makes `mV` the mean. -/
  centered : ∀ i j, ∫ ω, Vr ω i j ∂P = 0
  /-- `A - 𝔼A = W W'ᵀ` is independent of `V`. -/
  indepVA : IndepFun Vr (fun ω => (Wr ω, Wr' ω)) P
  /-- the entries of `V - 𝔼V` are independent of each other. -/
  indepV : iIndepFun (fun p : Fin d × Fin d => fun ω => Vr ω p.1 p.2) P
  /-- the entries of `W` and `W'` are independent of each other. -/
  indepW : iIndepFun
    (fun p : Bool × Fin d × Fin d => fun ω =>
      if p.1 then Wr ω p.2.1 p.2.2 else Wr' ω p.2.1 p.2.2) P
  subgaussianV : ∀ i j, HasSubgaussianMGF (fun ω => Vr ω i j) (C * σV ^ 2) P
  subgaussianW : ∀ i j, HasSubgaussianMGF (fun ω => Wr ω i j) (C * σA ^ 2) P
  subgaussianW' : ∀ i j, HasSubgaussianMGF (fun ω => Wr' ω i j) (C * σA ^ 2) P

/-- `ass:high_order_short` as a property of the weight law `ρ*` alone: some
source space, some decomposition, some constant `C > 0` realizes it. -/
def HasHighOrderLaw (d : ℕ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) : Prop :=
  ∃ (C : ℝ≥0) (Ω : Type) (_ : MeasurableSpace Ω) (P : Measure Ω)
    (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ) (mV mA : Matrix (Fin d) (Fin d) ℝ),
    0 < C ∧ IsHighOrderLaw d C σV σA ρ P Vr Wr Wr' mV mA

/-- The degenerate weight law `ρ* = δ_0` satisfies `ass:high_order_short`, with
all three centered parts equal to `0` on a one-point space and both scales
`σ_V = σ_A = 0`.  This is the witness that the assumption is not contradictory.

Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem hasHighOrderLaw_dirac_zero (d : ℕ) :
    HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) := by
  refine ⟨1, Unit, inferInstance, Measure.dirac (), 0, 0, 0, 0, 0, one_pos, ?_⟩
  refine ⟨inferInstance, measurable_const, measurable_const, measurable_const,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
    rfl
  · intro i j; simp
  · exact indepFun_const_left _ _
  · exact iIndepFun_of_unit _
  · exact iIndepFun_of_unit _
  · intro i j; simp
  · intro i j; simp
  · intro i j; simp

/-- The discrete chain `eq:update_tokens` driven by heads drawn i.i.d. from the
weight law: `θ^ℓ = (θ^ℓ_1, …, θ^ℓ_H)` with all `θ^ℓ_h` independent of law `ρ*`,
and `X^0 = x₀` deterministic.

Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`. -/
structure IsRandomChain {d n H : ℕ} (η β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Θ : ℕ → Idx H → Ω → HeadParam d)
    (X : Ω → ℕ → Idx n → EucSpace d) (x₀ : Idx n → EucSpace d) : Prop where
  prob : IsProbabilityMeasure P
  meas : ∀ ℓ h, Measurable (Θ ℓ h)
  /-- the heads are independent across layers and across heads. -/
  indep : iIndepFun (fun p : ℕ × Idx H => Θ p.1 p.2) P
  /-- each head has law `ρ*`. -/
  law : ∀ ℓ h, Measure.map (Θ ℓ h) P = ρ
  chain : ∀ ω, IsLayerChain η β (fun ℓ h => Θ ℓ h ω) (X ω)
  init : ∀ ω, X ω 0 = x₀

/-- The degenerate chain: under `ρ* = δ_0` every head is the zero head, the
attention field vanishes, and the update of `eq:update_tokens` leaves a unit
tuple where it is.  So `IsRandomChain` is satisfiable.

Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`. -/
theorem isRandomChain_dirac_zero {d n H : ℕ} (η β : ℝ) (x₀ : Idx n → EucSpace d)
    (hx₀ : ∀ i, ‖x₀ i‖ = 1) :
    IsRandomChain (H := H) η β (Measure.dirac (0 : HeadParam d)) (Measure.dirac ())
      (fun _ _ _ => 0) (fun _ _ => x₀) x₀ := by
  refine ⟨inferInstance, fun _ _ => measurable_const, iIndepFun_of_unit _, ?_, ?_,
    fun _ => rfl⟩
  · intro ℓ h
    simp
  · intro ω
    exact ⟨fun _ i => hx₀ i, fun ℓ i => by
      simp [layerUpdate, normalizeLayer_of_norm_one (hx₀ i)]⟩

/-- The hypothesis of `isRandomChain_dirac_zero` is satisfiable. -/
example (d : ℕ) : ∀ _i : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  fun _ => by simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
