/-
# The 2-Wasserstein distance

Mathlib carries no Wasserstein distance, and two of the papers formalized here
are about one, so it is defined once: the square root of the infimum of the
quadratic transport cost over the couplings of the two measures,

  `W_2(μ, ν)² = inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ(x, y)`.

The infimum is a `sInf` over the set of costs, which is bounded below by `0`
and, for probability measures, nonempty — the diagonal coupling is one when
`μ = ν`, and the product coupling always is.  That is all the papers' proofs
use of the definition: an upper bound on `W_2` from one exhibited coupling,
and, the other way, `‖∫ f dμ - ∫ f dν‖ ≤ L W_2(μ, ν)` for an `L`-Lipschitz `f`
on a bounded set carrying both measures (`norm_integral_sub_le_W2`).

The ambient space is any pseudometric measurable space.
arXiv:2411.04551v3 reads it on `𝕊^{d-1}` and arXiv:2604.01978v1 on `ℝ^d`
against measures carried by the sphere; the distance is the same chordal
`‖x - y‖` in both, so there is one object here and not two.

Sources: arXiv:2411.04551v3, §1 (`W_2` throughout); arXiv:2604.01978v1,
`prop:satisfying_MF`, `prop: poc`.
-/

import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Map
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.Convex.Mul

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Wasserstein

variable {X : Type*} [MeasurableSpace X] [PseudoMetricSpace X]

/-- `γ` is a *coupling* of `μ` and `ν`: a measure on `X × X` whose two
marginals are `μ` and `ν`. -/
def IsCoupling (μ ν : Measure X) (γ : Measure (X × X)) : Prop :=
  Measure.map Prod.fst γ = μ ∧ Measure.map Prod.snd γ = ν

/-- The quadratic transport costs `∫ d(x, y)² dγ` of the couplings `γ` of `μ`
and `ν`. -/
noncomputable def transportCosts (μ ν : Measure X) : Set ℝ :=
  { c | ∃ γ : Measure (X × X), IsCoupling μ ν γ ∧ ∫ p, dist p.1 p.2 ^ 2 ∂γ = c }

/-- **The 2-Wasserstein distance**:

  `W_2(μ, ν) = (inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ)^{1/2}`. -/
noncomputable def W2 (μ ν : Measure X) : ℝ :=
  Real.sqrt (sInf (transportCosts μ ν))

/-- Every transport cost is nonnegative, the cost being an integral of a
nonnegative function. -/
theorem transportCosts_nonneg (μ ν : Measure X) :
    ∀ c ∈ transportCosts μ ν, 0 ≤ c := by
  rintro c ⟨γ, -, rfl⟩
  exact integral_nonneg fun p => by positivity

/-- The costs are bounded below, by `0`. -/
theorem bddBelow_transportCosts (μ ν : Measure X) :
    BddBelow (transportCosts μ ν) :=
  ⟨0, transportCosts_nonneg μ ν⟩

/-- `W_2` is nonnegative, being a square root. -/
theorem W2_nonneg (μ ν : Measure X) : 0 ≤ W2 μ ν :=
  Real.sqrt_nonneg _

/-- **One coupling bounds `W_2` from above.**  This is the only property of the
definition the papers' proofs use. -/
theorem W2_le_of_coupling {μ ν : Measure X} (γ : Measure (X × X))
    (hγ : IsCoupling μ ν γ) :
    W2 μ ν ≤ Real.sqrt (∫ p, dist p.1 p.2 ^ 2 ∂γ) :=
  Real.sqrt_le_sqrt (csInf_le (bddBelow_transportCosts μ ν) ⟨γ, hγ, rfl⟩)

/-- The hypothesis of `W2_le_of_coupling` is satisfiable: a Dirac mass on the
diagonal couples a Dirac mass with itself. -/
example (x : ℝ) : IsCoupling (Measure.dirac x) (Measure.dirac x) (Measure.dirac (x, x)) := by
  constructor <;> simp [measurable_fst, measurable_snd]

/-- **The product coupling**: two probability measures always have one, so the
infimum defining `W_2` is over a nonempty set. -/
theorem transportCosts_nonempty (μ ν : Measure X) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] : (transportCosts μ ν).Nonempty :=
  ⟨_, μ.prod ν, ⟨by simp, by simp⟩, rfl⟩

omit [PseudoMetricSpace X] in
/-- A coupling of `μ` with `ν` puts almost every first coordinate where `μ`
lives. -/
theorem ae_fst_mem {μ ν : Measure X} {γ : Measure (X × X)} (hγ : IsCoupling μ ν γ)
    {s : Set X} (hs : MeasurableSet s) (hμ : μ sᶜ = 0) : ∀ᵐ p ∂γ, p.1 ∈ s := by
  rw [ae_iff]
  have := Measure.map_apply measurable_fst hs.compl (μ := γ)
  rw [hγ.1] at this
  rw [← hμ, this]; rfl

omit [PseudoMetricSpace X] in
/-- A coupling of `μ` with `ν` puts almost every second coordinate where `ν`
lives. -/
theorem ae_snd_mem {μ ν : Measure X} {γ : Measure (X × X)} (hγ : IsCoupling μ ν γ)
    {s : Set X} (hs : MeasurableSet s) (hν : ν sᶜ = 0) : ∀ᵐ p ∂γ, p.2 ∈ s := by
  rw [ae_iff]
  have := Measure.map_apply measurable_snd hs.compl (μ := γ)
  rw [hγ.2] at this
  rw [← hν, this]; rfl

section Diagonal

/-! The diagonal coupling needs `d(·,·)` to be measurable for the product
σ-algebra, which is what these two instances buy; the definitions above do not
need them, and neither do the papers' uses of the upper bound. -/
variable [OpensMeasurableSpace X] [SecondCountableTopology X]

/-- **The diagonal coupling**: `W_2(μ, μ) = 0` for a probability measure `μ`.
The map `x ↦ (x, x)` pushes `μ` to a coupling of `μ` with itself carrying no
cost, which is what makes `W2` a distance-like quantity rather than an
artefact of the `sInf`. -/
theorem W2_self (μ : Measure X) [IsProbabilityMeasure μ] : W2 μ μ = 0 := by
  have hdiag : Measurable (fun x : X => (x, x)) := by fun_prop
  have hcoup : IsCoupling μ μ (Measure.map (fun x : X => (x, x)) μ) := by
    constructor <;>
      rw [Measure.map_map (by fun_prop) hdiag] <;> simp [Function.comp_def]
  have hle := W2_le_of_coupling _ hcoup
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂(Measure.map (fun x : X => (x, x)) μ) = 0 := by
    have hmeasf : AEStronglyMeasurable (fun p : X × X => dist p.1 p.2 ^ 2)
        (Measure.map (fun x : X => (x, x)) μ) := by fun_prop
    rw [integral_map hdiag.aemeasurable hmeasf]
    simp
  rw [hcost, Real.sqrt_zero] at hle
  exact le_antisymm hle (W2_nonneg μ μ)

/-- The hypothesis of `W2_self` is satisfiable. -/
example : IsProbabilityMeasure (Measure.dirac (0 : ℝ)) := inferInstance

/-- **`W_2` controls integrals of Lipschitz functions**: if `μ` and `ν` live
on a bounded set on which `f` is `L`-Lipschitz, then

  `‖∫ f dμ - ∫ f dν‖ ≤ L W_2(μ, ν)`.

For each coupling `γ`, `‖∫ f dμ - ∫ f dν‖ ≤ L ∫ d dγ ≤ L (∫ d² dγ)^{1/2}` by
Jensen; the infimum over `γ` follows.  The bounded set makes every cost the
integral of a bounded function, which is what keeps the Bochner integral
honest here.

This is `W_1 ≤ W_2` together with the easy half of Kantorovich–Rubinstein, as
used in arXiv:2305.05465v6, proof of `e:lipinmu`. -/
theorem norm_integral_sub_le_W2 {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (μ ν : Measure X) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {s : Set X} (hs : MeasurableSet s) (hsb : Bornology.IsBounded s)
    (hμ : μ sᶜ = 0) (hν : ν sᶜ = 0) (f : X → E) (hfμ : Integrable f μ) (hfν : Integrable f ν)
    {L : ℝ} (hL : 0 ≤ L) (hf : ∀ x ∈ s, ∀ y ∈ s, ‖f x - f y‖ ≤ L * dist x y) :
    ‖∫ x, f x ∂μ - ∫ x, f x ∂ν‖ ≤ L * W2 μ ν := by
  set B := ‖∫ x, f x ∂μ - ∫ x, f x ∂ν‖
  have hc : ∀ c ∈ transportCosts μ ν, B ≤ L * Real.sqrt c := by
    rintro c ⟨γ, hγ, rfl⟩
    have hγ1 : IsProbabilityMeasure γ := ⟨by
      have := Measure.map_apply measurable_fst MeasurableSet.univ (μ := γ)
      rw [hγ.1] at this; simpa using this.symm⟩
    have h1 : Integrable (fun p : X × X => f p.1) γ :=
      (hγ.1 ▸ hfμ).comp_measurable measurable_fst
    have h2 : Integrable (fun p : X × X => f p.2) γ :=
      (hγ.2 ▸ hfν).comp_measurable measurable_snd
    have e1 : ∫ x, f x ∂μ = ∫ p, f p.1 ∂γ := by
      rw [← hγ.1, integral_map measurable_fst.aemeasurable (hγ.1 ▸ hfμ.aestronglyMeasurable)]
    have e2 : ∫ x, f x ∂ν = ∫ p, f p.2 ∂γ := by
      rw [← hγ.2, integral_map measurable_snd.aemeasurable (hγ.2 ▸ hfν.aestronglyMeasurable)]
    have hae : ∀ᵐ p ∂γ, p.1 ∈ s ∧ p.2 ∈ s :=
      (ae_fst_mem hγ hs hμ).and (ae_snd_mem hγ hs hν)
    have hd : Integrable (fun p : X × X => dist p.1 p.2) γ := by
      refine Integrable.of_bound (by fun_prop) (Metric.diam s) ?_
      filter_upwards [hae] with p hp
      rw [Real.norm_of_nonneg dist_nonneg]
      exact Metric.dist_le_diam_of_mem hsb hp.1 hp.2
    have hd2 : Integrable (fun p : X × X => dist p.1 p.2 ^ 2) γ := by
      refine Integrable.of_bound (by fun_prop) (Metric.diam s ^ 2) ?_
      filter_upwards [hae] with p hp
      rw [Real.norm_of_nonneg (by positivity)]
      exact pow_le_pow_left₀ dist_nonneg (Metric.dist_le_diam_of_mem hsb hp.1 hp.2) 2
    have hB : B ≤ L * ∫ p, dist p.1 p.2 ∂γ := by
      simp only [B]
      rw [e1, e2, ← integral_sub h1 h2, ← integral_const_mul]
      refine norm_integral_le_of_norm_le (hd.const_mul L) ?_
      filter_upwards [hae] with p hp
      exact hf _ hp.1 _ hp.2
    have hJ : (∫ p, dist p.1 p.2 ∂γ) ^ 2 ≤ ∫ p, dist p.1 p.2 ^ 2 ∂γ :=
      (convexOn_pow 2).map_integral_le (continuous_pow 2).continuousOn isClosed_Ici
        (Filter.Eventually.of_forall fun p => (dist_nonneg : 0 ≤ dist p.1 p.2)) hd hd2
    refine hB.trans (mul_le_mul_of_nonneg_left ?_ hL)
    exact Real.le_sqrt_of_sq_le hJ
  obtain ⟨c0, hc0⟩ := transportCosts_nonempty μ ν
  rcases hL.eq_or_lt with h0 | hpos
  · simpa [← h0] using hc c0 hc0
  · have hsq : B ^ 2 / L ^ 2 ≤ sInf (transportCosts μ ν) := by
      refine le_csInf ⟨c0, hc0⟩ fun c hcm => ?_
      have hcn := transportCosts_nonneg μ ν c hcm
      rw [div_le_iff₀ (by positivity)]
      have := hc c hcm
      have hB0 : 0 ≤ B := norm_nonneg _
      calc B ^ 2 ≤ (L * Real.sqrt c) ^ 2 := pow_le_pow_left₀ hB0 this 2
        _ = c * L ^ 2 := by rw [mul_pow, Real.sq_sqrt hcn]; ring
    rw [W2, ← Real.sqrt_sq hpos.le, ← Real.sqrt_mul (by positivity)]
    refine Real.le_sqrt_of_sq_le ?_
    rw [div_le_iff₀ (by positivity)] at hsq
    linarith

/-- The hypotheses of `norm_integral_sub_le_W2` are satisfiable: a Dirac mass
at the origin lives on the unit ball, and the identity is `1`-Lipschitz. -/
example : MeasurableSet (Metric.closedBall (0 : ℝ) 1) ∧
    Bornology.IsBounded (Metric.closedBall (0 : ℝ) 1) ∧
    Measure.dirac (0 : ℝ) (Metric.closedBall (0 : ℝ) 1)ᶜ = 0 ∧
    Integrable (fun x : ℝ => x) (Measure.dirac 0) ∧
    ∀ x ∈ Metric.closedBall (0 : ℝ) 1, ∀ y ∈ Metric.closedBall (0 : ℝ) 1,
      ‖x - y‖ ≤ 1 * dist x y :=
  ⟨Metric.isClosed_closedBall.measurableSet, Metric.isBounded_closedBall,
    by simp [Measure.dirac_apply' _ Metric.isClosed_closedBall.measurableSet.compl],
    integrable_dirac (by simp), fun x _ y _ => by simp [dist_eq_norm]⟩

end Diagonal

end Wasserstein
end Transformer
