/-
# The empirical probability measure of the actual attention particles

ArXiv:2312.10794v5, §3.1, `eq: mean.field.ips`,
`eq: partition.function` and `eq: vfSd`.

The empirical measure already defined in the development is a probability
measure for a nonempty tuple. Its integrals are the actual finite averages.
Substituting those averages into the mean-field vector field cancels the
factor `1/n` and recovers the original attention field at every token.

This identifies the global particle solution with actual characteristics of
the mean-field equation. No characteristic or probability property is built
into a definition without proving it first. The nonempty-tuple hypothesis
excludes the zero measure produced by an empty empirical sum.
-/

import Transformer.Perspective.Section2_FlowRotations
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer.Perspective

variable {d n : ℕ}

/-- A nonempty empirical tuple has total mass one.
Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`;
arXiv:2411.04551v3, §1, the empirical measure before `eq: vf`. -/
theorem isProbabilityMeasure_empiricalMeasure (hn : 0 < n) (X : SphereTuple d n) :
    IsProbabilityMeasure (empiricalMeasure d n X) := by
  constructor
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  simp only [empiricalMeasure, Measure.smul_apply, Measure.finsetSum_apply,
    measure_univ, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one]
  change ENNReal.ofReal ((n : ℝ)⁻¹) * (n : ℝ≥0∞) = 1
  rw [ENNReal.ofReal_inv_of_pos hn', ENNReal.ofReal_natCast]
  exact ENNReal.inv_mul_cancel (by exact_mod_cast hn.ne') (ENNReal.natCast_ne_top n)

example : (0 : ℕ) < 1 := one_pos

/-- The actual empirical measure, regarded as a probability measure using
the proved normalization. Source: arXiv:2312.10794v5, §3.1,
`eq: mean.field.ips`. -/
noncomputable def empiricalProb (hn : 0 < n) (X : SphereTuple d n) : ProbSphere d :=
  ⟨empiricalMeasure d n X, isProbabilityMeasure_empiricalMeasure hn X⟩

/-- The construction retains exactly the source's empirical measure.
Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`. -/
theorem coe_empiricalProb (hn : 0 < n) (X : SphereTuple d n) :
    (empiricalProb hn X : Measure (SSphere d)) = empiricalMeasure d n X := rfl

example : (0 : ℕ) < 1 := one_pos

/-- The single-token empirical probability measure is the Dirac mass at
that token. Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`. -/
theorem empiricalProb_one (X : SphereTuple d 1) :
    empiricalProb one_pos X = diracProb d (X 0) := by
  apply Subtype.ext
  simp only [empiricalProb, empiricalMeasure, diracProb, Nat.cast_one,
    inv_one, Real.toNNReal_one, one_smul, Fin.sum_univ_one]

/-- Every function into a normed real vector space is integrable for a
finite empirical measure. Source: arXiv:2312.10794v5, §3.1,
the finite empirical sum in `eq: mean.field.ips`. -/
theorem integrable_empiricalMeasure {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (X : SphereTuple d n) (f : SSphere d → E) :
    Integrable f (empiricalMeasure d n X) := by
  apply Integrable.smul_measure_nnreal
  apply integrable_finsetSum_measure.mpr
  intro i _
  exact integrable_dirac enorm_lt_top

/-- Integration against the actual empirical measure is a finite average.
Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`. -/
theorem integral_empiricalMeasure {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] (X : SphereTuple d n) (f : SSphere d → E) :
    (∫ x, f x ∂empiricalMeasure d n X) = (n : ℝ)⁻¹ • ∑ i : Idx n, f (X i) := by
  rw [empiricalMeasure, integral_smul_nnreal_measure,
    integral_finsetSum_measure (fun i _ => integrable_dirac enorm_lt_top)]
  simp only [integral_dirac, NNReal.smul_def,
    Real.coe_toNNReal _ (inv_nonneg.mpr (Nat.cast_nonneg n))]

/-- The probability wrapper does not alter any empirical integral.
Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`. -/
theorem integral_empiricalProb {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] (hn : 0 < n)
    (X : SphereTuple d n) (f : SSphere d → E) :
    (∫ x, f x ∂(empiricalProb hn X : Measure (SSphere d))) =
      (n : ℝ)⁻¹ • ∑ i : Idx n, f (X i) :=
  integral_empiricalMeasure X f

example : (0 : ℕ) < 1 := one_pos

/-- The empirical partition function is the normalized finite exponential
sum. Source: arXiv:2312.10794v5, §3.1, `eq: partition.function`. -/
theorem partitionMu_empiricalProb (hn : 0 < n) (β : ℝ)
    (X : SphereTuple d n) (x : EucSpace d) :
    partitionMu d β (empiricalProb hn X) x =
      (n : ℝ)⁻¹ * ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) x (X j : EucSpace d)) := by
  unfold partitionMu
  rw [integral_empiricalProb]
  rfl

example : (0 : ℕ) < 1 := one_pos

/-- **The actual mean-field velocity at each empirical token is the original
particle attention velocity.** The normalization factor `1/n` cancels
between numerator and denominator.
Source: arXiv:2312.10794v5, §3.1, `eq: mean.field.ips` and `eq: vfSd`. -/
theorem vectorField_empiricalProb_eq_saField (hn : 0 < n) (β : ℝ)
    (X : SphereTuple d n) (i : Idx n) :
    vectorField d β (empiricalProb hn X) (X i : EucSpace d) =
      saField d n β (fun j => (X j : EucSpace d)) i := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold vectorField partitionMu
  rw [integral_empiricalProb, integral_empiricalProb]
  simp only [smul_eq_mul]
  rw [smul_smul, mul_inv_rev, mul_assoc, inv_mul_cancel₀ (inv_ne_zero hn'), mul_one]
  unfold saField
  congr 1
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rw [smul_smul, softmaxWeight, saScore, inv_mul_eq_div]
  rfl

example : (0 : ℕ) < 1 := one_pos

/-- Actual attention trajectories are characteristics of their empirical
mean-field solution. Source: arXiv:2312.10794v5, §3.1,
`eq: mean.field.ips`, with the original `SA` of §2. -/
theorem meanFieldIPS_of_SA (hn : 0 < n) (β : ℝ)
    (X : ℝ → SphereTuple d n) (hX : SA d n β X) :
    meanFieldIPS d n β (fun t => empiricalProb hn (X t)) X := by
  intro t i
  rw [vectorField_empiricalProb_eq_saField]
  exact (SA_iff d n β X).mp hX t i

/-- Both the nonempty-tuple and actual ODE hypotheses hold for a nonzero
stationary token. Source: arXiv:2312.10794v5, §2, `SA`. -/
example (β : ℝ) (x : SSphere 1) :
    (0 : ℕ) < 1 ∧ SA 1 1 β (fun _ _ => x) :=
  ⟨one_pos, SA_const_consensus 1 1 one_pos β x⟩

/-- The proved global attention flow follows its actual empirical
mean-field velocity. Source: arXiv:2312.10794v5, §3.1,
`eq: mean.field.ips`; the initial measure is the given empirical tuple. -/
theorem meanFieldIPS_saFlow (hn : 0 < n) (β : ℝ) (X₀ : SphereTuple d n) :
    meanFieldIPS d n β (fun t => empiricalProb hn (saFlow β X₀ t)) (saFlow β X₀) :=
  meanFieldIPS_of_SA hn β _ (saFlow_spec β X₀).2

example : (0 : ℕ) < 1 := one_pos

end Transformer.Perspective
