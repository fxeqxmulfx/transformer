/-
# Homogenized Transformers — a counterexample to global mean-field Lipschitzness

`prop:satisfying_MF` of arXiv:2604.01978v1 is false even after its displayed
rate is discarded and the Lipschitz constant may depend on the fixed model.
The witness has dimension two, temperature three, centered value matrix
`V₀₀ = ±1`, and centered attention matrix `A₀₁ = GH`, where the sign and the
two standard Gaussians are independent. All other matrix entries are zero.
`isHighOrderLaw_productHead` proves every clause of `ass:high_order_short`.

At query `e₂`, compare `δ_{-e₁}` with
`(1-ε)δ_{-e₁} + εδ_{e₁}`, for `ε = exp(-6u²)`.
Their squared transport distance is at most `4ε`. On `[u,u+1]²` the actual
diffusion change has norm at least one. That rectangle has probability at
least `(2π)⁻¹ exp(-4u²)`, so no finite global Lipschitz constant works.
The mean field, projection, normalization, head law and integrability are
all established for the actual coefficients in the preceding modules.

For clarity, the query/key realization has only `W₀₀ = G` and `W'₁₀ = H`
nonzero, so `W W'ᵀ` has exactly the displayed entry. Constants in the other
entries are independent subGaussian random variables with the same allowed
proxy. The value matrix is independent of both query/key matrices.

The query `e₂` is perpendicular to both token locations. The tangent
projection therefore preserves the first-axis change. Averaging the value
coin cancels the mean field before taking the difference; the remaining
coefficient has norm exactly
`2ε exp(6GH)/(1-ε+ε exp(6GH))`.
It is bounded by two, so the failure is a failure of uniform Lipschitz
continuity in the measure, despite finite second moments at every measure
in the counterexample. Neither query leaves the unit sphere.

Source: arXiv:2604.01978v1, `prop:satisfying_MF`, `eq:Lipschitz_measure`.
-/

import Transformer.Homogenized.RareTokenVariance

open scoped NNReal ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The actual variance is strictly positive at every threshold in the
counterexample family. Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary
to the counterexample to `prop:satisfying_MF`. -/
theorem integral_GfieldOf_rare_sub_sq_pos {u : ℝ} (hu : 1 ≤ u) :
    0 < ∫ θ,
      ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw := by
  exact (mul_pos (sq_pos_of_pos gaussianUnitConstant_pos) (Real.exp_pos _)).trans_le
    (integral_GfieldOf_rare_sub_sq_lower hu)

example : (1 : ℝ) ≤ 1 := le_rfl

/-- No constant bounds the actual diffusion variance by the rare mass
uniformly over these genuine probability laws. The variance is integrable
by `integrable_norm_GfieldOf_rare_sub_sq`.
Source: arXiv:2604.01978v1, counterexample to `prop:satisfying_MF`. -/
theorem integral_GfieldOf_rare_sub_sq_not_upper (C : ℝ) :
    ∃ u : ℝ, 1 ≤ u ∧ C * rareMass u <
      ∫ θ, ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw := by
  obtain ⟨u, hu, h⟩ := gaussian_product_rectangle_not_upper C
  refine ⟨u, hu, h.trans_le (gaussian_rectangle_le_integral_GfieldOf_rare_sub_sq hu)⟩

/-- Every candidate constant fails on a two-atom law, with both atoms having
positive mass. The query is the same on both sides, so this already refutes
the measure part of the claimed joint Lipschitz bound.

Squaring a supposed bound would give variance at most `4L² exp(-6u²)`,
contradicting the actual Gaussian lower bound. The argument also rules out
zero or negative candidate constants.
Source: arXiv:2604.01978v1, counterexample to `prop:satisfying_MF`. -/
theorem rareToken_lipschitz_ratio_unbounded (L : ℝ) :
    ∃ u : ℝ, 1 ≤ u ∧
      L * Wasserstein.W2 (rareTokenLaw (rareMass u)) (Measure.dirac (-rareAtom)) <
        Real.sqrt (∫ θ,
          ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
            GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw) := by
  obtain ⟨u, hu, hbad⟩ := integral_GfieldOf_rare_sub_sq_not_upper (4 * L ^ 2)
  refine ⟨u, hu, ?_⟩
  by_contra h
  have hle : Real.sqrt (∫ θ,
      ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw) ≤
      L * Wasserstein.W2 (rareTokenLaw (rareMass u)) (Measure.dirac (-rareAtom)) := le_of_not_gt h
  have hsq := pow_le_pow_left₀ (Real.sqrt_nonneg _) hle 2
  rw [Real.sq_sqrt (integral_nonneg (fun _ => sq_nonneg _)), mul_pow] at hsq
  have hW := W2_rareTokenLaw_sq_le ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩
  have hupper : (∫ θ,
      ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw) ≤
      4 * L ^ 2 * rareMass u := by
    calc _ ≤ L ^ 2 * Wasserstein.W2 (rareTokenLaw (rareMass u)) (Measure.dirac (-rareAtom)) ^ 2 := hsq
      _ ≤ L ^ 2 * (4 * rareMass u) := mul_le_mul_of_nonneg_left hW (sq_nonneg L)
      _ = _ := by ring
  exact (not_le_of_gt hbad) hupper

/-- **The qualitative assertion of `prop:satisfying_MF` is false.** At the
single fixed admissible model `productHeadLaw`, no finite positive constant
works for every sphere-supported probability measure and every unit query.
This is stronger than refuting the displayed rate: the temperature is fixed
at three and the constant may depend on all model data.

The hypotheses are realized by `hasHighOrderLaw_productHead`; every token
measure used below is a probability law on the actual unit circle. The
coefficient is the centered, projected attention field `eq:G_def`.
Source: arXiv:2604.01978v1, `prop:satisfying_MF`. -/
theorem not_satisfying_MF_productHead :
    ¬ ∃ L : ℝ, 0 < L ∧
      ∀ (μ ν : Measure (EucSpace 2)), IsProbabilityMeasure μ → IsProbabilityMeasure ν →
        μ {z : EucSpace 2 | ‖z‖ = 1}ᶜ = 0 → ν {z : EucSpace 2 | ‖z‖ = 1}ᶜ = 0 →
      ∀ x y : EucSpace 2, ‖x‖ = 1 → ‖y‖ = 1 →
        Real.sqrt (∫ θ, ‖GfieldOf 3 productHeadLaw μ x θ - GfieldOf 3 productHeadLaw ν y θ‖ ^ 2
          ∂productHeadLaw) ≤ L * (Wasserstein.W2 μ ν + ‖x - y‖) := by
  rintro ⟨L, h⟩
  obtain ⟨u, hu, hbad⟩ := rareToken_lipschitz_ratio_unbounded L
  have hε : rareMass u ∈ Set.Icc 0 1 := ⟨(rareMass_mem u).1.le, (rareMass_lt_one hu).le⟩
  let := isProbabilityMeasure_rareTokenLaw hε
  have hν : (Measure.dirac (-rareAtom)) {z : EucSpace 2 | ‖z‖ = 1}ᶜ = 0 := by
    have hs := rareTokenLaw_sphere 0
    simpa only [rareTokenLaw, sub_zero, ENNReal.ofReal_one, ENNReal.ofReal_zero,
      one_smul, zero_smul, add_zero] using hs
  have hLip := h.2 (rareTokenLaw (rareMass u)) (Measure.dirac (-rareAtom))
    inferInstance inferInstance (rareTokenLaw_sphere _) hν
    rareQuery rareQuery norm_rareQuery norm_rareQuery
  simp only [sub_self, norm_zero, add_zero] at hLip
  exact (not_le_of_gt hbad) hLip

/-- This fixed model satisfies the source assumption with common variance
proxies one and with both matrix means zero. -/
example : HasHighOrderLaw 2 1 1 productHeadLaw := hasHighOrderLaw_productHead

/-- **Counterexample to `prop:satisfying_MF` with its original quantifiers.**
The assumption of independent subGaussian query/key entries allows their
product to have an exponential tail; it does not supply the exponential
moments at arbitrary temperature that the proof of `eq:Lipschitz_measure`
integrates. The counterexample above rules out the claimed conclusion
itself, without altering the source coefficient or imposing a new hypothesis.

The former unproved positive statement `satisfying_MF` is therefore replaced
by this proved negation. Source: arXiv:2604.01978v1, `prop:satisfying_MF`. -/
theorem not_forall_satisfying_MF :
    ¬ ∀ (d : ℕ) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ →
      ∃ L : ℝ, 0 < L ∧
        ∀ (μ ν : Measure (EucSpace d)), IsProbabilityMeasure μ → IsProbabilityMeasure ν →
          μ {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 → ν {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 →
        ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
          Real.sqrt (∫ θ, ‖GfieldOf β ρ μ x θ - GfieldOf β ρ ν y θ‖ ^ 2 ∂ρ) ≤
            L * (Wasserstein.W2 μ ν + ‖x - y‖) := by
  intro h
  exact not_satisfying_MF_productHead (h 2 3 1 1 productHeadLaw hasHighOrderLaw_productHead)

end Transformer.Homogenized
