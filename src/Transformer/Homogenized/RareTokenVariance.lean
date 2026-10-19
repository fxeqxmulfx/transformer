module

public import Transformer.Homogenized.RareTokenDiffusion

/-!
# Homogenized Transformers — the actual variance of a rare-token change

The squared norm of the diffusion change is integrable under the admissible
head law. Its integral is the second moment of the exact softmax gain under
the two independent standard Gaussians. On their rectangle `[u,u+1]²` the
gain is at least one, giving a lower bound by that rectangle's actual
probability. All integral comparisons retain the coefficient of the paper.

Source: arXiv:2604.01978v1, `eq:G_def`, `prop:satisfying_MF`,
`eq:Lipschitz_measure`; these are auxiliary estimates for its counterexample.
-/

@[expose] public section

open scoped ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The exact gain is measurable in the Gaussian pair at every token mass.
Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem measurable_rareGain_product (ε : ℝ) :
    Measurable (fun p : ℝ × ℝ => rareGain (Real.exp (6 * (p.1 * p.2))) ε) := by
  unfold rareGain
  fun_prop

/-- The squared gain is an integrable scalar random variable, bounded by four.
Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary to `prop:satisfying_MF`. -/
theorem integrable_rareGain_product_sq {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Integrable (fun p : ℝ × ℝ => rareGain (Real.exp (6 * (p.1 * p.2))) ε ^ 2)
      gaussianProductLaw := by
  apply Integrable.of_bound ((measurable_rareGain_product ε).pow_const 2).aestronglyMeasurable 4
  filter_upwards [] with p
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  have h := rareGain_mem (Real.exp_pos (6 * (p.1 * p.2))) hε
  have hp := pow_le_pow_left₀ h.1 h.2 2
  norm_num at hp
  exact hp

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The squared norm of the actual coefficient change is measurable before
mapping the source to heads. Source: arXiv:2604.01978v1, `eq:G_def`. -/
theorem measurable_norm_GfieldOf_rare_sub_sq {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Measurable (fun θ : HeadParam 2 =>
      ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2) := by
  have hzero : rareTokenLaw 0 = Measure.dirac (-rareAtom) := by simp [rareTokenLaw]
  have hbase : Measurable (fun θ : HeadParam 2 =>
      GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ) := by
    rw [← hzero]
    exact measurable_GfieldOf_rare (show (0 : ℝ) ∈ Set.Icc 0 1 by norm_num)
  exact ((measurable_GfieldOf_rare hε).sub hbase).norm.pow_const 2

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The actual squared diffusion change is integrable, so its variance is a
genuine second moment. Source: arXiv:2604.01978v1, `eq:G_def`. -/
theorem integrable_norm_GfieldOf_rare_sub_sq {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Integrable (fun θ : HeadParam 2 =>
      ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2) productHeadLaw := by
  have hm := measurable_norm_GfieldOf_rare_sub_sq hε
  rw [productHeadLaw]
  apply (integrable_map_measure hm.aestronglyMeasurable measurable_productHead.aemeasurable).2
  apply Integrable.of_bound (hm.comp measurable_productHead).aestronglyMeasurable 4
  filter_upwards [] with ω
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  have hp := pow_le_pow_left₀ (norm_nonneg _)
    (norm_GfieldOf_productHead_rare_sub_le hε ω.1 ω.2) 2
  norm_num at hp
  exact hp

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The actual variance equals the exact gain's Gaussian second moment. The
independent sign disappears only after taking the squared norm; Fubini is
applied to a bounded, measurable integrand.
Source: arXiv:2604.01978v1, `eq:G_def`, `eq:decomposition_G_mf`. -/
theorem integral_norm_GfieldOf_rare_sub_sq {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    (∫ θ, ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw) =
      ∫ p : ℝ × ℝ, rareGain (Real.exp (6 * (p.1 * p.2))) ε ^ 2 ∂gaussianProductLaw := by
  have hm := measurable_norm_GfieldOf_rare_sub_sq hε
  have hs : Integrable (fun ω : Bool × (ℝ × ℝ) =>
      rareGain (Real.exp (6 * (ω.2.1 * ω.2.2))) ε ^ 2)
      (fairCoin.prod gaussianProductLaw) := by
    apply Integrable.of_bound
      (((measurable_rareGain_product ε).pow_const 2).comp measurable_snd).aestronglyMeasurable 4
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have h := rareGain_mem (Real.exp_pos (6 * (ω.2.1 * ω.2.2))) hε
    have hp := pow_le_pow_left₀ h.1 h.2 2
    norm_num at hp
    exact hp
  change (∫ θ, ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery θ -
      GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2
      ∂Measure.map productHead (fairCoin.prod gaussianProductLaw)) = _
  rw [integral_map measurable_productHead.aemeasurable hm.aestronglyMeasurable]
  simp only [norm_GfieldOf_productHead_rare_sub hε]
  rw [integral_prod _ hs]
  simp only [integral_const, probReal_univ, one_smul]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The probability of the actual Gaussian rectangle bounds the gain's second
moment below. Source: arXiv:2604.01978v1, `eq:Lipschitz_measure`, auxiliary
to the counterexample to `prop:satisfying_MF`. -/
theorem gaussian_rectangle_le_integral_rareGain_sq {u : ℝ} (hu : 1 ≤ u) :
    gaussianProductLaw.real (Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) ≤
      ∫ p : ℝ × ℝ, rareGain (Real.exp (6 * (p.1 * p.2))) (rareMass u) ^ 2
        ∂gaussianProductLaw := by
  let S : Set (ℝ × ℝ) := Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)
  have hs : MeasurableSet S := measurableSet_Icc.prod measurableSet_Icc
  have hnon : 0 ≤ᵐ[gaussianProductLaw] S.indicator (fun _ => (1 : ℝ)) := by
    filter_upwards [] with p
    by_cases hp : p ∈ S <;> simp [hp]
  have hbound : S.indicator (fun _ => (1 : ℝ)) ≤ᵐ[gaussianProductLaw]
      (fun p => rareGain (Real.exp (6 * (p.1 * p.2))) (rareMass u) ^ 2) := by
    filter_upwards [] with p
    by_cases hp : p ∈ S
    · rw [Set.indicator_of_mem hp]
      have h := one_le_rareGain_rectangle hu hp
      nlinarith
    · rw [Set.indicator_of_notMem hp]
      exact sq_nonneg _
  have h := integral_mono_of_nonneg hnon
    (integrable_rareGain_product_sq ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩) hbound
  rw [integral_indicator hs, setIntegral_const] at h
  simpa only [smul_eq_mul, mul_one] using h

example : (1 : ℝ) ≤ 1 := le_rfl

/-- The lower bound retains the paper's actual diffusion coefficient and its
actual head law. Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary to the
counterexample to `prop:satisfying_MF`. -/
theorem gaussian_rectangle_le_integral_GfieldOf_rare_sub_sq {u : ℝ} (hu : 1 ≤ u) :
    gaussianProductLaw.real (Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) ≤
      ∫ θ, ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw := by
  rw [integral_norm_GfieldOf_rare_sub_sq ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩]
  exact gaussian_rectangle_le_integral_rareGain_sq hu

example : (1 : ℝ) ≤ 1 := le_rfl

/-- The actual diffusion variance has a Gaussian lower bound of order
`exp(-4u²)`, while the transported mass is `exp(-6u²)`.
Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary to the counterexample
to `prop:satisfying_MF`. -/
theorem integral_GfieldOf_rare_sub_sq_lower {u : ℝ} (hu : 1 ≤ u) :
    gaussianUnitConstant ^ 2 * Real.exp (-4 * u ^ 2) ≤
      ∫ θ, ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery θ -
        GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery θ‖ ^ 2 ∂productHeadLaw := by
  exact (gaussian_product_rectangle_lower_quadratic hu).trans
    (gaussian_rectangle_le_integral_GfieldOf_rare_sub_sq hu)

example : (1 : ℝ) ≤ 1 := le_rfl

end Transformer.Homogenized
