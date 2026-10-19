module

public import Transformer.Homogenized.RareTokenAttention
public import Transformer.Homogenized.AttnAverage
public import Mathlib.MeasureTheory.Integral.Prod

/-!
# Homogenized Transformers — the actual diffusion coefficient at a rare atom

The value coin is independent of the Gaussian query/key entry and centered.
At the rare-token laws this gives a zero mean attention field, with genuine
integrability before applying Fubini. The tangent projection at `e₂` leaves
the first-axis field unchanged. Consequently the change in the actual
diffusion coefficient has norm exactly the softmax gain.

Source: arXiv:2604.01978v1, `eq:G_def`, `eq:decomposition_G_mf`,
auxiliary to `prop:satisfying_MF`.
-/

@[expose] public section

open scoped ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The actual two-atom attention field is measurable in every head, including
heads outside the support of the chosen law. Source: arXiv:2604.01978v1,
`EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem measurable_attnFieldOf_rare {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Measurable (fun θ : HeadParam 2 => attnFieldOf 3 θ (rareTokenLaw ε) rareQuery) := by
  have hw (z : EucSpace 2) : Measurable (fun θ : HeadParam 2 => attnWeight 3 θ rareQuery z) := by
    change Measurable (fun θ : HeadParam 2 => attnWeight 3 (0, θ.2) rareQuery z)
    exact (continuous_attnWeight_qk 3 rareQuery z).measurable.comp measurable_snd
  have hv (z : EucSpace 2) : Measurable (fun θ : HeadParam 2 => valueMap θ z) := by
    change Measurable (fun θ : HeadParam 2 => Matrix.toEuclideanLin θ.1 z)
    have hf : Measurable (fun θ : HeadParam 2 =>
        ((θ.1, z) : Matrix (Fin 2) (Fin 2) ℝ × EucSpace 2)) :=
      measurable_fst.prodMk measurable_const
    have hm := (continuous_toEuclideanLin_prod (d := 2)).measurable.comp hf
    exact hm
  have hwm := hw (-rareAtom)
  have hwp := hw rareAtom
  have hvm := hv (-rareAtom)
  have hvp := hv rareAtom
  have hden : Measurable (fun θ : HeadParam 2 => (1 - ε) * attnWeight 3 θ rareQuery (-rareAtom) +
      ε * attnWeight 3 θ rareQuery rareAtom) :=
    (measurable_const.mul hwm).add (measurable_const.mul hwp)
  have hnum : Measurable (fun θ : HeadParam 2 =>
      (1 - ε) • (attnWeight 3 θ rareQuery (-rareAtom) • valueMap θ (-rareAtom)) +
        ε • (attnWeight 3 θ rareQuery rareAtom • valueMap θ rareAtom)) :=
    ((measurable_const (a := 1 - ε)).smul (hwm.smul hvm)).add
      ((measurable_const (a := ε)).smul (hwp.smul hvp))
  simp only [attnFieldOf, integral_rareTokenLaw hε, smul_eq_mul]
  exact hden.inv.smul hnum

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The exact source attention field has norm at most one, ensuring that its
mean is an integrable average. Source: arXiv:2604.01978v1,
`EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem norm_attnFieldOf_productHead_rare_le {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    (b : Bool) (p : ℝ × ℝ) :
    ‖attnFieldOf 3 (productHead (b, p)) (rareTokenLaw ε) rareQuery‖ ≤ 1 := by
  rw [attnFieldOf_productHead_rare hε]
  rw [norm_smul, norm_rareAtom, mul_one, Real.norm_eq_abs, abs_mul]
  cases b <;> simpa using
    abs_rareGain_sub_one_le (Real.exp_pos (6 * (p.1 * p.2))) hε

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The genuine mean field vanishes: first average over the independent
value coin, then over the Gaussian pair. Fubini is used only after the
bounded, measurable source field has been proved integrable.
Source: arXiv:2604.01978v1, `eq:decomposition_G_mf`. -/
theorem meanFieldOf_productHead_rare {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    meanFieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery = 0 := by
  let F (ω : Bool × (ℝ × ℝ)) : EucSpace 2 :=
    attnFieldOf 3 (productHead ω) (rareTokenLaw ε) rareQuery
  have hmeas : Measurable F := (measurable_attnFieldOf_rare hε).comp measurable_productHead
  have hint : Integrable F (fairCoin.prod gaussianProductLaw) :=
    Integrable.of_bound hmeas.aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun ω => norm_attnFieldOf_productHead_rare_le hε ω.1 ω.2)
  rw [meanFieldOf, productHeadLaw,
    integral_map measurable_productHead.aemeasurable
      (measurable_attnFieldOf_rare hε).aestronglyMeasurable,
    integral_prod_symm _ hint]
  have hcoin (p : ℝ × ℝ) : ∫ b, F (b, p) ∂fairCoin = 0 := by
    rw [integral_fairCoin]
    dsimp only [F]
    rw [attnFieldOf_productHead_rare hε, attnFieldOf_productHead_rare hε]
    simp
    module
  simp only [hcoin, integral_zero]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The actual tangent projection at the query preserves every first-axis
field. Source: arXiv:2604.01978v1, `eq:G_def`. -/
theorem proj_rareQuery_smul_rareAtom (c : ℝ) : proj 2 rareQuery (c • rareAtom) = c • rareAtom := by
  simp [proj, rareQuery, rareAtom, EuclideanSpace.inner_single_left]

/-- The actual diffusion coefficient equals the signed barycenter coefficient;
no mean or projection term is discarded. Source: arXiv:2604.01978v1,
`eq:G_def`, `eq:decomposition_G_mf`. -/
theorem GfieldOf_productHead_rare {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    (b : Bool) (p : ℝ × ℝ) :
    GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery (productHead (b, p)) =
      ((if b then (1 : ℝ) else -1) * (rareGain (Real.exp (6 * (p.1 * p.2))) ε - 1)) •
        rareAtom := by
  rw [GfieldOf, fluctOf, meanFieldOf_productHead_rare hε, sub_zero,
    attnFieldOf_productHead_rare hε, proj_rareQuery_smul_rareAtom]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The gain is exactly the norm of the change from the one-atom coefficient.
Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary to `prop:satisfying_MF`. -/
theorem norm_GfieldOf_productHead_rare_sub {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    (b : Bool) (p : ℝ × ℝ) :
    ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery (productHead (b, p)) -
      GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery (productHead (b, p))‖ =
        rareGain (Real.exp (6 * (p.1 * p.2))) ε := by
  have hzero : rareTokenLaw 0 = Measure.dirac (-rareAtom) := by simp [rareTokenLaw]
  have hbase := GfieldOf_productHead_rare (show (0 : ℝ) ∈ Set.Icc 0 1 by norm_num) b p
  simp only [rareGain, mul_zero, zero_mul, zero_div, zero_sub, mul_neg_one] at hbase
  rw [← hzero, GfieldOf_productHead_rare hε, hbase]
  have he : (if b then (1 : ℝ) else -1) *
      (rareGain (Real.exp (6 * (p.1 * p.2))) ε - 1) - -(if b then (1 : ℝ) else -1) =
      (if b then (1 : ℝ) else -1) * rareGain (Real.exp (6 * (p.1 * p.2))) ε := by ring
  rw [← sub_smul, he, norm_smul, norm_rareAtom, mul_one, Real.norm_eq_abs, abs_mul]
  cases b <;> simp [abs_of_nonneg (rareGain_mem (Real.exp_pos _) hε).1]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The genuine coefficient difference is uniformly bounded by two.
Source: arXiv:2604.01978v1, `eq:G_def`, auxiliary to `prop:satisfying_MF`. -/
theorem norm_GfieldOf_productHead_rare_sub_le {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    (b : Bool) (p : ℝ × ℝ) :
    ‖GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery (productHead (b, p)) -
      GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery (productHead (b, p))‖ ≤ 2 := by
  rw [norm_GfieldOf_productHead_rare_sub hε]
  exact (rareGain_mem (Real.exp_pos _) hε).2

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- On the Gaussian rectangle the actual coefficient changes by at least
one despite the exponentially small token mass. Source:
arXiv:2604.01978v1, `eq:G_def`, auxiliary to `prop:satisfying_MF`. -/
theorem one_le_norm_GfieldOf_rare_rectangle {u : ℝ} (hu : 1 ≤ u)
    (b : Bool) {p : ℝ × ℝ} (hp : p ∈ Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) :
    1 ≤ ‖GfieldOf 3 productHeadLaw (rareTokenLaw (rareMass u)) rareQuery (productHead (b, p)) -
      GfieldOf 3 productHeadLaw (Measure.dirac (-rareAtom)) rareQuery (productHead (b, p))‖ := by
  rw [norm_GfieldOf_productHead_rare_sub ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩]
  exact one_le_rareGain_rectangle hu hp

example : (1 : ℝ) ≤ 1 ∧ ((1, 1) : ℝ × ℝ) ∈
    Set.Icc 1 (1 + 1) ×ˢ Set.Icc 1 (1 + 1) := by norm_num

/-- The actual coefficient, before composition with the head source, is
measurable. Source: arXiv:2604.01978v1, `eq:G_def`. -/
theorem measurable_GfieldOf_rare {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Measurable (fun θ : HeadParam 2 => GfieldOf 3 productHeadLaw (rareTokenLaw ε) rareQuery θ) := by
  have h := measurable_attnFieldOf_rare hε
  unfold GfieldOf fluctOf proj
  fun_prop

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

end Transformer.Homogenized
