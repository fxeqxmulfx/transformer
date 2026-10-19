module

public import Transformer.Homogenized.RareTokenMeasure
public import Transformer.Homogenized.MeanField

/-!
# Homogenized Transformers — exact softmax amplification of a rare token

For the admissible product head and the two-atom token law, at query `e₂`
and temperature `3`, the attention field is
`sign(V₀₀) (gain(exp(6GH),ε)-1) e₁`, where
`gain(R,ε) = 2εR/(1-ε+εR)`.
The coefficient is bounded by one in norm, but its change from `δ_{-e₁}`
is at least one whenever `εR ≥ 1-ε`.

These exact identities retain the normalization and value map of the paper.
Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`,
auxiliary to `prop:satisfying_MF`, `eq:Lipschitz_measure`.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The change in the two-atom barycenter when its likelihood ratio is `R`. -/
noncomputable def rareGain (R ε : ℝ) : ℝ := 2 * ε * R / (1 - ε + ε * R)

/-- The true softmax normalizer stays positive at every positive likelihood
ratio. Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem rareGain_den_pos {R ε : ℝ} (hR : 0 < R) (hε : ε ∈ Set.Icc 0 1) :
    0 < 1 - ε + ε * R := by
  rcases hε.1.eq_or_lt with h0 | hpos
  · rw [← h0]
    norm_num
  · have hprod := mul_pos hpos hR
    linarith [hε.2]

example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The amplification always lies between zero and two. -/
theorem rareGain_mem {R ε : ℝ} (hR : 0 < R) (hε : ε ∈ Set.Icc 0 1) :
    rareGain R ε ∈ Set.Icc 0 2 := by
  have hd := rareGain_den_pos hR hε
  refine ⟨div_nonneg (mul_nonneg (mul_nonneg (by norm_num) hε.1) hR.le) hd.le, ?_⟩
  rw [rareGain, div_le_iff₀ hd]
  nlinarith [hε.2]

example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The original barycenter coefficient is bounded by one in absolute value. -/
theorem abs_rareGain_sub_one_le {R ε : ℝ} (hR : 0 < R) (hε : ε ∈ Set.Icc 0 1) :
    |rareGain R ε - 1| ≤ 1 := by
  rw [abs_le]
  have h := rareGain_mem hR hε
  constructor <;> linarith [h.1, h.2]

example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- A rare atom with weighted mass at least the majority atom changes the
barycenter by at least one. Source: arXiv:2604.01978v1,
auxiliary to `eq:Lipschitz_measure`. -/
theorem one_le_rareGain {R ε : ℝ} (hR : 0 < R) (hε : ε ∈ Set.Icc 0 1)
    (hamp : 1 - ε ≤ ε * R) : 1 ≤ rareGain R ε := by
  rw [rareGain, le_div_iff₀ (rareGain_den_pos hR hε)]
  linarith

example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc 0 1 ∧
    1 - (1 / 2 : ℝ) ≤ (1 / 2 : ℝ) * 1 := by norm_num

/-- Factoring the common exponential from the actual two softmax weights
gives exactly the gain coefficient. -/
theorem rareGain_ratio {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) (r : ℝ) :
    ((1 - ε) * Real.exp (-3 * r) + ε * Real.exp (3 * r))⁻¹ *
        (ε * Real.exp (3 * r) - (1 - ε) * Real.exp (-3 * r)) =
      rareGain (Real.exp (6 * r)) ε - 1 := by
  have hd := (rareGain_den_pos (Real.exp_pos (6 * r)) hε).ne'
  have he : Real.exp (3 * r) = Real.exp (-3 * r) * Real.exp (6 * r) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hz : (1 - ε) * Real.exp (-3 * r) + ε * Real.exp (3 * r) =
      Real.exp (-3 * r) * (1 - ε + ε * Real.exp (6 * r)) := by rw [he]; ring
  rw [hz, he, rareGain]
  calc _ = ((Real.exp (-3 * r))⁻¹ * Real.exp (-3 * r)) *
      ((1 - ε + ε * Real.exp (6 * r))⁻¹ *
        (2 * ε * Real.exp (6 * r) - (1 - ε + ε * Real.exp (6 * r)))) := by
        rw [mul_inv]
        ring
    _ = (1 - ε + ε * Real.exp (6 * r))⁻¹ *
        (2 * ε * Real.exp (6 * r) - (1 - ε + ε * Real.exp (6 * r))) := by
        rw [inv_mul_cancel₀ (Real.exp_ne_zero _), one_mul]
    _ = _ := by
        rw [mul_sub, inv_mul_cancel₀ hd, div_eq_mul_inv]
        ring

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The sparse attention matrix gives the actual exponential score at each
scalar token, for every temperature. -/
theorem attnWeight_productHead_rare (β : ℝ) (b : Bool) (p : ℝ × ℝ) (c : ℝ) :
    attnWeight β (productHead (b, p)) rareQuery (c • rareAtom) =
      Real.exp (β * (p.1 * p.2 * c)) := by
  change Real.exp (β * inner (𝕜 := ℝ)
    (Matrix.toEuclideanLin (productAttentionMatrix (p.1 * p.2)) rareQuery) (c • rareAtom)) = _
  rw [productAttentionMatrix_apply]
  simp [rareQuery, rareAtom, real_inner_smul_left, real_inner_smul_right, mul_comm]

/-- The value entry reads the actual scalar token along the first axis. -/
theorem valueMap_productHead_rare (b : Bool) (p : ℝ × ℝ) (c : ℝ) :
    valueMap (productHead (b, p)) (c • rareAtom) =
      ((if b then (1 : ℝ) else -1) * c) • rareAtom := by
  cases b <;> ext i <;> fin_cases i <;>
    simp [valueMap, productHead, rareAtom, Matrix.toLpLin_apply, radMatrix_apply]

/-- The exact attention field of the paper at the rare-token probability
law. Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem attnFieldOf_productHead_rare {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    (b : Bool) (p : ℝ × ℝ) :
    attnFieldOf 3 (productHead (b, p)) (rareTokenLaw ε) rareQuery =
      ((if b then (1 : ℝ) else -1) * (rareGain (Real.exp (6 * (p.1 * p.2))) ε - 1)) •
        rareAtom := by
  have hwplus := attnWeight_productHead_rare 3 b p 1
  have hwminus := attnWeight_productHead_rare 3 b p (-1)
  have hvplus := valueMap_productHead_rare b p 1
  have hvminus := valueMap_productHead_rare b p (-1)
  simp only [one_smul, neg_one_smul, mul_one, mul_neg_one] at hwplus hwminus hvplus hvminus
  rw [attnFieldOf, integral_rareTokenLaw hε, integral_rareTokenLaw hε,
    hwplus, hwminus, hvplus, hvminus]
  simp only [smul_smul, smul_eq_mul]
  rw [← add_smul, smul_smul]
  congr 1
  have hratio := rareGain_ratio hε (p.1 * p.2)
  have hneg : 3 * -(p.1 * p.2) = -3 * (p.1 * p.2) := by ring
  rw [hneg]
  calc _ = (if b then (1 : ℝ) else -1) *
      (((1 - ε) * Real.exp (-3 * (p.1 * p.2)) + ε * Real.exp (3 * (p.1 * p.2)))⁻¹ *
        (ε * Real.exp (3 * (p.1 * p.2)) - (1 - ε) * Real.exp (-3 * (p.1 * p.2)))) := by ring
    _ = _ := by rw [hratio]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- On the Gaussian rectangle the exponentially rare atom is amplified to
a change of at least one. Source: arXiv:2604.01978v1,
auxiliary to `eq:Lipschitz_measure`. -/
theorem one_le_rareGain_rectangle {u : ℝ} (hu : 1 ≤ u) {p : ℝ × ℝ}
    (hp : p ∈ Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) :
    1 ≤ rareGain (Real.exp (6 * (p.1 * p.2))) (rareMass u) := by
  apply one_le_rareGain (Real.exp_pos _) ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩
  have hprod := gaussian_product_rectangle_subset (zero_le_one.trans hu) hp
  change u ^ 2 ≤ p.1 * p.2 at hprod
  have hlarge : 1 ≤ rareMass u * Real.exp (6 * (p.1 * p.2)) := by
    rw [rareMass, ← Real.exp_add]
    apply Real.one_le_exp
    linarith
  linarith [(rareMass_mem u).1]

example : (1 : ℝ) ≤ 1 ∧ ((1, 1) : ℝ × ℝ) ∈
    Set.Icc 1 (1 + 1) ×ˢ Set.Icc 1 (1 + 1) := by norm_num

end Transformer.Homogenized
