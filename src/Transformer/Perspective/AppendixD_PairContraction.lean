module

public import Transformer.Perspective.AppendixD_QuantitativeHemisphere
public import Transformer.Perspective.ConeFields
public import Transformer.Perspective.Section2_FlowMap

/-!
# Uniform pair contraction from a quantitative hemisphere

ArXiv:2312.10794v5, Appendix D, `e:ineqsecondpart`, and §6.1,
`lem: hemisphere.clustering`. The already proved chart-width contraction
gives an explicit bound on the actual pairwise inner products from any
fixed positive initial height floor. No differentiability of a minimum
over moving particles is required.

For `SA`, every weight is at least `exp(-2β)/n`. At initial height
`1/(2n)` this yields inner-product defect at most
`8n³ exp(-exp(-2β)t/n²)`. Both constants depend only on the token count
and temperature, uniformly in dimension and in the initial configuration.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Perspective

variable {d n : ℕ}

/-- Actual weighted sphere flows with a common initial height floor have
a dimension-independent exponential pair estimate. Source:
arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, applied in Appendix D. -/
theorem weighted_pair_inner_defect_exp {X : ℝ → SphereTuple d n}
    {a : ℝ → Idx n → Idx n → ℝ} (hX : IsWeightedFlow X a)
    (hn : 0 < n) {m r : ℝ} (hm : 0 < m) (ha : ∀ t i j, m ≤ a t i j)
    (w : SSphere d) (hr : 0 < r)
    (h0 : ∀ i : Idx n, r ≤ inner (𝕜 := ℝ) (X 0 i : EucSpace d) (w : EucSpace d)) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      1 - inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d) ≤
        (4 * (n : ℝ) ^ 2 / r) * Real.exp (-(2 * m * r * t)) := by
  have hfloor (t : ℝ) (ht : 0 ≤ t) (i : Idx n) :
      r ≤ inner (𝕜 := ℝ) (X t i : EucSpace d) (w : EucSpace d) :=
    floor_of_weightedFlow hX (fun t i j => hm.le.trans (ha t i j)) hn w hr h0 ht i
  have hspread := chartSpread_le_of_height X (w : EucSpace d) hr h0
  intro t ht i j
  have hwidth := norm_hemiChart_sub_le hX hm ha (norm_eq_of_mem_sphere w) hr hfloor ht i j
  calc
    _ ≤ 2 * ‖hemiChart (w : EucSpace d) (X t i : EucSpace d) -
        hemiChart (w : EucSpace d) (X t j : EucSpace d)‖ :=
      inner_defect_le_two_chart_width w _ _ (hr.trans_le (hfloor t ht i))
        (hr.trans_le (hfloor t ht j))
    _ ≤ 2 * (chartSpread w X * Real.exp (-(2 * m * r * t))) :=
      mul_le_mul_of_nonneg_left hwidth (by norm_num)
    _ ≤ 2 * ((2 * (n : ℝ) ^ 2 / r) * Real.exp (-(2 * m * r * t))) :=
      mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_right hspread (Real.exp_pos _).le) (by norm_num)
    _ = _ := by ring

/-- A genuine two-token consensus realizes all weighted contraction inputs. -/
example : IsWeightedFlow (fun _ _ => basePoint 0 : ℝ → SphereTuple 1 2) (fun _ _ _ => 1) ∧
    (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧
    (∀ (t : ℝ) (i j : Idx 2), (1 : ℝ) ≤ (fun _ _ _ => 1 : ℝ → Idx 2 → Idx 2 → ℝ) t i j) ∧
    ∀ i : Idx 2, (1 : ℝ) ≤ inner (𝕜 := ℝ)
      ((fun _ : Idx 2 => basePoint 0) i : EucSpace 1) (basePoint 0 : EucSpace 1) := by
  refine ⟨?_, by decide, one_pos, fun _ _ _ => le_rfl, ?_⟩
  · intro t i
    convert hasDerivAt_const t (basePoint 0 : EucSpace 1) using 1
    simp only [Fin.sum_univ_two, one_smul]
    rw [← two_smul ℝ (basePoint 0 : EucSpace 1), proj_smul_self (norm_eq_of_mem_sphere _)]
  · intro i; simp

/-- The original `SA` trajectory is the actual weighted flow with its
original normalized softmax coefficients. Source: arXiv:2312.10794v5,
§2, `SA`, and §6.1, the coefficients in `lem: hemisphere.clustering`. -/
theorem SA.isWeightedSphereFlow {β : ℝ} {X : ℝ → SphereTuple d n} (hX : SA d n β X) :
    IsWeightedFlow X (fun t i j => (partitionSA d n β X t i)⁻¹ *
      Real.exp (β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d))) := by
  intro t i
  refine (hX t i).congr_deriv ?_
  simp only [Finset.smul_sum, smul_smul]

example : SA 1 2 0 (fun _ _ => basePoint 0) :=
  SA_const_consensus 1 2 (by decide) 0 (basePoint 0)

/-- The original `SA` weight has the common lower bound `exp(-2β)/n`.
Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`.
Only the unit norms are needed for the bound, not the differential equation. -/
theorem sa_weight_lower_bound (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (t : ℝ) (i j : Idx n) :
    Real.exp (-(2 * β)) / (n : ℝ) ≤ (partitionSA d n β X t i)⁻¹ *
      Real.exp (β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d)) := by
  have : Nontrivial (EucSpace d) := nontrivial_of_ne (X t i : EucSpace d) 0 (by
    intro h
    have hn := norm_coe_tuple (X t) i
    rw [h, norm_zero] at hn
    norm_num at hn)
  have h := (softmaxNorm_bounds (β := β)
    (Q := ContinuousLinearMap.id ℝ (EucSpace d))
    (K := ContinuousLinearMap.id ℝ (EucSpace d)) (norm_coe_tuple (X t)) i j).1
  simpa only [ContinuousLinearMap.norm_id, abs_of_nonneg hβ, mul_one,
    softmaxNorm, attnExp, ContinuousLinearMap.id_apply, partitionSA] using h

example : (0 : ℝ) ≤ 0 := le_rfl

/-- With height `1/(2n)`, the actual `SA` pair defect has explicit constants
uniform over dimension and initial data. Source: arXiv:2312.10794v5,
Appendix D, the exponential branch of `eq: upto-t`. -/
theorem sa_pair_inner_defect_uniform {β : ℝ} (hβ : 0 ≤ β) (hn : 0 < n)
    (X : ℝ → SphereTuple d n) (hX : SA d n β X) (w : SSphere d)
    (h0 : ∀ i : Idx n, 1 / (2 * (n : ℝ)) ≤
      inner (𝕜 := ℝ) (X 0 i : EucSpace d) (w : EucSpace d)) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      1 - inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d) ≤
        (8 * (n : ℝ) ^ 3) * Real.exp (-(Real.exp (-(2 * β)) / (n : ℝ) ^ 2 * t)) := by
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have h := weighted_pair_inner_defect_exp hX.isWeightedSphereFlow hn
    (div_pos (Real.exp_pos _) hnpos) (sa_weight_lower_bound β hβ X) w
    (by positivity : 0 < 1 / (2 * (n : ℝ))) h0
  have hC : 4 * (n : ℝ) ^ 2 / (1 / (2 * (n : ℝ))) = 8 * (n : ℝ) ^ 3 := by field_simp; ring
  have hrate : 2 * (Real.exp (-(2 * β)) / (n : ℝ)) * (1 / (2 * (n : ℝ))) =
      Real.exp (-(2 * β)) / (n : ℝ) ^ 2 := by field_simp
  simpa only [hC, hrate] using h

/-- The source's positive-dimension and nonempty-token setting realizes
every uniform pair-contraction input with actual stationary attention. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 2 ∧ SA 1 2 0 (fun _ _ => basePoint 0) ∧
    ∀ i : Idx 2, 1 / (2 * (2 : ℝ)) ≤ inner (𝕜 := ℝ)
      ((fun _ : Idx 2 => basePoint 0) i : EucSpace 1) (basePoint 0 : EucSpace 1) := by
  refine ⟨le_rfl, by decide, SA_const_consensus 1 2 (by decide) 0 (basePoint 0), ?_⟩
  intro i; norm_num

/-- Two actual `SA` flows with the same quantitative height floor have
pairwise inner products within the same exponential error of one another.
Source: arXiv:2312.10794v5, Appendix D, the exponential comparison branch
in `eq: upto-t`; both flows' actual pair defects are used. -/
theorem sa_pair_inner_difference_uniform {β : ℝ} (hβ : 0 ≤ β) (hn : 0 < n)
    (X Y : ℝ → SphereTuple d n) (hX : SA d n β X) (hY : SA d n β Y)
    (wx wy : SSphere d)
    (hx : ∀ i : Idx n, 1 / (2 * (n : ℝ)) ≤
      inner (𝕜 := ℝ) (X 0 i : EucSpace d) (wx : EucSpace d))
    (hy : ∀ i : Idx n, 1 / (2 * (n : ℝ)) ≤
      inner (𝕜 := ℝ) (Y 0 i : EucSpace d) (wy : EucSpace d)) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d) -
        inner (𝕜 := ℝ) (Y t i : EucSpace d) (Y t j : EucSpace d)| ≤
          (8 * (n : ℝ) ^ 3) * Real.exp (-(Real.exp (-(2 * β)) / (n : ℝ) ^ 2 * t)) := by
  intro t ht i j
  have hxb := sa_pair_inner_defect_uniform hβ hn X hX wx hx t ht i j
  have hyb := sa_pair_inner_defect_uniform hβ hn Y hY wy hy t ht i j
  have hxu := height_le_one (norm_coe_tuple (X t) j) (X t i)
  have hyu := height_le_one (norm_coe_tuple (Y t) j) (Y t i)
  rw [abs_le]
  constructor <;> linarith

example : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 2 ∧ SA 1 2 0 (fun _ _ => basePoint 0) ∧
    1 / (2 * (2 : ℝ)) ≤ inner (𝕜 := ℝ)
      (basePoint 0 : EucSpace 1) (basePoint 0 : EucSpace 1) := by
  refine ⟨le_rfl, by decide, SA_const_consensus 1 2 (by decide) 0 (basePoint 0), ?_⟩
  norm_num

end Transformer.Perspective
