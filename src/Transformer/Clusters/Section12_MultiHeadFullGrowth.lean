module

public import Transformer.Clusters.Section12_MultiHeadExtremeRates
public import Transformer.Clusters.Section12_DiscreteGeometricError

/-!
# The full original layer-factor growth of escaping multi-head tokens

ArXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`,
with §3's discrete convention and the original head sum of §12.
Write q=1+Δt Σv_h and b=Δt n Σ(v_h/c_h). The proved original
self-pull recurrence is x_next≥qx-b/x for positive x.
Once the actual escaping coordinate is sufficiently large, squaring
this inequality gives x_next²≥q²x²-2qb. The exact affine recursion
estimate retains the full q^k factor with a positive prefactor.

The needed sufficiently late layer is supplied by actual positive
escape. No improved growth estimate is assumed. Consequently, any
finite limit of the corresponding coordinate divided by q^k is
strictly positive. Sign reversal of the original equation gives
negative full-factor escape and a strictly negative normalized limit.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- Every actual positively escaping token grows at least by the full original layer factor after a finite layer.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`, for §12's original discrete heads. -/
theorem multiHeadTransformer_positive_full_growth {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atTop) :
    ∃ (K : ℕ) (c : ℝ), 0 < c ∧ ∀ l, c * multiHeadLayerFactor Δt V ^ l ≤ X (K + l) i 0 := by
  let q := multiHeadLayerFactor Δt V
  have hq : 1 < q := multiHeadLayerFactor_gt_one hΔt V hV h
  have hq0 : 0 < q := by linarith
  let B := multiHeadPullCoefficient (m + 1) Q K V
  have hB : 0 ≤ B := (multiHead_coefficients_pos (m + 1) Q K V hV hQK h).2
  let b := Δt * B
  have hb : 0 ≤ b := mul_nonneg hΔt.le hB
  let C := 2 * q * b
  have hC : 0 ≤ C := by dsimp [C]; positivity
  let D := C / (q ^ 2 - 1)
  have hD : 0 ≤ D := div_nonneg hC (by nlinarith)
  let A := max (b + 2) (Real.sqrt D + 2)
  obtain ⟨K, hK⟩ := eventually_atTop.mp (tendsto_atTop.mp hi A)
  let f := fun l => X (K + l) i 0
  have hlarge : ∀ l : ℕ, b + 2 ≤ f l ∧ Real.sqrt D + 2 ≤ f l := by
    intro l
    have hk := hK (K + l) (by omega)
    exact ⟨(le_max_left _ _).trans hk, (le_max_right _ _).trans hk⟩
  have hpos : ∀ l, 0 < f l := fun l => by have hl := (hlarge l).1; linarith
  have hstep : ∀ l, q ^ 2 * f l ^ 2 - C ≤ f (l + 1) ^ 2 := by
    intro l
    have hs := multiHeadTransformer_self_pull hΔt.le hV hQK hX (K + l) i (hpos l)
    have he : q * f l - b / f l = f l + Δt *
        (multiHeadValueCoefficient V * f l - B / f l) := by
      dsimp [q, b, multiHeadLayerFactor]
      ring
    have hrec : q * f l - b / f l ≤ f (l + 1) := by rw [he]; exact hs
    have hsq := (sq_le_sq₀ (show 0 ≤ b + 2 by linarith) (hpos l).le).mpr (hlarge l).1
    have hsqb : b ≤ f l ^ 2 := by nlinarith [sq_nonneg b]
    have hm := mul_le_mul_of_nonneg_right hq.le (sq_nonneg (f l))
    have hdiv : b / f l ≤ q * f l := (div_le_iff₀ (hpos l)).mpr (by nlinarith)
    have hbase : 0 ≤ q * f l - b / f l := sub_nonneg.mpr hdiv
    have hnext := (sq_le_sq₀ hbase (hpos (l + 1)).le).mpr hrec
    have hcancel : q * f l * (b / f l) = q * b := by field_simp [(hpos l).ne']
    dsimp [C]
    nlinarith [sq_nonneg (b / f l)]
  have hinit : C / (q ^ 2 - 1) < f 0 ^ 2 := by
    have hs := (sq_le_sq₀ (show 0 ≤ Real.sqrt D + 2 by positivity) (hpos 0).le).mpr (hlarge 0).2
    have he := Real.sq_sqrt hD
    change D < f 0 ^ 2
    nlinarith [Real.sqrt_nonneg D]
  obtain ⟨c, hc, hbound⟩ := discrete_positive_square_growth f q C hq hC (fun l => (hpos l).le) hstep hinit
  exact ⟨K, c, hc, hbound⟩

/-- Every actual negatively escaping token has the reflected full original layer-factor lower magnitude bound.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`, for §12's original heads. -/
theorem multiHeadTransformer_negative_full_growth {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atBot) :
    ∃ (K : ℕ) (c : ℝ), 0 < c ∧ ∀ l, X (K + l) i 0 ≤ -(c * multiHeadLayerFactor Δt V ^ l) := by
  obtain ⟨K, c, hc, hg⟩ := multiHeadTransformer_positive_full_growth hΔt hV hQK
    (multiHeadTransformer_neg hX) h i (tendsto_neg_atTop_iff.mpr hi)
  refine ⟨K, c, hc, ?_⟩
  intro l
  have hs := hg l
  change c * multiHeadLayerFactor Δt V ^ l ≤ -(X (K + l) i 0) at hs
  linarith

/-- A finite normalized limit of an actual positively escaping original token is strictly positive.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic`, the nonzero rate, for §12's original equation. -/
theorem multiHeadTransformer_positive_limit_pos {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atTop) {γ : ℝ}
    (hlim : Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) : 0 < γ := by
  obtain ⟨K, c, hc, hg⟩ := multiHeadTransformer_positive_full_growth hΔt hV hQK hX h i hi
  let q := multiHeadLayerFactor Δt V
  have hq : 0 < q := by have hge := multiHeadLayerFactor_gt_one hΔt V hV h; linarith
  have ht : Tendsto (fun l => X (K + l) i 0 / q ^ (K + l)) atTop (nhds γ) := by
    simpa only [Nat.add_comm] using (tendsto_add_atTop_iff_nat K).mpr hlim
  have hlower : ∀ l, c / q ^ K ≤ X (K + l) i 0 / q ^ (K + l) := by
    intro l
    have hm := div_le_div_of_nonneg_right (hg l) (pow_pos hq (K + l)).le
    have he : c * q ^ l / q ^ (K + l) = c / q ^ K := by
      rw [pow_add]
      field_simp [(pow_pos hq K).ne', (pow_pos hq l).ne']
    rwa [he] at hm
  exact (div_pos hc (pow_pos hq K)).trans_le (ge_of_tendsto ht (Eventually.of_forall hlower))

/-- A finite normalized limit of an actual negatively escaping original token is strictly negative.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic`, the reflected nonzero rate, for §12. -/
theorem multiHeadTransformer_negative_limit_neg {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atBot) {γ : ℝ}
    (hlim : Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) : γ < 0 := by
  have hn : Tendsto (fun k => (-X k i) 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (-γ)) := by
    simpa only [PiLp.neg_apply, neg_div] using hlim.neg
  have hp := multiHeadTransformer_positive_limit_pos hΔt hV hQK (multiHeadTransformer_neg hX)
    h i (tendsto_neg_atTop_iff.mpr hi) hn
  linarith

/-- Actual positive and negative original singleton trajectories jointly realize all
strict matrix, full-factor escape, and normalized finite-limit premises. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 1 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ MultiHeadTransformer 1 V V V (fun k i => -X k i) ∧
    Tendsto (fun k => X k 0 0) atTop atTop ∧ Tendsto (fun k => (-X k 0) 0) atTop atBot ∧
    Tendsto (fun k => X k 0 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds 1) ∧
    Tendsto (fun k => (-X k 0) 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds (-1)) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  let X := fun k : ℕ => fun _ : Idx 1 => (3 : ℝ) ^ k • unit1
  have hX : MultiHeadTransformer 1 V V V X := by
    convert multiHeadTransformer_singleton_power 1 V V V unit1 using 1
    norm_num [V, X, unit1]
  have he : ∀ k, X k 0 0 = (3 : ℝ) ^ k := fun k => by simp [X, unit1]
  have ht : Tendsto (fun k => X k 0 0) atTop atTop := by
    simp_rw [he]
    exact tendsto_pow_atTop_atTop_of_one_lt (by norm_num)
  have hq : multiHeadLayerFactor 1 V = 3 := by norm_num [multiHeadLayerFactor, multiHeadValueCoefficient, V, unit1]
  have hnorm : ∀ k, X k 0 0 / multiHeadLayerFactor 1 V ^ k = 1 := by
    intro k
    rw [he, hq, div_self (pow_pos (by norm_num : (0 : ℝ) < 3) k).ne']
  have hlim : Tendsto (fun k => X k 0 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds 1) := by
    simp_rw [hnorm]
    exact tendsto_const_nhds
  refine ⟨V, X, one_pos, fun _ => isPosDefOp_id 1,
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1), hX, multiHeadTransformer_neg hX,
    ht, tendsto_neg_atTop_atBot.comp ht, hlim, ?_⟩
  simpa only [PiLp.neg_apply, neg_div] using hlim.neg

end Transformer.Clusters
