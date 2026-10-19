module

public import Transformer.Clusters.Section12_MultiHeadScalar
public import Transformer.Clusters.Section7_OnlyOneEscape

/-!
# Negative contributions to each original positive scalar head

ArXiv:2305.05465v6, §7, `e:minorationpourxn` and the proof of
`l:onlyone`, with the separate original head coefficients of §12.
A positive query receives the largest positive token with weight at
least 1/n. All other negative terms are bounded from below by their
unnormalized exponential contributions, because the original partition
function is at least one. This bounds the average below by the query
itself divided by n plus those negative contributions.

If a query stays above a positive threshold, an anchor tending to
negative infinity contributes arbitrarily little to that lower bound.
The same holds for an anchor eventually above every negative epsilon.
The statements concern the actual softmax of each original head and
allow different positive query-key coefficients.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m : ℕ}

/-- An original positive scalar head pulls toward the maximum, up to its explicit negative terms.
Source: arXiv:2305.05465v6, §7, `e:minorationpourxn`, for an original head of §12. -/
theorem scalarHeadAverage_ge_max_add_min (c : ℝ) (hc : 0 < c)
    (y : Idx (m + 1) → ℝ) (i M : Idx (m + 1)) (hi : 0 < y i) (hM : ∀ j, y j ≤ y M) :
    y M / ((m : ℝ) + 1) + ∑ j, min 0 (Real.exp (c * y i * y j) * y j) ≤
      scalarHeadAverage c y (y i) := by
  let p := Perspective.softmaxWeight (fun l => c * y i * y l)
  have hMp : 0 < y M := hi.trans_le (hM i)
  have hle : ∀ j, p j ≤ Real.exp (c * y i * y j) :=
    softmaxWeight_le_exp_of_nonneg y (c * y i) M (mul_pos (mul_pos hc hi) hMp).le
  have hterm : ∀ j, (if j = M then y M / ((m : ℝ) + 1) else 0) +
      min 0 (Real.exp (c * y i * y j) * y j) ≤ p j * y j := by
    intro j
    by_cases hj : j = M
    · subst j
      have h1 := inv_le_softmaxWeight_of_max y (mul_pos hc hi).le M hM
      have h2 : min 0 (Real.exp (c * y i * y M) * y M) = 0 :=
        min_eq_left (mul_pos (Real.exp_pos _) hMp).le
      rw [ite_eq_left rfl, h2, add_zero, div_eq_mul_one_div, mul_comm (p M)]
      exact mul_le_mul_of_nonneg_left h1 hMp.le
    · rw [ite_eq_right hj, zero_add]
      rcases le_or_gt 0 (y j) with hp | hn
      · exact (min_le_left _ _).trans (mul_nonneg (Perspective.softmaxWeight_nonneg _ _) hp)
      · exact (min_le_right _ _).trans (mul_le_mul_of_nonpos_right (hle j) hn.le)
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hterm j
  rwa [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ M, ite_eq_left (Finset.mem_univ M)] at hsum

/-- The positive original query itself gives a uniform positive term in the head-average lower bound.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, from `e:minorationpourxn`, for §12. -/
theorem scalarHeadAverage_ge_self_add_min (c : ℝ) (hc : 0 < c)
    (y : Idx (m + 1) → ℝ) (i : Idx (m + 1)) (hi : 0 < y i) :
    y i / ((m : ℝ) + 1) + ∑ j, min 0 (Real.exp (c * y i * y j) * y j) ≤
      scalarHeadAverage c y (y i) := by
  obtain ⟨M, _, hM⟩ := Finset.exists_max_image Finset.univ y Finset.univ_nonempty
  have hd := scalarHeadAverage_ge_max_add_min c hc y i M hi (fun j => hM j (Finset.mem_univ j))
  have he := div_le_div_of_nonneg_right (hM i (Finset.mem_univ i))
    (show 0 ≤ (m : ℝ) + 1 by positivity)
  linarith

/-- Negative original head terms vanish under a uniformly positive query,
for anchors approaching the nonnegative half-line or tending to negative infinity.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, `e:minorationpourxn`, for §12. -/
theorem scalar_head_negative_part_eventually (c : ℝ) (hc : 0 < c) (r z : ℕ → ℝ)
    {δ η : ℝ} (hδ : 0 < δ) (hη : 0 < η) (hr : ∀ᶠ k in atTop, δ ≤ r k)
    (hz : (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ z k) ∨ Tendsto z atTop atBot) :
    ∀ᶠ k in atTop, -η ≤ min 0 (Real.exp (c * r k * z k) * z k) := by
  rcases hz with hz | hz
  · filter_upwards [hz η hη, hr] with k hk hkr
    refine le_min (neg_nonpos.mpr hη.le) ?_
    rcases le_or_gt 0 (z k) with hpos | hneg
    · exact (neg_nonpos.mpr hη.le).trans (mul_nonneg (Real.exp_pos _).le hpos)
    · have he : Real.exp (c * r k * z k) ≤ 1 :=
        Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos (mul_pos hc (hδ.trans_le hkr)).le hneg.le)
      nlinarith
  · have hcδ : 0 < c * δ := mul_pos hc hδ
    let A := 2 / ((c * δ) ^ 2 * η)
    have hA : 0 < A := by dsimp [A]; positivity
    filter_upwards [tendsto_atBot.mp hz (-A), hr] with k hk hkr
    have hzk : 0 < -z k := hA.trans_le (by linarith)
    have he := exp_neg_mul_mul_le hcδ hzk
    have hsmall : 2 / ((c * δ) ^ 2 * -z k) ≤ η := by
      rw [div_le_iff₀ (by positivity)]
      have hcancel : A * ((c * δ) ^ 2 * η) = 2 := by dsimp [A]; field_simp
      have hm := mul_le_mul_of_nonneg_left (show A ≤ -z k by linarith)
        (show 0 ≤ (c * δ) ^ 2 * η by positivity)
      nlinarith
    have hcoeff := mul_le_mul_of_nonneg_left hkr hc.le
    have hprod := mul_le_mul_of_nonpos_right hcoeff (show z k ≤ 0 by linarith)
    have hex : Real.exp (c * r k * z k) ≤ Real.exp (-((c * δ) * -z k)) :=
      Real.exp_le_exp.mpr (by nlinarith)
    have hweighted := mul_le_mul_of_nonneg_right hex hzk.le
    refine le_min (neg_nonpos.mpr hη.le) ?_
    nlinarith

/-- An original head has a uniformly positive average when its query stays positive
and the negative anchor terms vanish as in the source's bounded-token argument.
Source: arXiv:2305.05465v6, §7, `e:minorationpourxn`, for an original head of §12. -/
theorem scalarHeadAverage_eventually_ge_of_positive_query (c : ℝ) (hc : 0 < c)
    (y : ℕ → Idx (m + 1) → ℝ) (i : Idx (m + 1)) {δ : ℝ} (hδ : 0 < δ)
    (hr : ∀ᶠ k in atTop, δ ≤ y k i)
    (htok : ∀ j, (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ y k j) ∨
      Tendsto (fun k => y k j) atTop atBot) :
    ∀ᶠ k in atTop, δ / (2 * ((m : ℝ) + 1)) ≤ scalarHeadAverage c (y k) (y k i) := by
  let N : ℝ := (m : ℝ) + 1
  have hN : 0 < N := by dsimp [N]; positivity
  let η := δ / (2 * N ^ 2)
  have hη : 0 < η := by dsimp [η]; positivity
  have hterm : ∀ j, ∀ᶠ k in atTop, -η ≤ min 0 (Real.exp (c * y k i * y k j) * y k j) :=
    fun j => scalar_head_negative_part_eventually c hc (fun k => y k i) (fun k => y k j) hδ hη hr (htok j)
  filter_upwards [hr, eventually_all.mpr hterm] with k hq ht
  have hs := scalarHeadAverage_ge_self_add_min c hc (y k) i (hδ.trans_le hq)
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => ht j
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_add, Nat.cast_one] at hsum
  have hquot := div_le_div_of_nonneg_right hq hN.le
  have he : N * (-η) = -(δ / N) + δ / (2 * N) := by
    dsimp [η]
    field_simp
    ring
  change N * (-η) ≤ _ at hsum
  rw [he] at hsum
  change δ / (2 * N) ≤ _
  change y k i / N + _ ≤ _ at hs
  linarith

/-- Two distinct nonzero original tokens realize the positive-query and maximum premises. -/
example : ∃ y : Idx 2 → ℝ, (0 : ℝ) < 2 ∧ 0 < y 1 ∧ (∀ j, y j ≤ y 1) := by
  refine ⟨![-1, 1], by norm_num, by norm_num, ?_⟩
  intro j
  fin_cases j <;> norm_num

/-- A nonzero constant query and a zero anchor jointly realize the first negative-term alternative. -/
example : ∃ r z : ℕ → ℝ, (0 : ℝ) < 1 ∧ (∀ᶠ k in atTop, (1 : ℝ) ≤ r k) ∧
    (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ z k) := by
  refine ⟨fun _ => 1, fun _ => 0, one_pos, Eventually.of_forall fun _ => le_rfl, ?_⟩
  intro ε hε
  exact Eventually.of_forall fun _ => by linarith

/-- A nonzero constant query and a genuinely escaping anchor realize the second negative-term alternative. -/
example : ∃ r z : ℕ → ℝ, (0 : ℝ) < 1 ∧ (∀ᶠ k in atTop, (1 : ℝ) ≤ r k) ∧ Tendsto z atTop atBot := by
  refine ⟨fun _ => 1, fun k => -(2 : ℝ) ^ k, one_pos, Eventually.of_forall fun _ => le_rfl, ?_⟩
  exact tendsto_neg_atTop_atBot.comp (tendsto_pow_atTop_atTop_of_one_lt (by norm_num))

/-- A positive bounded anchor family realizes every query and anchor premise
of the eventual positive head-average estimate. -/
example : ∃ y : ℕ → Idx 2 → ℝ, (∀ᶠ k in atTop, (1 : ℝ) ≤ y k 0) ∧
    (∀ j, ∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ y k j) := by
  refine ⟨fun _ => ![1, 2], Eventually.of_forall fun _ => by norm_num, ?_⟩
  intro j ε hε
  apply Eventually.of_forall
  intro k
  fin_cases j <;> norm_num <;> linarith

end Transformer.Clusters
