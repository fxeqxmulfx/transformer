module

public import Transformer.Clusters.Section12_MultiHeadEscape

/-!
# Attention rows of escaping original multi-head tokens

ArXiv:2305.05465v6, §7, `l:unboundedparticles`, applied separately
to the original positive heads in §12, `sec:conclusion`.
A positive scalar query sees every token a fixed gap below the maximum
with weight at most exp(-gap*query). Normalization gives the diagonal
entry as well. The estimate retains the original query-key coefficient.

The original initial gap persists by the proved discrete separation
estimate. Consequently, every original head row of a token tending to
positive infinity converges to the original maximum's basis row.
Sign reversal gives the minimum's basis row for negative escape.
The hypotheses asserting escape are supplied by the original trajectory
classification; no replacement attention matrix or head is used.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- A separated maximum gives an explicit basis-row error for any positive original scalar query.
Source: arXiv:2305.05465v6, §7, proof of `l:unboundedparticles`, the score-difference estimate. -/
theorem abs_softmaxWeight_sub_le_query (y : Idx (m + 1) → ℝ) (a : ℝ) (ha : 0 < a)
    (M : Idx (m + 1)) (g : ℝ) (hgap : ∀ j, j ≠ M → y j + g ≤ y M) (j : Idx (m + 1)) :
    |Perspective.softmaxWeight (fun l => a * y l) j - (if j = M then 1 else 0)| ≤
      (m + 1) * Real.exp (-(g * a)) := by
  let p := Perspective.softmaxWeight (fun l => a * y l)
  have hoff : ∀ l, l ≠ M → p l ≤ Real.exp (-(g * a)) := fun l hl =>
    (softmaxWeight_le_exp_sub _ M l).trans (Real.exp_le_exp.mpr (by nlinarith [hgap l hl]))
  have he : 0 ≤ Real.exp (-(g * a)) := (Real.exp_pos _).le
  split_ifs with hj
  · subst j
    have hs := Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => a * y l)
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ M)] at hs
    have hle : ∑ l ∈ Finset.univ.erase M, p l ≤ ∑ _l ∈ Finset.univ.erase M, Real.exp (-(g * a)) :=
      Finset.sum_le_sum fun l hl => hoff l (Finset.ne_of_mem_erase hl)
    have hnn : 0 ≤ ∑ l ∈ Finset.univ.erase M, p l :=
      Finset.sum_nonneg fun l _ => Perspective.softmaxWeight_nonneg _ _
    rw [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ M), Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, Nat.add_sub_cancel] at hle
    rw [abs_of_nonpos (by linarith), neg_sub]
    linarith
  · rw [sub_zero, abs_of_nonneg (Perspective.softmaxWeight_nonneg _ _)]
    exact (hoff j hj).trans (le_mul_of_one_le_left he (by linarith [Nat.cast_nonneg (α := ℝ) m]))

/-- Diverging original positive queries converge to the basis row of a uniformly separated maximum.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, for discrete layer indices. -/
theorem scalar_query_row_tendsto (y : ℕ → Idx (m + 1) → ℝ) (a : ℕ → ℝ)
    (M : Idx (m + 1)) {g : ℝ} (hg : 0 < g)
    (hgap : ∀ k j, j ≠ M → y k j + g ≤ y k M) (ha : Tendsto a atTop atTop) (j : Idx (m + 1)) :
    Tendsto (fun k => Perspective.softmaxWeight (fun l => a k * y k l) j) atTop
      (nhds (if j = M then 1 else 0)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  simp only [Real.norm_eq_abs]
  have ht : Tendsto (fun k => -(g * a k)) atTop atBot :=
    tendsto_neg_atTop_atBot.comp (ha.const_mul_atTop hg)
  have hex : Tendsto (fun k : ℕ => Real.exp (-(g * a k))) atTop (nhds (0 : ℝ)) :=
    Real.tendsto_exp_atBot.comp ht
  have he : Tendsto (fun k => ((m : ℝ) + 1) * Real.exp (-(g * a k))) atTop (nhds 0) := by
    simpa only [mul_zero] using hex.const_mul ((m : ℝ) + 1)
  apply squeeze_zero' (Eventually.of_forall fun k => abs_nonneg _) _ he
  filter_upwards [tendsto_atTop.mp ha 1] with k hk
  exact abs_softmaxWeight_sub_le_query (y k) (a k) (by linarith) M g (hgap k) j

/-- Original attention scores are unchanged by simultaneous token sign reversal.
Source: arXiv:2305.05465v6, §7, proof of `l:auxiliary`, the involution, for every head of §12. -/
theorem attentionMatrix_neg_one (Q K : ParamMatrix 1) (Y : Idx (m + 1) → EucSpace 1)
    (i j : Idx (m + 1)) : attentionMatrix Q K (fun l => -Y l) i j = attentionMatrix Q K Y i j := by
  rw [attentionMatrix_eq_one_dim, attentionMatrix_eq_one_dim]
  simp only [PiLp.neg_apply, neg_mul, mul_neg, neg_neg]

/-- A strict initial maximum retains its original uniform gap at every actual layer.
Source: arXiv:2305.05465v6, §7, `e:infdist`, for §12's original discrete heads. -/
theorem multiHeadTransformer_max_gap {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (M : Idx (m + 1)) {g : ℝ} (hg : 0 < g)
    (hgap : ∀ j, j ≠ M → X 0 j 0 + g ≤ X 0 M 0) :
    ∀ k j, j ≠ M → X k j 0 + g ≤ X k M 0 := by
  intro k l hl
  have hinit := hgap l hl
  have hs := multiHeadTransformer_initial_gap hΔt hV hQK hX l M (by linarith) k
  linarith

/-- Every original positive head row of a positive escaping token tends to the original maximum's basis row.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, for the original separate heads of §12. -/
theorem multiHeadTransformer_positive_row {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (i M : Idx (m + 1)) {g : ℝ} (hg : 0 < g)
    (hgap : ∀ j, j ≠ M → X 0 j 0 + g ≤ X 0 M 0) (h : Idx H)
    (hi : Tendsto (fun k => X k i 0) atTop atTop) (j : Idx (m + 1)) :
    Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = M then 1 else 0)) := by
  have hc := (scalar_head_parameters_pos _ _ _ (hV h) (hQK h)).1
  have hcurrent := multiHeadTransformer_max_gap hΔt hV hQK hX M hg hgap
  have ht := scalar_query_row_tendsto (fun k l => X k l 0)
    (fun k => inner (𝕜 := ℝ) (Q h unit1) (K h unit1) * X k i 0) M hg hcurrent
    (hi.const_mul_atTop hc) j
  simpa only [attentionMatrix_eq_one_dim, mul_assoc] using ht

/-- Every original positive head row of a negative escaping token tends to the original minimum's basis row.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, for the original separate heads of §12. -/
theorem multiHeadTransformer_negative_row {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (i L : Idx (m + 1)) {g : ℝ} (hg : 0 < g)
    (hgap : ∀ j, j ≠ L → X 0 L 0 + g ≤ X 0 j 0) (h : Idx H)
    (hi : Tendsto (fun k => X k i 0) atTop atBot) (j : Idx (m + 1)) :
    Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = L then 1 else 0)) := by
  have hgn : ∀ l, l ≠ L → -(X 0 l 0) + g ≤ -(X 0 L 0) := by
    intro l hl
    linarith [hgap l hl]
  have ht := multiHeadTransformer_positive_row hΔt hV hQK (multiHeadTransformer_neg hX) i L hg hgn h
    (tendsto_neg_atTop_iff.mpr hi) j
  simpa only [attentionMatrix_neg_one] using ht

/-- Two separated nonzero tokens and an actual diverging query jointly realize
all static row-error and row-convergence premises. -/
example : ∃ (y : ℕ → Idx 2 → ℝ) (a : ℕ → ℝ),
    (0 : ℝ) < 2 ∧ (∀ k, 0 < a k) ∧ Tendsto a atTop atTop ∧
    (∀ (k : ℕ) (j : Idx 2), j ≠ 1 → y k j + 2 ≤ y k 1) := by
  refine ⟨fun _ => ![-1, 1], fun k => (2 : ℝ) ^ k, by norm_num, fun _ => by positivity,
    tendsto_pow_atTop_atTop_of_one_lt (by norm_num), ?_⟩
  intro k j hj
  fin_cases j <;> norm_num at hj
  norm_num

/-- Actual positive and negative original singleton trajectories realize the escape premises
for every original head; both have a nonzero initial token. -/
example : ∃ X : ℕ → Idx 1 → EucSpace 1,
    MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X ∧
    MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) (fun k i => -X k i) ∧
    Tendsto (fun k => X k 0 0) atTop atTop ∧ Tendsto (fun k => (-X k 0) 0) atTop atBot ∧
    (∀ j : Idx 1, j ≠ 0 → X 0 j 0 + 1 ≤ X 0 0 0) ∧
    (∀ j : Idx 1, j ≠ 0 → (-X 0 0) 0 + 1 ≤ (-X 0 j) 0) := by
  let X := fun k : ℕ => fun _ : Idx 1 => (3 : ℝ) ^ k • unit1
  have hX : MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X := by
    convert multiHeadTransformer_singleton_power (H := 2) 1 (fun _ => (1 : ParamMatrix 1))
      (fun _ => 1) (fun _ => 1) unit1 using 1
    norm_num [X, unit1]
  have ht : Tendsto (fun k => X k 0 0) atTop atTop := by
    have he : (fun k => X k 0 0) = (fun k => (3 : ℝ) ^ k) := by
      funext k
      simp [X, unit1]
    rw [he]
    exact tendsto_pow_atTop_atTop_of_one_lt (by norm_num)
  refine ⟨X, hX, multiHeadTransformer_neg hX, ht, ?_, ?_, ?_⟩
  · exact tendsto_neg_atTop_atBot.comp ht
  · intro j hj; exact (hj (Subsingleton.elim j 0)).elim
  · intro j hj; exact (hj (Subsingleton.elim j 0)).elim

end Transformer.Clusters
