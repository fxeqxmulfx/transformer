/-
# Collinear differences of injections allow consensus

**Not a statement of any paper.**  The hypothesis of `blockDrive_spread` is two
linearly independent *differences* of injections, `z_k - z_l` and `z_m - z_p`.
For `V = I_d`, `IsInjectedFlow.spread` asks only for two independent
injections; for a block with other values that is not enough, and the
hypothesis on differences is sharp:

* if every difference `z_i - z_{i₀}` is a multiple of one unit vector `x`, all
  tokens resting at `x` is a flow driven by a block with one head, any weights
  whose rows sum to one — softmax attention of any `β`, `Q`, `K` among them —
  and the rank-one value `V y = -⟨x, y⟩ z_{i₀}` (`blockDrive_consensus`);
* two tokens always have collinear differences, so for `n = 2` nothing about
  the injections alone rules collapse out (`blockDrive_consensus_two`);
* `z_0 = e₁` and `z_1 = e₀ + e₁` are independent, yet two tokens resting at
  `e₀` is such a flow (`not_blockDrive_spread_of_independent`).

If the differences `z_i - z_j` span two dimensions, `blockDrive_spread`
applies; if they span at most one, `blockDrive_consensus` does.
-/

import Transformer.Perspective.BlockDrive
import Transformer.Perspective.InjectedAttention

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Collinear differences of injections allow consensus.**  If every
difference `z_i(t) - z_{i₀}(t)` is a multiple of one unit vector `x`, then all
tokens resting at `x` is a flow driven by a block (`blockDrive`) with one head,
any weights whose rows sum to one, the rank-one value
`V(t) y = -⟨x, y⟩ z_{i₀}(t)` and no feed-forward term: at `x` the head returns
`-z_{i₀}`, and what is left of the drive of token `i`, `z_i - z_{i₀}`, is normal
to the sphere.  The weights may be softmax attention of any `β`, `Q` and `K`
(`sum_attention`).

Source: none — posed here, against `blockDrive_spread`. -/
theorem blockDrive_consensus (x : SSphere d) (a : ℝ → Idx 1 → Idx n → Idx n → ℝ)
    (ha : ∀ t h i, ∑ j, a t h i j = 1) (z : ℝ → Idx n → EucSpace d) (i₀ : Idx n)
    (hz : ∀ t i, ∃ c : ℝ, z t i - z t i₀ = c • (x : EucSpace d)) (t₀ t₁ : ℝ) :
    IsDrivenFlowOn (fun _ _ => x)
      (blockDrive a (fun t _ => -(innerSL ℝ (x : EucSpace d)).smulRight (z t i₀))
        (fun _ _ => 0) z (fun _ _ => x)) t₀ t₁ := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  refine ⟨fun _ => continuousOn_const, fun t _ i => (hasDerivAt_const t _).congr_deriv ?_⟩
  obtain ⟨c, hc⟩ := hz t i
  have hdrive : blockDrive a (fun t _ => -(innerSL ℝ (x : EucSpace d)).smulRight (z t i₀))
      (fun _ _ => 0) z (fun _ _ => x) t i = c • (x : EucSpace d) := by
    rw [← hc]
    simp only [blockDrive, neg_apply, ContinuousLinearMap.smulRight_apply,
      innerSL_apply_apply, real_inner_self_eq_norm_sq, hx, one_pow, one_smul, smul_neg,
      Finset.sum_neg_distrib, ← Finset.sum_smul, ha, Fin.sum_univ_one, add_zero]
    abel
  beta_reduce
  rw [hdrive, proj_smul_self hx]

/-- The hypotheses of `blockDrive_consensus` are satisfiable, by softmax
attention of any `β`, `Q`, `K` on any configuration `Y`, and by three
injections `e₁`, `e₁ + e₀`, `e₁ - 2 e₀` in `ℝ²` whose differences from the first
are multiples of `x = e₀`. -/
example (β : ℝ) (Q K : TimeParam 2) (Y : ℝ → SphereTuple 2 3) :
    ∃ (a : ℝ → Idx 1 → Idx 3 → Idx 3 → ℝ) (z : ℝ → Idx 3 → EucSpace 2),
      (∀ t h i, ∑ j, a t h i j = 1) ∧
      ∀ t i, ∃ c : ℝ, z t i - z t 0 = c • (basePoint 1 : EucSpace 2) := by
  refine ⟨fun t _ i j => attention 2 3 β Q K Y t i j, fun _ => ![EuclideanSpace.single 1 1,
    EuclideanSpace.single 1 1 + EuclideanSpace.single 0 1,
    EuclideanSpace.single 1 1 - (2 : ℝ) • EuclideanSpace.single 0 1],
    fun t _ i => sum_attention β Q K Y t i, fun _ i => ?_⟩
  fin_cases i
  · exact ⟨0, by simp⟩
  · exact ⟨1, by simp [basePoint]⟩
  · exact ⟨-2, by simp [basePoint]⟩

/-- **Two tokens can always reach consensus.**  For any two injections `z_0`,
`z_1` in `ℝ^{d+1}` there is a unit vector `x` — along `z_1 - z_0` when they
differ — at which two resting tokens are a flow driven by a block as in
`blockDrive_consensus`.  So for `n = 2` nothing about the injections alone
rules collapse out once `V ≠ I_d`, and `blockDrive_spread`, whose two
independent differences need three tokens, has nothing to say.

Source: none — posed here, against `blockDrive_spread`. -/
theorem blockDrive_consensus_two (a : ℝ → Idx 1 → Idx 2 → Idx 2 → ℝ)
    (ha : ∀ t h i, ∑ j, a t h i j = 1) (z : Idx 2 → EucSpace (d + 1)) (t₀ t₁ : ℝ) :
    ∃ x : SSphere (d + 1), IsDrivenFlowOn (fun _ _ => x)
      (blockDrive a (fun _ _ => -(innerSL ℝ (x : EucSpace (d + 1))).smulRight (z 0))
        (fun _ _ => 0) (fun _ => z) (fun _ _ => x)) t₀ t₁ := by
  by_cases h : z 1 = z 0
  · refine ⟨basePoint d, blockDrive_consensus _ a ha (fun _ => z) 0 (fun _ i => ⟨0, ?_⟩) t₀ t₁⟩
    fin_cases i <;> simp [h]
  · have hne : z 1 - z 0 ≠ 0 := sub_ne_zero.2 h
    refine ⟨⟨‖z 1 - z 0‖⁻¹ • (z 1 - z 0), mem_sphere_zero_iff_norm.2 (norm_smul_inv_norm hne)⟩,
      blockDrive_consensus _ a ha (fun _ => z) 0 (fun _ i => ?_) t₀ t₁⟩
    fin_cases i
    · exact ⟨0, by simp⟩
    · exact ⟨‖z 1 - z 0‖, by simp [smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.2 hne)]⟩

/-- The hypothesis of `blockDrive_consensus_two` is satisfiable, by softmax
attention of any `β`, `Q`, `K` on any configuration `Y`. -/
example (β : ℝ) (Q K : TimeParam 2) (Y : ℝ → SphereTuple 2 2) :
    ∃ a : ℝ → Idx 1 → Idx 2 → Idx 2 → ℝ, ∀ t h i, ∑ j, a t h i j = 1 :=
  ⟨fun t _ i j => attention 2 2 β Q K Y t i j, fun t _ i => sum_attention β Q K Y t i⟩

/-- **Two independent injections do not rule collapse out once `V ≠ I_d`.**
The claim refuted: `blockDrive_spread` with two linearly independent injections
`z_0`, `z_1` in place of two independent differences — the hypothesis of
`IsInjectedFlow.spread`, where `V = I_d`.  It fails already for two tokens in
`ℝ²`, one head of uniform weights (softmax attention at `β = 0`), values of
norm at most `1`, no feed-forward term and injections of norm at most `2`:
`z_0 = e₁` and `z_1 = e₀ + e₁` are independent, and both tokens resting at
`e₀` is a flow driven by such a block (`blockDrive_consensus` with
`V y = -⟨e₀, y⟩ e₁`), whose tokens never part.

Source: none — posed here; it refutes the extension of `IsInjectedFlow.spread`
to `V ≠ I_d`. -/
theorem not_blockDrive_spread_of_independent :
    ¬ ∀ z₀ z₁ : EucSpace 2, LinearIndependent ℝ ![z₀, z₁] →
      ∃ δ : ℝ, 0 < δ ∧ ∀ (X : ℝ → SphereTuple 2 2) (a : ℝ → Idx 1 → Idx 2 → Idx 2 → ℝ)
        (V : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2),
        IsDrivenFlowOn X (blockDrive a V (fun _ _ => 0) (fun _ => ![z₀, z₁]) X) 0 1 →
        IsBoundedBlockOn a V (fun _ _ => 0) (fun _ => ![z₀, z₁]) 0 1 1 1 0 0 2 →
        ∃ t ∈ Set.Icc (0 : ℝ) 1, ∃ i j : Idx 2, δ < ‖(X t i : EucSpace 2) - X t j‖ := by
  intro h
  have hind : LinearIndependent ℝ ![(EuclideanSpace.single 1 1 : EucSpace 2),
      EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1] := by
    rw [LinearIndependent.pair_iff]
    intro s t hst
    have h0 := congrArg (inner ℝ (EuclideanSpace.single 0 1 : EucSpace 2)) hst
    have h1 := congrArg (inner ℝ (EuclideanSpace.single 1 1 : EucSpace 2)) hst
    simp [inner_add_right, inner_smul_right, EuclideanSpace.inner_single_left] at h0 h1
    exact ⟨by linarith, h0⟩
  obtain ⟨δ, hδ, hsp⟩ := h _ _ hind
  have hflow := blockDrive_consensus (basePoint 1) (fun _ _ _ _ => 1 / 2)
    (fun _ _ _ => by norm_num [Fin.sum_univ_two])
    (fun _ => ![EuclideanSpace.single 1 1,
      EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1]) 0 (fun _ i => by
        fin_cases i
        · exact ⟨0, by simp⟩
        · exact ⟨1, by simp [basePoint]⟩) 0 1
  obtain ⟨t, -, i, j, hij⟩ := hsp _ _ _ hflow
    ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ => by norm_num [Fin.sum_univ_two],
      fun _ _ _ => by simp [basePoint], fun _ _ _ => by simp, fun _ _ _ _ => by simp,
      fun _ _ i => by
        fin_cases i
        · simp
        · exact (norm_add_le _ _).trans (by norm_num)⟩
  simp at hij
  linarith

end Perspective
end Transformer
