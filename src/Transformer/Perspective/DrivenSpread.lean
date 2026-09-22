/-
# Driven flows do not collapse, on any window

**Not a statement of any paper.**  The theorem that `Perspective.DrivenFlow`
prepares: along `ẋ_i = Proj_{x_i}(v_i(t))` on a window of length `τ`, if two
drive differences stay near two independent vectors whenever the tokens are
close — `‖v_k - v_l - w₁‖ ≤ L ε` and `‖v_m - v_p - w₂‖ ≤ L ε` while the tokens
are pairwise `ε`-close — and the drives are bounded by `M`, then at some time
of the window two tokens are more than `δ` apart, where `δ` depends on `w₁`,
`w₂`, `L`, `M` and `τ` only (`IsDrivenFlowOn.spread`).

Unlike `IsInjectedFlow.spread` the window is arbitrary and short: `τ` is not
the theorem's to choose, and the drive is asked to be of this form on the window
only.  A block of a residual network that runs for time `τ` is one such window;
a stack of different blocks is a sequence of them.
-/

import Transformer.Perspective.DrivenFlow

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- **Theorem (no collapse on a window).**  Let `w₁`, `w₂` be linearly
independent and `L, M ≥ 0`, `τ > 0`.  There is `δ > 0`, depending on `w₁`,
`w₂`, `L`, `M` and `τ` only, such that along every driven flow
`ẋ_i = Proj_{x_i}(v_i(t))` on a window `[t₀, t₀ + τ]` whose drives are bounded
by `M` and whose drive differences satisfy, at every time of the window,

  `‖v_k - v_l - w₁‖ ≤ L ε` and `‖v_m - v_p - w₂‖ ≤ L ε`
  whenever the tokens are pairwise within `ε`,

some time of the window has two tokens more than `δ` apart.  `δ` does not
depend on the number of tokens, on the flow, on the drives or on `t₀`.

The proof: `ψ = ⟨x_k - x_l, w₁⟩ + ⟨x_m - x_p, w₂⟩` satisfies `|ψ| ≤ δ W`,
`W = ‖w₁‖ + ‖w₂‖`, while the tokens are `δ`-close, and then
`ψ̇ ≥ c - K δ ≥ c / 2` (`le_inner_proj_sub_proj_add`, `δ ≤ c / (2K)`), so over
the window `ψ` would rise by `c τ / 2 > 2 δ W` (`δ ≤ c τ / (8W)`).

Source: none — posed here, as the counterpart of `lem: hemisphere.clustering`
(arXiv:2312.10794v5, §6.1) for a drive that is asked to have this form on one
window only. -/
theorem IsDrivenFlowOn.spread {w₁ w₂ : EucSpace d} (hw : LinearIndependent ℝ ![w₁, w₂])
    {L M τ : ℝ} (hL : 0 ≤ L) (hM : 0 ≤ M) (hτ : 0 < τ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (n : ℕ) (X : ℝ → SphereTuple d n) (v : ℝ → Idx n → EucSpace d)
      (t₀ : ℝ) (k l m p : Idx n), IsDrivenFlowOn X v t₀ (t₀ + τ) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ i, ‖v t i‖ ≤ M) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ ε : ℝ, 0 ≤ ε →
        (∀ i j, ‖(X t i : EucSpace d) - X t j‖ ≤ ε) →
          ‖v t k - v t l - w₁‖ ≤ L * ε ∧ ‖v t m - v t p - w₂‖ ≤ L * ε) →
      ∃ t ∈ Set.Icc t₀ (t₀ + τ), ∃ i j : Idx n, δ < ‖(X t i : EucSpace d) - X t j‖ := by
  have hw₁ : 0 < ‖w₁‖ := norm_pos_iff.2 (by simpa using hw.ne_zero 0)
  have hw₂ : 0 < ‖w₂‖ := norm_pos_iff.2 (by simpa using hw.ne_zero 1)
  obtain ⟨c, hc_def⟩ : ∃ c, c = (‖w₁‖ ^ 2 * ‖w₂‖ ^ 2 - (inner (𝕜 := ℝ) w₁ w₂) ^ 2)
      / (2 * (‖w₁‖ ^ 2 + ‖w₂‖ ^ 2)) := ⟨_, rfl⟩
  obtain ⟨K, hK_def⟩ : ∃ K, K = (L + 2 * M) * (‖w₁‖ + ‖w₂‖) + 2 * ‖w₂‖ ^ 2 := ⟨_, rfl⟩
  obtain ⟨W, hW_def⟩ : ∃ W, W = ‖w₁‖ + ‖w₂‖ := ⟨_, rfl⟩
  have hc : 0 < c := by
    rw [hc_def]
    exact div_pos (sub_pos.2 (sq_inner_lt_of_linearIndependent hw)) (by positivity)
  have hK : 0 < K := by rw [hK_def]; positivity
  have hW : 0 < W := by rw [hW_def]; positivity
  obtain ⟨δ, hδ_def⟩ : ∃ δ, δ = min (c / (2 * K)) (c * τ / (8 * W)) := ⟨_, rfl⟩
  have hδ : 0 < δ := by rw [hδ_def]; exact lt_min (by positivity) (by positivity)
  have hKδ : K * δ ≤ c / 2 :=
    calc K * δ ≤ K * (c / (2 * K)) :=
          mul_le_mul_of_nonneg_left (by rw [hδ_def]; exact min_le_left _ _) hK.le
      _ = c / 2 := by rw [← mul_div_assoc, mul_comm K c, mul_div_mul_right c 2 hK.ne']
  have hWδ : δ * W ≤ c * τ / 8 :=
    calc δ * W ≤ c * τ / (8 * W) * W :=
          mul_le_mul_of_nonneg_right (by rw [hδ_def]; exact min_le_right _ _) hW.le
      _ = c * τ / 8 := by rw [div_mul_eq_mul_div, mul_div_mul_right _ _ hW.ne']
  refine ⟨δ, hδ, fun n X v t₀ k l m p hX hv hvw => ?_⟩
  by_contra hcon
  push Not at hcon
  obtain ⟨ψ, hψ_def⟩ : ∃ ψ : ℝ → ℝ, ψ = fun s =>
      inner (𝕜 := ℝ) ((X s k : EucSpace d) - X s l) w₁
        + inner (𝕜 := ℝ) ((X s m : EucSpace d) - X s p) w₂ := ⟨_, rfl⟩
  have hψ : ∀ s ∈ Set.Ioo t₀ (t₀ + τ), HasDerivAt ψ
      (inner (𝕜 := ℝ) (proj d (X s k) (v s k) - proj d (X s l) (v s l)) w₁
        + inner (𝕜 := ℝ) (proj d (X s m) (v s m) - proj d (X s p) (v s p)) w₂) s := by
    intro s hs
    rw [hψ_def]
    convert (((hX.2 s hs k).fun_sub (hX.2 s hs l)).inner ℝ (hasDerivAt_const s w₁)).fun_add
      (((hX.2 s hs m).fun_sub (hX.2 s hs p)).inner ℝ (hasDerivAt_const s w₂)) using 1
    simp only [inner_zero_right, zero_add]
  have hcont : ContinuousOn ψ (Set.Icc t₀ (t₀ + τ)) := by
    rw [hψ_def]
    exact (((hX.1 k).sub (hX.1 l)).inner continuousOn_const).add
      (((hX.1 m).sub (hX.1 p)).inner continuousOn_const)
  have hle : t₀ ≤ t₀ + τ := le_add_of_nonneg_right hτ.le
  have hmvt := (convex_Icc t₀ (t₀ + τ)).mul_sub_le_image_sub_of_le_deriv hcont
    (fun s hs => (hψ s (by rwa [interior_Icc] at hs)).differentiableAt.differentiableWithinAt)
    (C := c / 2) (fun s hs => by
      rw [interior_Icc] at hs
      have hs' := Set.Ioo_subset_Icc_self hs
      obtain ⟨h₁, h₂⟩ := hvw s hs δ hδ.le (hcon s hs')
      have h := le_inner_proj_sub_proj_add (X s) (v s) k l m p w₁ w₂ hδ.le (hcon s hs')
        (hv s hs) h₁ h₂
      rw [← hc_def, ← hK_def] at h
      rw [(hψ s hs).deriv]
      linarith)
    t₀ (Set.left_mem_Icc.2 hle) (t₀ + τ) (Set.right_mem_Icc.2 hle) hle
  have hbound : ∀ s ∈ Set.Icc t₀ (t₀ + τ), |ψ s| ≤ δ * W := fun s hs => by
    have h1 := (abs_real_inner_le_norm ((X s k : EucSpace d) - X s l) w₁).trans
      (mul_le_mul_of_nonneg_right (hcon s hs k l) (norm_nonneg _))
    have h2 := (abs_real_inner_le_norm ((X s m : EucSpace d) - X s p) w₂).trans
      (mul_le_mul_of_nonneg_right (hcon s hs m p) (norm_nonneg _))
    rw [hψ_def, hW_def, mul_add]
    exact (abs_add_le _ _).trans (add_le_add h1 h2)
  obtain ⟨h0, -⟩ := abs_le.1 (hbound t₀ (Set.left_mem_Icc.2 hle))
  obtain ⟨-, h1⟩ := abs_le.1 (hbound (t₀ + τ) (Set.right_mem_Icc.2 hle))
  rw [add_sub_cancel_left] at hmvt
  linarith [mul_pos hc hτ]

/-- The hypotheses of `IsDrivenFlowOn.spread` are satisfiable, those inside its
conclusion included: in `ℝ²`, `w₁ = e₀`, `w₂ = e₁`, `L = 0`, `M = 1`, and three
tokens `e₀`, `e₀`, `e₁` at rest under the drives `0`, `-e₀`, `-e₁`, each normal
to the sphere at its token, with `v₀ - v₁ = w₁` and `v₀ - v₂ = w₂` exactly. -/
example : ∃ (w₁ w₂ : EucSpace 2) (X : ℝ → SphereTuple 2 3) (v : ℝ → Idx 3 → EucSpace 2),
    LinearIndependent ℝ ![w₁, w₂] ∧ IsDrivenFlowOn X v 0 (0 + 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ i, ‖v t i‖ ≤ 1) ∧
    ∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ ε : ℝ, 0 ≤ ε →
      (∀ i j, ‖(X t i : EucSpace 2) - X t j‖ ≤ ε) →
        ‖v t 0 - v t 1 - w₁‖ ≤ 0 * ε ∧ ‖v t 0 - v t 2 - w₂‖ ≤ 0 * ε := by
  refine ⟨EuclideanSpace.single 0 1, EuclideanSpace.single 1 1,
    fun _ => ![basePoint 1, basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩],
    fun _ => ![0, -EuclideanSpace.single 0 1, -EuclideanSpace.single 1 1], ?_,
    ⟨fun _ => continuousOn_const, fun t _ i => ?_⟩, fun _ _ i => ?_, fun _ _ _ _ _ => ?_⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases i <;> simp
    · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]
  · refine (hasDerivAt_const t _).congr_deriv ?_
    fin_cases i <;> simp [proj, basePoint]
  · fin_cases i <;> simp
  · simp

end Perspective
end Transformer
