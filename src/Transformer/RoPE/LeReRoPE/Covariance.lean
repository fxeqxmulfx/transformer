module

public import Transformer.RoPE.LeRoPE.SectionB1_Covariance
public import Transformer.RoPE.LeReRoPE.Fold

/-!
# LeReRoPE: a frequency's gradient as a covariance under ReRoPE's clamp

In no paper. LeRoPE's Lemma B.3 (Karypis et al., arXiv:2607.10134v1, Appendix B.1;
`Transformer.RoPE.LeRoPE.hasDerivAt_loss_cov`) writes the gradient of band `m`'s frequency as
`−Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S_st)`, with `p_s` the attention of the query at `s` on the keys
`t < s` and `S_st = (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ)`. With the pair `(s, t)` rotated by
`ψ(s − t)` (`positionedLogits`):

* `hasDerivAt_positioned_loss_cov`: `∂L/∂θ = −Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S^ψ_st)`, with
  `S^ψ_st = ψ(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + ψ(s − t) θ)` (`positionedSine`);
  `positionedSine_natCast`: `ψ(δ) = δ` gives LeRoPE's `S_st`;
* `positionedSine_rectified_of_le`: under ReRoPE's clamp every pair at least `w` apart has
  `S^ψ_st = w ‖q_s‖ ‖k_t‖ sin(φ_st + w θ)`, the term of the offset `w`;
* `abs_positionedSine_rectified_le`: so `|S^ψ_st| ≤ min(s − t, w) ‖q_s‖ ‖k_t‖`. A pair's weight
  in the frequency's gradient stops growing at the window, where LeRoPE's carries the factor
  `s − t`; Leaky ReRoPE's grows beyond the window with the slope `1/k`;
* `abs_sum_weightedCov_rectified_le`, `abs_deriv_rectified_loss_le`: hence, at every context
  length, `|∂L/∂θ| ≤ w Σ_s Σ_{t<s} α_st ‖q_s‖ ‖k_t‖ |(v_t − o_s)ᵀ g_s|`, the bound LeRoPE has
  with `s − t` in place of `w`.
-/

@[expose] public section

open scoped ComplexConjugate InnerProductSpace
open Finset

namespace Transformer.RoPE.LeReRoPE

open Transformer.AttentionResiduals Transformer.RoPE.LeRoPE Transformer.RoPE.ReRoPE

variable {n : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The sine term of the pair `(s, t)` at the positions `ψ`,
`S^ψ_st = ψ(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + ψ(s − t) θ)`: `∂D_st/∂θ = −S^ψ_st`. -/
noncomputable def positionedSine (ψ : ℕ → ℝ) (q k : Fin n → ℂ) (θ : ℝ) (s t : Fin n) : ℝ :=
  ψ ((s : ℕ) - t) * ‖q s‖ * ‖k t‖ *
    Real.sin (Complex.arg (q s * conj (k t)) + ψ ((s : ℕ) - t) * θ)

/-- `ψ(δ) = δ` gives LeRoPE's `S_st` on the causal pairs `t ≤ s`. -/
theorem positionedSine_natCast (q k : Fin n → ℂ) (θ : ℝ) {s t : Fin n} (hts : t ≤ s) :
    positionedSine (fun δ : ℕ => (δ : ℝ)) q k θ s t = offsetSine q k θ s t := by
  rw [positionedSine, offsetSine, Nat.cast_sub (Fin.le_def.1 hts)]

/-- The hypothesis of `positionedSine_natCast` holds for the pair `(1, 0)`. -/
example (q k : Fin 2 → ℂ) (θ : ℝ) := positionedSine_natCast q k θ (s := 1) (t := 0) (by decide)

/-- **LeRoPE's Lemma B.3 at the positions `ψ`**: for a loss `L` of the outputs of causal
attention on the logits `positionedLogits ψ r q k θ`, differentiable there with the gradients
`g_s = ∂L/∂o_s`, `∂L/∂θ = −Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S^ψ_st)`. -/
theorem hasDerivAt_positioned_loss_cov (ψ : ℕ → ℝ) {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ}
    (v : Fin n → E) {L : (Fin n → E) → ℝ} {L' : (Fin n → E) →L[ℝ] ℝ} {g : Fin n → E} {θ : ℝ}
    (hL : HasFDerivAt L L' (causalOutput (positionedLogits ψ r q k θ) v))
    (hg : ∀ s x, L' (Pi.single s x) = ⟪g s, x⟫_ℝ) :
    HasDerivAt (fun θ => L (causalOutput (positionedLogits ψ r q k θ) v))
      (-∑ s, weightedCov (causalWeight (positionedLogits ψ r q k θ) s) (fun t => ⟪v t, g s⟫_ℝ)
        fun t => positionedSine ψ q k θ s t) θ := by
  have hz : HasDerivAt (positionedLogits ψ r q k) (fun p => -positionedSine ψ q k θ p.1 p.2) θ :=
    hasDerivAt_pi.2 fun p => ((hasDerivAt_bandProduct _ _ _ _ θ).const_add (r p)).congr_deriv
      (by rw [positionedSine, sub_zero]; ring)
  refine (hasDerivAt_loss_causal hz v hL hg).congr_deriv ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun s _ => weightedCov_neg _ _ _

/-- The hypotheses of `hasDerivAt_positioned_loss_cov` hold for ReRoPE's positions with the window
`1` and the loss `⟪c, o_1⟫` of two tokens, whose gradients are `c` at `1` and `0` at `0`. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (v : Fin 2 → E) (c : E) (θ : ℝ) :=
  hasDerivAt_positioned_loss_cov (fun δ => rectifiedPosition 1 δ) (r := r) (q := q) (k := k) v
    (g := Pi.single 1 c) (θ := θ)
    ((innerSL ℝ c).comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => E) 1)).hasFDerivAt
    fun s x => by fin_cases s <;> simp

/-- Under ReRoPE's clamp **every pair at least `w` apart has the sine term of the offset `w`**:
`S^ψ_st = w ‖q_s‖ ‖k_t‖ sin(φ_st + w θ)`. -/
theorem positionedSine_rectified_of_le {w : ℕ} (q k : Fin n → ℂ) (θ : ℝ) {s t : Fin n}
    (h : w ≤ (s : ℕ) - t) :
    positionedSine (fun δ => rectifiedPosition w δ) q k θ s t =
      w * ‖q s‖ * ‖k t‖ * Real.sin (Complex.arg (q s * conj (k t)) + w * θ) := by
  have hw : (w : ℝ) ≤ (((s : ℕ) - t : ℕ) : ℝ) := by exact_mod_cast h
  simp only [positionedSine, rectifiedPosition, min_eq_right hw]

/-- The hypothesis of `positionedSine_rectified_of_le` holds for the window `1` and the pair
`(1, 0)`. -/
example (q k : Fin 2 → ℂ) (θ : ℝ) :=
  positionedSine_rectified_of_le (w := 1) q k θ (s := 1) (t := 0) (by decide)

/-- Under ReRoPE's clamp **a pair's weight in the frequency's gradient stops growing at the
window**: `|S^ψ_st| ≤ min(s − t, w) ‖q_s‖ ‖k_t‖`. -/
theorem abs_positionedSine_rectified_le (w : ℕ) (q k : Fin n → ℂ) (θ : ℝ) (s t : Fin n) :
    |positionedSine (fun δ => rectifiedPosition w δ) q k θ s t| ≤
      min (((s : ℕ) - t : ℕ) : ℝ) w * ‖q s‖ * ‖k t‖ := by
  simp only [positionedSine, rectifiedPosition]
  rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg (le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _)),
    abs_norm, abs_norm]
  exact mul_le_of_le_one_right (by positivity) (Real.abs_sin_le_one _)

/-- Under ReRoPE's clamp **the covariance form is bounded at the window**: with the weights
`α_st` and the outputs `o_s` of causal attention on any logits `z`,
`|Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S^ψ_st)| ≤ w Σ_s Σ_{t<s} α_st ‖q_s‖ ‖k_t‖ |(v_t − o_s)ᵀ g_s|`. -/
theorem abs_sum_weightedCov_rectified_le (w : ℕ) (z : Fin n × Fin n → ℝ) (q k : Fin n → ℂ)
    (v g : Fin n → E) (θ : ℝ) :
    |∑ s, weightedCov (causalWeight z s) (fun t => ⟪v t, g s⟫_ℝ)
        (fun t => positionedSine (fun δ => rectifiedPosition w δ) q k θ s t)| ≤
      w * ∑ s, ∑ t, causalWeight z s t *
        (‖q s‖ * ‖k t‖ * |⟪v t - causalOutput z v s, g s⟫_ℝ|) := by
  rw [Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun s _ => ?_)
  rcases isEmpty_or_nonempty (Iio s) with hs | hs
  · simp [weightedCov]
  have hX : ∀ t : Iio s, ⟪v t, g s⟫_ℝ - ∑ u, causalWeight z s u * ⟪v u, g s⟫_ℝ =
      ⟪v t - causalOutput z v s, g s⟫_ℝ := fun t => by
    rw [inner_sub_left, causalOutput, attend, Residual.weightedSum, sum_inner]
    simp only [real_inner_smul_left]
    rfl
  have hp : ∑ t, causalWeight z s t = 1 := depthWeight_sum _
  rw [← sum_mul_sub_eq_weightedCov hp]
  simp only [hX, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun t _ => ?_)
  have hS : |positionedSine (fun δ => rectifiedPosition w δ) q k θ s t| ≤ w * ‖q s‖ * ‖k t‖ :=
    (abs_positionedSine_rectified_le w q k θ s t).trans (by gcongr; exact min_le_right _ _)
  have hα : 0 ≤ causalWeight z s t := (depthWeight_pos _ _).le
  rw [abs_mul, abs_mul, abs_of_nonneg hα]
  exact (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hS hα)
    (abs_nonneg _)).trans_eq (by ring)

/-- Under ReRoPE's clamp **the frequency's gradient is bounded at the window**, at every context
length: `|∂L/∂θ| ≤ w Σ_s Σ_{t<s} α_st ‖q_s‖ ‖k_t‖ |(v_t − o_s)ᵀ g_s|`, where LeRoPE's `S_st`
carries `s − t` in place of `w`. -/
theorem abs_deriv_rectified_loss_le (w : ℕ) {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ}
    (v : Fin n → E) {L : (Fin n → E) → ℝ} {L' : (Fin n → E) →L[ℝ] ℝ} {g : Fin n → E} {θ : ℝ}
    (hL : HasFDerivAt L L'
      (causalOutput (positionedLogits (fun δ => rectifiedPosition w δ) r q k θ) v))
    (hg : ∀ s x, L' (Pi.single s x) = ⟪g s, x⟫_ℝ) :
    |deriv (fun θ => L
        (causalOutput (positionedLogits (fun δ => rectifiedPosition w δ) r q k θ) v)) θ| ≤
      w * ∑ s, ∑ t, causalWeight (positionedLogits (fun δ => rectifiedPosition w δ) r q k θ) s t *
        (‖q s‖ * ‖k t‖ * |⟪v t - causalOutput
          (positionedLogits (fun δ => rectifiedPosition w δ) r q k θ) v s, g s⟫_ℝ|) := by
  rw [(hasDerivAt_positioned_loss_cov _ v hL hg).deriv, abs_neg]
  exact abs_sum_weightedCov_rectified_le w _ q k v g θ

/-- The hypotheses of `abs_deriv_rectified_loss_le` hold for the window `1` and the loss
`⟪c, o_1⟫` of two tokens. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (v : Fin 2 → E) (c : E) (θ : ℝ) :=
  abs_deriv_rectified_loss_le 1 (r := r) (q := q) (k := k) v (g := Pi.single 1 c) (θ := θ)
    ((innerSL ℝ c).comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => E) 1)).hasFDerivAt
    fun s x => by fin_cases s <;> simp

end Transformer.RoPE.LeReRoPE
