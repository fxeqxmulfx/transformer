/-
# Two tokens under the Pre-LN head

The layer recursion `PreLNHead` at `T = 2`, and three facts about it that hold
for every stream, every temperature and every position:

* the first token attends only to itself, and XSA leaves it a nonnegative
  multiple of its own value, so it stays on its initial ray
  (`preLN_token_zero`);
* the second token's component along the first token's direction never
  decreases (`preLN_token_one`);
* a layer moves a token by at most `2√d`, however large the stream
  (`preLN_drift`, from `attentionHead_norm_le`).

The last is what makes a large stream turn slowly: its direction moves by
`O(√d / ‖x‖)` per layer (`Bridge.norm_inv_smul_sub_le`).  They are the input of
`GPTMini.not_polynomial_rate`.

Source: `reference/model.py` (`Block.forward`), through `PreLNHead`.
-/

import Transformer.GPTMini.ClusteringTheorem
import Transformer.GPTMini.AttentionBounds

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- A lone token attends only to itself, and XSA leaves it a nonnegative
multiple of its own value: the first token only grows along its direction. -/
theorem attentionHead_zero (alpha eps : ℝ) (heps : 0 < eps) (positions : Fin 2 → ℝ)
    (u : Fin 2 → EucSpace cfg.head_dim) :
    ∃ c : ℝ, 0 ≤ c ∧ attentionHead cfg alpha eps (fun j => rmsNormEps eps (u j))
      (fun j => rmsNormEps eps (u j)) (fun j => rmsNormEps eps (u j)) positions 0 = c • u 0 := by
  classical
  set v := fun j => rmsNormEps eps (u j)
  set q := fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (v j)
  have h01 : causalAttnWeights cfg alpha eps q q 0 1 = 0 :=
    causalAttnWeights_zero_above cfg alpha eps q q 0 1 (by decide)
  have h00 : causalAttnWeights cfg alpha eps q q 0 0 = 1 := by
    have := causalAttnWeights_row_sum cfg alpha eps q q 0
    rwa [Fin.sum_univ_two, h01, add_zero] at this
  have hy : attnOutput cfg alpha eps q q v 0 = v 0 := by
    rw [attnOutput, Fin.sum_univ_two, h00, h01, one_smul, zero_smul, add_zero]
  have hw : ‖normL2 eps (v 0)‖ ≤ 1 := normL2_norm_le eps heps.le _
  have hv : v 0 = (Real.sqrt (cfg.head_dim : ℝ) /
      Real.sqrt (‖u 0‖ ^ 2 + (cfg.head_dim : ℝ) * eps)) • u 0 := rfl
  refine ⟨(1 - ‖normL2 eps (v 0)‖ ^ 2) * (Real.sqrt (cfg.head_dim : ℝ) /
      Real.sqrt (‖u 0‖ ^ 2 + (cfg.head_dim : ℝ) * eps)), ?_, ?_⟩
  · have : ‖normL2 eps (v 0)‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg (normL2 eps (v 0))]
    have : 0 ≤ 1 - ‖normL2 eps (v 0)‖ ^ 2 := by linarith
    positivity
  have hxsa : attentionHead cfg alpha eps v v v positions 0
      = v 0 - inner (𝕜 := ℝ) (v 0) (normL2 eps (v 0)) • normL2 eps (v 0) := by
    show xsaProjection cfg eps v (fun i' => if i' = 0 then attnOutput cfg alpha eps q q v 0 else 0) 0 = _
    rw [xsaProjection]; simp only [ite_true, hy]
  rw [hxsa, mul_smul, ← hv]
  have : inner (𝕜 := ℝ) (v 0) (normL2 eps (v 0)) • normL2 eps (v 0)
      = ‖normL2 eps (v 0)‖ ^ 2 • v 0 := by
    rw [normL2, real_inner_smul_right, smul_smul, norm_smul, real_inner_self_eq_norm_sq,
      Real.norm_eq_abs, mul_pow, sq_abs]
    congr 1; ring
  rw [this, sub_smul, one_smul]

/-- The hypothesis of `attentionHead_zero` is satisfiable: `eps = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- The second token moves, by the head, never against the first token's
direction `e`, as long as it does not point against it already: the head is a
nonnegative mixture of the two values with XSA along its own value removed. -/
theorem inner_attentionHead_one_nonneg (alpha eps : ℝ) (heps : 0 < eps)
    (positions : Fin 2 → ℝ) (u : Fin 2 → EucSpace cfg.head_dim) (e : EucSpace cfg.head_dim)
    (k : ℝ) (hk : 0 < k) (hu0 : u 0 = k • e) (hu1 : 0 ≤ inner (𝕜 := ℝ) (u 1) e) :
    0 ≤ inner (𝕜 := ℝ) (attentionHead cfg alpha eps (fun j => rmsNormEps eps (u j))
      (fun j => rmsNormEps eps (u j)) (fun j => rmsNormEps eps (u j)) positions 1) e := by
  classical
  set v := fun j => rmsNormEps eps (u j)
  set q := fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (v j)
  set a := causalAttnWeights cfg alpha eps q q 1 0
  set b := causalAttnWeights cfg alpha eps q q 1 1
  have ha : 0 ≤ a := causalAttnWeights_nonneg cfg alpha eps q q 1 0
  have hb : 0 ≤ b := causalAttnWeights_nonneg cfg alpha eps q q 1 1
  set w := normL2 eps (v 1)
  have hw : ‖w‖ ≤ 1 := normL2_norm_le eps heps.le _
  set c := ‖v 1‖ + eps
  have hc : 0 < c := by positivity
  have hv1 : v 1 = c • w := by
    simp only [w, normL2, smul_smul]; rw [mul_one_div_cancel hc.ne', one_smul]
  set r₀ := Real.sqrt (cfg.head_dim : ℝ) / Real.sqrt (‖u 0‖ ^ 2 + (cfg.head_dim : ℝ) * eps)
  have hr₀ : 0 ≤ r₀ := by positivity
  have hv0 : v 0 = (r₀ * k) • e := by
    show r₀ • u 0 = _; rw [hu0, smul_smul]
  set r₁ := Real.sqrt (cfg.head_dim : ℝ) / Real.sqrt (‖u 1‖ ^ 2 + (cfg.head_dim : ℝ) * eps)
  have hwe : 0 ≤ inner (𝕜 := ℝ) w e := by
    have : w = (1 / c * r₁) • u 1 := by
      simp only [w, normL2, v, rmsNormEps, smul_smul]; rfl
    rw [this, real_inner_smul_left]; positivity
  have hy : attnOutput cfg alpha eps q q v 1 = (a * (r₀ * k)) • e + (b * c) • w := by
    rw [attnOutput, Fin.sum_univ_two, hv0, hv1, smul_smul, smul_smul]
  have hxsa : attentionHead cfg alpha eps v v v positions 1
      = attnOutput cfg alpha eps q q v 1
        - inner (𝕜 := ℝ) (attnOutput cfg alpha eps q q v 1) w • w := by
    show xsaProjection cfg eps v (fun i' => if i' = 1 then attnOutput cfg alpha eps q q v 1 else 0) 1 = _
    rw [xsaProjection]; simp only [ite_true]; rfl
  rw [hxsa, hy]
  simp only [inner_sub_left, inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
  set p := inner (𝕜 := ℝ) w e
  have hp : p ≤ ‖w‖ * ‖e‖ := real_inner_le_norm _ _
  have hwe' : inner (𝕜 := ℝ) e w = p := real_inner_comm _ _
  rw [hwe']
  have hA : 0 ≤ a * (r₀ * k) := by positivity
  have hB : 0 ≤ b * c := by positivity
  have hn : ‖w‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg w]
  have hp2 : p ^ 2 ≤ ‖e‖ ^ 2 := by
    have := mul_le_mul hp hp hwe (by positivity)
    nlinarith [mul_le_mul_of_nonneg_right hn (sq_nonneg ‖e‖)]
  nlinarith [mul_nonneg hA (sub_nonneg.mpr hp2), mul_nonneg (mul_nonneg hB hwe) (sub_nonneg.mpr hn)]

/-- The hypotheses of `inner_attentionHead_one_nonneg` are satisfiable: both
tokens at `e`, `k = 1`, `eps = 1`. -/
example (e : EucSpace cfg.head_dim) :
    (0 : ℝ) < 1 ∧ (fun _ : Fin 2 => e) 0 = (1 : ℝ) • e ∧
      0 ≤ inner (𝕜 := ℝ) ((fun _ : Fin 2 => e) 1) e :=
  ⟨one_pos, (one_smul _ _).symm, real_inner_self_nonneg⟩

/-- The layer recursion from `x₀`, which exists for every `x₀`: the
hypothesis `PreLNHead` of the lemmas below is always satisfiable. -/
noncomputable def preLNRun (alpha eps : ℝ) (positions : Fin 2 → ℝ)
    (x₀ : Fin 2 → EucSpace cfg.head_dim) (L : ℕ) : Fin 2 → EucSpace cfg.head_dim :=
  Nat.rec x₀ (fun _ xL i => xL i + attentionHead cfg alpha eps (fun j => rmsNormEps eps (xL j))
    (fun j => rmsNormEps eps (xL j)) (fun j => rmsNormEps eps (xL j)) positions i) L

theorem preLNHead_preLNRun (alpha eps : ℝ) (positions : Fin 2 → ℝ)
    (x₀ : Fin 2 → EucSpace cfg.head_dim) :
    PreLNHead cfg alpha eps positions (preLNRun cfg alpha eps positions x₀) :=
  fun _ _ => rfl

/-- The hypotheses of the three trajectory lemmas are satisfiable: `eps = 1`,
the run from the origin, whose second token has `⟨x 1, x 0⟩ = 0`. -/
example : (0 : ℝ) < 1 ∧ PreLNHead cfg 0 1 (fun _ => 0) (preLNRun cfg 0 1 (fun _ => 0) 0) ∧
    0 ≤ inner (𝕜 := ℝ) (preLNRun cfg 0 1 (fun _ => 0) 0 0 1) (preLNRun cfg 0 1 (fun _ => 0) 0 0 0) :=
  ⟨one_pos, preLNHead_preLNRun cfg _ _ _ _, by simp [preLNRun]⟩

section Trajectory

variable {cfg} {alpha eps : ℝ} {positions : Fin 2 → ℝ} {x : ℕ → Fin 2 → EucSpace cfg.head_dim}

/-- The first token stays on its initial ray, moving outward. -/
theorem preLN_token_zero (heps : 0 < eps) (hx : PreLNHead cfg alpha eps positions x) (L : ℕ) :
    ∃ k : ℝ, 1 ≤ k ∧ x L 0 = k • x 0 0 := by
  induction L with
  | zero => exact ⟨1, le_rfl, (one_smul _ _).symm⟩
  | succ L ih =>
    obtain ⟨k, hk, hL⟩ := ih
    obtain ⟨c, hc, hhead⟩ := attentionHead_zero cfg alpha eps heps positions (x L)
    refine ⟨(1 + c) * k, by nlinarith, ?_⟩
    rw [hx L 0, hhead, hL, smul_smul, ← add_smul]; ring_nf

/-- The second token's component along the first token's direction never
decreases. -/
theorem preLN_token_one (heps : 0 < eps) (hx : PreLNHead cfg alpha eps positions x)
    (h0 : 0 ≤ inner (𝕜 := ℝ) (x 0 1) (x 0 0)) (L : ℕ) :
    inner (𝕜 := ℝ) (x 0 1) (x 0 0) ≤ inner (𝕜 := ℝ) (x L 1) (x 0 0) := by
  induction L with
  | zero => exact le_rfl
  | succ L ih =>
    obtain ⟨k, hk, hL⟩ := preLN_token_zero heps hx L
    have := inner_attentionHead_one_nonneg cfg alpha eps heps positions (x L) (x 0 0) k
      (by linarith) hL (by linarith)
    rw [hx L 1, inner_add_left]; linarith

/-- Each layer moves a token by at most `2√d`, whatever the size of the stream. -/
theorem preLN_drift (heps : 0 < eps) (hx : PreLNHead cfg alpha eps positions x) (i : Fin 2)
    (L : ℕ) : ‖x L i - x 0 i‖ ≤ 2 * Real.sqrt (cfg.head_dim : ℝ) * L := by
  induction L with
  | zero => simp
  | succ L ih =>
    have hstep : ‖x (L + 1) i - x L i‖ ≤ 2 * Real.sqrt (cfg.head_dim : ℝ) := by
      rw [hx L i, add_sub_cancel_left]
      exact attentionHead_norm_le cfg alpha eps heps.le _ _ _ positions i _
        fun j => rmsNormEps_norm_le eps heps cfg.head_dim_pos _
    calc ‖x (L + 1) i - x 0 i‖ ≤ ‖x (L + 1) i - x L i‖ + ‖x L i - x 0 i‖ := norm_sub_le_norm_sub_add_norm_sub _ _ _
      _ ≤ 2 * Real.sqrt (cfg.head_dim : ℝ) + 2 * Real.sqrt (cfg.head_dim : ℝ) * L := add_le_add hstep ih
      _ = 2 * Real.sqrt (cfg.head_dim : ℝ) * ((L + 1 : ℕ) : ℝ) := by push_cast; ring

end Trajectory
end GPTMini
end Transformer
