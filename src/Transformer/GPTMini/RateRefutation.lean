/-
# The depth rate of `gpt-mini` is not uniform in the initial stream

`polynomial_rate` stood in `GPTMini.ClusteringTheorem` as

> there is `C`, depending on a temperature bound `A` alone, such that for almost
> every initial stream the token directions reach a common limit at the rate
> `‖Φ(x_L(i)) - x_∞‖ ≤ C / L³`,

citing `thm: preln-slow` of arXiv:2510.22026v2.  That theorem says nothing of
the kind: it states `r_k(t) ≥ (1-δ)t` and `Var' = -Θ(Var/t)` for the
continuous-time Pre-LN dynamics started in a narrow cone; the exponent `3`,
the almost-every initial stream and the uniformity of `C` were not in it.  The
statement is false, and this file proves it.

A layer adds at most `2√d` to a token, whatever the size of the stream, so a
token at distance `R` from the origin turns by `O(√d / R)` per layer.  With
two tokens, the first stays on its ray forever and fixes the limit; the second,
started far out at an angle of `π/4` from it, is still at an angle after
`L₀ ≪ R / √d` layers, while `C / L₀³` is already small.  Such starts form an
open set, of positive Lebesgue measure, so "almost every" does not save the
statement.

Source: the former `GPTMini.polynomial_rate`; arXiv:2510.22026v2,
`thm: preln-slow`, for what the cited theorem does say.
-/

import Transformer.GPTMini.TwoTokens

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- **`polynomial_rate` is false.**

For every temperature bound `A ≥ 0` and every head dimension `d ≥ 3` there is
no `C` with `‖Φ(x_L(i)) - x_∞‖ ≤ C / L³` for almost every initial stream of
two tokens: the negation of the former statement at `T = 2`.  The failing
starts are the ball of radius `1/40` about `(e₀, R(e₀ + e₁))`, with
`R = 64 √d L₀` and `L₀ = ⌈8C⌉ + 1`; at layer `L₀` the second token's
direction is still `> 3/8` from the first's, while both must be within
`C / L₀³ ≤ 1/8` of `x_∞`.

Source: the former `GPTMini.polynomial_rate`, citing arXiv:2510.22026v2,
`thm: preln-slow`. -/
theorem not_polynomial_rate (A : ℝ) (hA : 0 ≤ A) (hd : 3 ≤ cfg.head_dim) :
    ¬ ∃ C : ℝ, 0 < C ∧
    ∀ (alpha eps : ℝ) (positions : Fin 2 → ℝ), |alpha| ≤ A → 0 < eps →
      ∀ᵐ x₀ : Fin 2 → EucSpace cfg.head_dim, ∀ x : ℕ → Fin 2 → EucSpace cfg.head_dim,
        x 0 = x₀ → PreLNHead cfg alpha eps positions x →
        (∀ (L : ℕ) (i : Fin 2), x L i ≠ 0) →
          ∃ xinf : EucSpace cfg.head_dim, ‖xinf‖ = 1 ∧
            ∀ (L : ℕ), 1 ≤ L → ∀ i : Fin 2,
              ‖Bridge.toSphere cfg.head_dim (x L i) - xinf‖ ≤ C / (L : ℝ) ^ 3 := by
  rintro ⟨C, hC, h⟩
  have hae := h 0 1 (fun _ => 0) (by simpa using hA) one_pos
  rw [ae_iff] at hae
  have hd0 : 0 < cfg.head_dim := by omega
  set e₀ : EucSpace cfg.head_dim := EuclideanSpace.single ⟨0, hd0⟩ 1
  set e₁ : EucSpace cfg.head_dim := EuclideanSpace.single ⟨1, by omega⟩ 1
  have he₀ : ‖e₀‖ = 1 := by simp [e₀]
  have he₁ : ‖e₁‖ = 1 := by simp [e₁]
  have he₀₁ : inner (𝕜 := ℝ) e₀ e₁ = 0 := by simp [e₀, e₁, EuclideanSpace.inner_single_left]
  have he₁₁ : inner (𝕜 := ℝ) e₁ e₁ = 1 := by simp [e₁]
  have he₀₀ : inner (𝕜 := ℝ) e₀ e₀ = 1 := by simp [e₀]
  have he₁₀ : inner (𝕜 := ℝ) e₁ e₀ = 0 := by rw [real_inner_comm]; exact he₀₁
  set L₀ : ℕ := ⌈8 * C⌉₊ + 1
  have hL₀ : (1 : ℝ) ≤ L₀ := by simp [L₀]
  have hL₀C : 8 * C ≤ L₀ := by
    simp only [L₀]; push_cast; linarith [Nat.le_ceil (8 * C)]
  have hsd : 1 ≤ Real.sqrt (cfg.head_dim : ℝ) :=
    Real.one_le_sqrt.mpr (by exact_mod_cast hd0)
  set R : ℝ := 64 * Real.sqrt (cfg.head_dim : ℝ) * L₀
  have hR : 64 ≤ R := by
    have := mul_le_mul hsd hL₀ zero_le_one (by positivity); simp only [R]; nlinarith
  set p : Fin 2 → EucSpace cfg.head_dim := ![e₀, R • (e₀ + e₁)]
  refine ((Metric.isOpen_ball (x := p)).measure_pos volume
    (Metric.nonempty_ball.2 (by norm_num : (0 : ℝ) < 1 / 40))).ne'
    (measure_mono_null (fun x₀ hx₀ => ?_) hae)
  simp only [Set.mem_ofPred_eq]
  intro hQ
  rw [Metric.mem_ball, dist_pi_lt_iff (by norm_num)] at hx₀
  have hδ0 : ‖x₀ 0 - e₀‖ < 1 / 40 := by simpa [p, dist_eq_norm] using hx₀ 0
  have hδ1 : ‖x₀ 1 - R • (e₀ + e₁)‖ < 1 / 40 := by simpa [p, dist_eq_norm] using hx₀ 1
  -- the first token sits near `e₀`, the second far out near `R (e₀ + e₁)`
  have hn0 : 39 / 40 ≤ ‖x₀ 0‖ := by
    have := norm_sub_norm_le e₀ (e₀ - x₀ 0); rw [sub_sub_cancel, norm_sub_rev] at this; linarith
  have hn0' : ‖x₀ 0‖ ≤ 41 / 40 := by
    have := norm_le_insert' (x₀ 0) e₀; linarith
  have hi01 : inner (𝕜 := ℝ) (x₀ 0) e₁ ≤ 1 / 40 := by
    have h := real_inner_le_norm (x₀ 0 - e₀) e₁
    rw [inner_sub_left, he₀₁, sub_zero, he₁, mul_one] at h; linarith
  have hsum : ‖e₀ + e₁‖ ≤ 2 := by linarith [norm_add_le e₀ e₁]
  have hi11 : R - 1 / 40 ≤ inner (𝕜 := ℝ) (x₀ 1) e₁ := by
    have h := neg_le_of_abs_le ((abs_real_inner_le_norm (x₀ 1 - R • (e₀ + e₁)) e₁).trans
      (by rw [he₁, mul_one]))
    rw [inner_sub_left, real_inner_smul_left, inner_add_left, he₀₁, he₁₁] at h; linarith
  have hi10 : R - 1 / 40 ≤ inner (𝕜 := ℝ) (x₀ 1) e₀ := by
    have h := neg_le_of_abs_le ((abs_real_inner_le_norm (x₀ 1 - R • (e₀ + e₁)) e₀).trans
      (by rw [he₀, mul_one]))
    rw [inner_sub_left, real_inner_smul_left, inner_add_left, he₀₀, he₁₀] at h; linarith
  have hn1 : R / 2 ≤ ‖x₀ 1‖ := by
    have := real_inner_le_norm (x₀ 1) e₀; rw [he₀, mul_one] at this; linarith
  have hn1' : ‖x₀ 1‖ ≤ 2 * R + 1 / 40 := by
    have h1 := norm_le_insert' (x₀ 1) (R • (e₀ + e₁))
    have h2 : ‖R • (e₀ + e₁)‖ ≤ 2 * R := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by linarith)]; nlinarith
    linarith
  have hpos : 0 < inner (𝕜 := ℝ) (x₀ 1) (x₀ 0) := by
    have h := neg_le_of_abs_le (abs_real_inner_le_norm (R • (e₀ + e₁)) (x₀ 0 - e₀))
    have h' := neg_le_of_abs_le (abs_real_inner_le_norm (x₀ 1 - R • (e₀ + e₁)) (x₀ 0))
    have h2 : ‖R • (e₀ + e₁)‖ ≤ 2 * R := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by linarith)]; nlinarith
    have hx0e : inner (𝕜 := ℝ) (R • (e₀ + e₁)) e₀ = R := by
      rw [real_inner_smul_left, inner_add_left, he₀₀, he₁₀]; ring
    have hsplit : inner (𝕜 := ℝ) (x₀ 1) (x₀ 0) = inner (𝕜 := ℝ) (x₀ 1 - R • (e₀ + e₁)) (x₀ 0)
        + inner (𝕜 := ℝ) (R • (e₀ + e₁)) (x₀ 0 - e₀) + R := by
      rw [inner_sub_left, inner_sub_right, hx0e]; ring
    have b1 : ‖R • (e₀ + e₁)‖ * ‖x₀ 0 - e₀‖ ≤ 2 * R * (1 / 40) :=
      mul_le_mul h2 hδ0.le (norm_nonneg _) (by linarith)
    have b2 : ‖x₀ 1 - R • (e₀ + e₁)‖ * ‖x₀ 0‖ ≤ 1 / 40 * (41 / 40) :=
      mul_le_mul hδ1.le hn0' (norm_nonneg _) (by norm_num)
    linarith
  have hx00 : x₀ 0 ≠ 0 := norm_pos_iff.mp (by linarith)
  have hx01 : x₀ 1 ≠ 0 := norm_pos_iff.mp (by linarith)
  -- the trajectory from `x₀`
  set x := preLNRun cfg 0 1 (fun _ => 0) x₀
  have hx : PreLNHead cfg 0 1 (fun _ => 0) x := preLNHead_preLNRun cfg _ _ _ _
  have hz0 := preLN_token_zero one_pos hx
  have hz1 := preLN_token_one one_pos hx hpos.le
  have hnz : ∀ (L : ℕ) (i : Fin 2), x L i ≠ 0 := by
    refine fun L => Fin.forall_fin_two.mpr ⟨?_, fun h0 => ?_⟩
    · obtain ⟨k, hk, hL⟩ := hz0 L
      exact hL ▸ smul_ne_zero (by positivity) hx00
    · have := hz1 L
      rw [h0, inner_zero_left] at this
      exact absurd this (not_le.mpr hpos)
  obtain ⟨xinf, -, hb⟩ := hQ x rfl hx hnz
  have hC3 : C / (L₀ : ℝ) ^ 3 ≤ 1 / 8 := by
    rw [div_le_iff₀ (by positivity)]
    nlinarith [pow_le_pow_right₀ hL₀ (by norm_num : 1 ≤ 3), pow_one (L₀ : ℝ)]
  have hb0 := (hb L₀ (by simp [L₀]) 0).trans hC3
  have hb1 := (hb L₀ (by simp [L₀]) 1).trans hC3
  -- the first token's direction is `Φ(x₀ 0)` at every layer
  obtain ⟨k, hk, hL0⟩ := hz0 L₀
  change x L₀ 0 = k • x₀ 0 at hL0
  have hdir0 : Bridge.toSphere cfg.head_dim (x L₀ 0) = ‖x₀ 0‖⁻¹ • x₀ 0 := by
    rw [Bridge.toSphere_eq hd0 (hnz L₀ 0), hL0, norm_smul, Real.norm_eq_abs,
      abs_of_pos (by linarith), smul_smul]
    congr 1; field_simp
  -- the second token has barely turned
  have hdir1 : ‖Bridge.toSphere cfg.head_dim (x L₀ 1) - ‖x₀ 1‖⁻¹ • x₀ 1‖ ≤ 1 / 8 := by
    rw [Bridge.toSphere_eq hd0 (hnz L₀ 1)]
    refine (Bridge.norm_inv_smul_sub_le _ hx01).trans ?_
    rw [div_le_iff₀ (by linarith)]
    have := preLN_drift one_pos hx 1 L₀
    change ‖x L₀ 1 - x₀ 1‖ ≤ _ at this
    have hR' : R / 2 = 32 * Real.sqrt (cfg.head_dim : ℝ) * L₀ := by simp only [R]; ring
    nlinarith
  -- so the two directions are within `3/8`, but they are `> 3/8` apart along `e₁`
  have hclose : ‖‖x₀ 1‖⁻¹ • x₀ 1 - ‖x₀ 0‖⁻¹ • x₀ 0‖ ≤ 3 / 8 := by
    rw [← hdir0]
    calc _ ≤ ‖‖x₀ 1‖⁻¹ • x₀ 1 - Bridge.toSphere cfg.head_dim (x L₀ 1)‖
          + ‖Bridge.toSphere cfg.head_dim (x L₀ 1) - xinf‖
          + ‖xinf - Bridge.toSphere cfg.head_dim (x L₀ 0)‖ :=
          (norm_sub_le_norm_sub_add_norm_sub _ xinf _).trans
            (by gcongr; exact norm_sub_le_norm_sub_add_norm_sub _ _ _)
      _ ≤ 1 / 8 + 1 / 8 + 1 / 8 := by
          rw [norm_sub_rev (‖x₀ 1‖⁻¹ • x₀ 1), norm_sub_rev xinf]; gcongr
      _ = 3 / 8 := by norm_num
  have hfar := real_inner_le_norm (‖x₀ 1‖⁻¹ • x₀ 1 - ‖x₀ 0‖⁻¹ • x₀ 0) e₁
  rw [he₁, mul_one, inner_sub_left, real_inner_smul_left, real_inner_smul_left] at hfar
  have hq1 : 0.45 ≤ ‖x₀ 1‖⁻¹ * inner (𝕜 := ℝ) (x₀ 1) e₁ := by
    rw [inv_mul_eq_div, le_div_iff₀ (by linarith)]; nlinarith
  have hq0 : ‖x₀ 0‖⁻¹ * inner (𝕜 := ℝ) (x₀ 0) e₁ ≤ 1 / 39 := by
    rw [inv_mul_eq_div, div_le_iff₀ (by linarith)]; nlinarith
  norm_num at hq1
  linarith

/-- The hypotheses of `not_polynomial_rate` are satisfiable: `A = 0` and the
default config, `d_head = 64`. -/
example : (0 : ℝ) ≤ 0 ∧ 3 ≤ Config.default.head_dim := by
  refine ⟨le_rfl, ?_⟩
  norm_num [Config.head_dim, Config.default]

end GPTMini
end Transformer
