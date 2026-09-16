/-
# How far the attention sub-layer sends two residual streams apart

`HeadLipschitz` estimates one head; this file wraps the reshapes of
`reference/model.py` (`CausalMHA.forward`) around it and reaches the whole
sub-layer of `Block.attnSubLayer`.

The chain, for two streams `x, y` no further apart than `D` at any position:

  - `rmsNormEps` is `2/√eps`-Lipschitz and `W_qkv` is linear, so the projected
    vector moves by at most `G = 2 ‖W_qkv‖ D / √eps`;
  - `qkvSlice` and `headSlice` read each coordinate at most once, so every
    head's `q`, `k`, `v` moves by at most `G` as well;
  - the values of the second stream are bounded by `B = ‖W_qkv‖ √d_model`,
    exactly as in `Block.attnSubLayer_bounded`;
  - each head moves by `headLipschitz (log α_h) eps B · G`, and
    `headLipschitz_mono` replaces `log α_h` by a uniform `α_max`;
  - `headMerge` collects `n_heads` of them, costing `√n_heads`, and `W_o`
    costs `‖W_o‖`.

Nothing here is local: the constant does not depend on `D`, so the attention
sub-layer is Lipschitz on the whole residual stream space.
-/

import Transformer.GPTMini.Block
import Transformer.GPTMini.ReshapeDist
import Transformer.GPTMini.HeadLipschitz

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- The `q`, `k` or `v` stream that `attnSubLayer` hands to head `h`: RMSNorm,
the QKV projection, the `chunk(3, dim=-1)` selected by `e`, and finally the
`view(T, n_heads, head_dim)` of head `h`.  Source: `reference/model.py`
(`CausalMHA.forward`). -/
noncomputable def attnHeadInput (cfg : Config) (params : AttnParams cfg) (eps : ℝ)
    (e : Fin cfg.d_model → Fin (3 * cfg.d_model))
    {T : ℕ} (x : Fin T → EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) (j : Fin T) : EucSpace cfg.head_dim :=
  headSlice cfg (qkvSlice cfg e (params.W_qkv (rmsNormEps eps (x j)))) h

/-- `attnSubLayer` with its `let`s expanded. -/
theorem attnSubLayer_eq (cfg : Config) (params : AttnParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    attnSubLayer cfg params eps positions x i
      = params.W_o (headMerge cfg (fun h =>
          attentionHead cfg (params.log_alpha h) eps
            (attnHeadInput cfg params eps (qkvQ cfg) x h)
            (attnHeadInput cfg params eps (qkvK cfg) x h)
            (attnHeadInput cfg params eps (qkvV cfg) x h) positions i)) := rfl

/-- **The attention sub-layer is Lipschitz.**

With `B = ‖W_qkv‖ √d_model` the value scale and `α_max` an upper bound for
every `log α_h`,

  `‖attn(x) i - attn(y) i‖ ≤ ‖W_o‖ √n_heads · L(α_max, eps, B) · 2‖W_qkv‖D/√eps`

whenever the two streams are `D`-close at every position.  The estimate is
uniform in the position because the head mixes them: a bound at `i` alone
would not do.

Source: `reference/model.py` (`Block.forward`, first sub-layer), through
`GPTMini.rmsNormEps_lipschitz`, the reshape bounds above, and
`GPTMini.attentionHead_dist_le`. -/
theorem attnSubLayer_dist_le
    (cfg : Config) (params : AttnParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ) (alpha_max : ℝ)
    (halpha : ∀ h, params.log_alpha h ≤ alpha_max)
    (x y : Fin T → EucSpace cfg.d_model) (D : ℝ)
    (hD : ∀ j, ‖x j - y j‖ ≤ D) (i : Fin T) :
    ‖attnSubLayer cfg params eps positions x i
        - attnSubLayer cfg params eps positions y i‖
      ≤ ‖params.W_o‖ * Real.sqrt (cfg.n_heads : ℝ)
          * headLipschitz alpha_max eps (‖params.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ))
          * (2 * ‖params.W_qkv‖ / Real.sqrt eps * D) := by
  have hD0 : 0 ≤ D := (norm_nonneg _).trans (hD i)
  have hsqrt : (0 : ℝ) < Real.sqrt eps := Real.sqrt_pos.mpr heps
  set B := ‖params.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ) with hBdef
  set G := 2 * ‖params.W_qkv‖ / Real.sqrt eps * D with hGdef
  have hB0 : 0 ≤ B := by rw [hBdef]; positivity
  have hG0 : 0 ≤ G := by rw [hGdef]; positivity
  -- the projected vector moves by at most `G`
  have hproj : ∀ j : Fin T,
      ‖params.W_qkv (rmsNormEps eps (x j))
        - params.W_qkv (rmsNormEps eps (y j))‖ ≤ G := by
    intro j
    rw [← map_sub]
    refine (params.W_qkv.le_opNorm _).trans ?_
    have h1 : ‖rmsNormEps eps (x j) - rmsNormEps eps (y j)‖
        ≤ 2 / Real.sqrt eps * D := by
      refine (rmsNormEps_lipschitz eps heps cfg.d_model_pos (x j) (y j)).trans ?_
      have : (0 : ℝ) ≤ 2 / Real.sqrt eps := by positivity
      exact mul_le_mul_of_nonneg_left (hD j) this
    calc ‖params.W_qkv‖ * ‖rmsNormEps eps (x j) - rmsNormEps eps (y j)‖
        ≤ ‖params.W_qkv‖ * (2 / Real.sqrt eps * D) :=
          mul_le_mul_of_nonneg_left h1 (norm_nonneg _)
      _ = G := by rw [hGdef]; ring
  -- hence so does every head's `q`, `k` and `v`
  have hslice : ∀ (e : Fin cfg.d_model → Fin (3 * cfg.d_model)),
      Function.Injective e → ∀ (h : Fin cfg.n_heads) (j : Fin T),
      ‖attnHeadInput cfg params eps e x h j - attnHeadInput cfg params eps e y h j‖ ≤ G :=
    fun e he h j =>
      ((headSlice_dist_le cfg _ _ h).trans (qkvSlice_dist_le cfg e he _ _)).trans (hproj j)
  -- the values of the second stream stay inside the ball of radius `B`
  have hvalue : ∀ (h : Fin cfg.n_heads) (j : Fin T),
      ‖attnHeadInput cfg params eps (qkvV cfg) y h j‖ ≤ B := by
    intro h j
    refine (headSlice_norm_le cfg _ h).trans ?_
    refine (qkvSlice_norm_le cfg _ (qkvV_injective cfg) _).trans ?_
    refine (params.W_qkv.le_opNorm _).trans ?_
    exact mul_le_mul_of_nonneg_left
      (rmsNormEps_norm_le eps heps cfg.d_model_pos _) (norm_nonneg _)
  -- each head moves by at most `L(α_max, eps, B) · G`
  have hhead : ∀ h : Fin cfg.n_heads,
      ‖attentionHead cfg (params.log_alpha h) eps
          (attnHeadInput cfg params eps (qkvQ cfg) x h)
          (attnHeadInput cfg params eps (qkvK cfg) x h)
          (attnHeadInput cfg params eps (qkvV cfg) x h) positions i
        - attentionHead cfg (params.log_alpha h) eps
          (attnHeadInput cfg params eps (qkvQ cfg) y h)
          (attnHeadInput cfg params eps (qkvK cfg) y h)
          (attnHeadInput cfg params eps (qkvV cfg) y h) positions i‖
      ≤ headLipschitz alpha_max eps B * G := by
    intro h
    refine (attentionHead_dist_le cfg (params.log_alpha h) eps heps _ _ _ _ _ _
      positions i B G (hvalue h)
      (hslice _ (qkvQ_injective cfg) h) (hslice _ (qkvK_injective cfg) h)
      (hslice _ (qkvV_injective cfg) h)).trans ?_
    exact mul_le_mul_of_nonneg_right
      (headLipschitz_mono _ _ eps B heps hB0 (halpha h)) hG0
  have hmerge := headMerge_dist_le cfg _ _ (headLipschitz alpha_max eps B * G) hhead
  calc ‖attnSubLayer cfg params eps positions x i
          - attnSubLayer cfg params eps positions y i‖
      = ‖params.W_o (headMerge cfg (fun h =>
            attentionHead cfg (params.log_alpha h) eps
              (attnHeadInput cfg params eps (qkvQ cfg) x h)
              (attnHeadInput cfg params eps (qkvK cfg) x h)
              (attnHeadInput cfg params eps (qkvV cfg) x h) positions i)
          - headMerge cfg (fun h =>
            attentionHead cfg (params.log_alpha h) eps
              (attnHeadInput cfg params eps (qkvQ cfg) y h)
              (attnHeadInput cfg params eps (qkvK cfg) y h)
              (attnHeadInput cfg params eps (qkvV cfg) y h) positions i))‖ := by
        rw [map_sub, ← attnSubLayer_eq, ← attnSubLayer_eq]
    _ ≤ ‖params.W_o‖ * ‖headMerge cfg (fun h =>
            attentionHead cfg (params.log_alpha h) eps
              (attnHeadInput cfg params eps (qkvQ cfg) x h)
              (attnHeadInput cfg params eps (qkvK cfg) x h)
              (attnHeadInput cfg params eps (qkvV cfg) x h) positions i)
          - headMerge cfg (fun h =>
            attentionHead cfg (params.log_alpha h) eps
              (attnHeadInput cfg params eps (qkvQ cfg) y h)
              (attnHeadInput cfg params eps (qkvK cfg) y h)
              (attnHeadInput cfg params eps (qkvV cfg) y h) positions i)‖ :=
        params.W_o.le_opNorm _
    _ ≤ ‖params.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * (headLipschitz alpha_max eps B * G)) :=
        mul_le_mul_of_nonneg_left hmerge (norm_nonneg _)
    _ = ‖params.W_o‖ * Real.sqrt (cfg.n_heads : ℝ) * headLipschitz alpha_max eps B * G := by
        ring

/-- The hypotheses are satisfiable: the `eps = 10⁻⁶` of `reference/model.py`,
`α_max` the largest `log α` of the block itself, and the stream compared with
itself at `D = 0`. -/
example (params : AttnParams Config.default)
    (positions : Fin 1 → ℝ) (x : Fin 1 → EucSpace Config.default.d_model)
    (alpha_max : ℝ) (halpha : ∀ h, params.log_alpha h ≤ alpha_max) :
    ‖attnSubLayer Config.default params 1e-6 positions x 0
        - attnSubLayer Config.default params 1e-6 positions x 0‖
      ≤ ‖params.W_o‖ * Real.sqrt (Config.default.n_heads : ℝ)
          * headLipschitz alpha_max 1e-6
              (‖params.W_qkv‖ * Real.sqrt (Config.default.d_model : ℝ))
          * (2 * ‖params.W_qkv‖ / Real.sqrt 1e-6 * 0) :=
  attnSubLayer_dist_le Config.default params 1e-6 (by norm_num) positions
    alpha_max halpha x x 0 (fun _ => by simp) 0

end GPTMini
end Transformer
