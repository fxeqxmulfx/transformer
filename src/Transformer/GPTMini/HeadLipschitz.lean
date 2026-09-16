/-
# How far one whole attention head can move

`AttentionLipschitz` estimates the pieces; this file puts them together into
a single statement about `attentionHead`, the map of `reference/model.py`
(`CausalMHA.forward`) from `(q, k, v)` to the XSA-projected head output.

Given a uniform bound `D` on the movement of `q`, `k` and `v`, and a bound
`B` on the values themselves:

  - RoPE is an isometry, so the roped queries and keys move by at most `D`;
  - `normL2` is `2/eps`-Lipschitz, so the normalized ones move by `2D/eps`;
  - `score_lipschitz` turns that into a score shift `r = e^α · 4D/eps`;
  - `attnOutput_dist_le` gives `2(e^{2r} - 1) B + D` for the output;
  - `xsaProjection_dist_le` doubles that and adds `2B · 2D/eps` for the
    movement of the self-value direction.

The `e^{2r} - 1` is not linear in `D`: a head is locally Lipschitz, with a
constant that degrades as the inputs are allowed to move further apart.  That
is a property of softmax, not an artefact of the estimate.
-/

import Transformer.GPTMini.AttentionLipschitz

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- `attentionHead` with its `let`s expanded: the XSA projection applied to
the attention output of the RoPE-rotated queries and keys, placed at
position `i`. -/
theorem attentionHead_eq
    {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T) :
    attentionHead cfg alpha eps q k v positions i
      = xsaProjection cfg eps v (fun i' => if i' = i then
          attnOutput cfg alpha eps
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i
          else 0) i := rfl

/-- **How far one attention head moves.**

If every query, key and value moves by at most `D`, and the values of the
second stream are bounded by `B`, then with `r = e^α · 4D/eps`

  `‖head - head'‖ ≤ 4 (e^{2r} - 1) B + 2 D + 4 B D / eps`.

Source: `reference/model.py` (`CausalMHA.forward`), through
`GPTMini.attnOutput_dist_le`, `GPTMini.xsaProjection_dist_le`,
`GPTMini.score_lipschitz`, `GPTMini.normL2_lipschitz` and
`GPTMini.applyRope_dist`. -/
theorem attentionHead_dist_le
    {T : ℕ} (alpha eps : ℝ) (heps : 0 < eps)
    (q k v q' k' v' : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i : Fin T) (B D : ℝ)
    (hB : ∀ j, ‖v' j‖ ≤ B)
    (hq : ∀ j, ‖q j - q' j‖ ≤ D) (hk : ∀ j, ‖k j - k' j‖ ≤ D)
    (hv : ∀ j, ‖v j - v' j‖ ≤ D) :
    ‖attentionHead cfg alpha eps q k v positions i
        - attentionHead cfg alpha eps q' k' v' positions i‖
      ≤ 4 * (Real.exp (2 * (Real.exp alpha * (4 / eps * D))) - 1) * B
          + 2 * D + 4 * B * D / eps := by
  classical
  have hD0 : 0 ≤ D := (norm_nonneg _).trans (hq i)
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB i)
  have hrope : ∀ (a a' : Fin T → EucSpace cfg.head_dim) (j : Fin T),
      (∀ j', ‖a j' - a' j'‖ ≤ D) →
      ‖normL2 eps (applyRope cfg.head_dim cfg.rope_theta (positions j) (a j))
        - normL2 eps (applyRope cfg.head_dim cfg.rope_theta (positions j) (a' j))‖
        ≤ 2 / eps * D := by
    intro a a' j ha
    refine (normL2_lipschitz eps heps _ _).trans ?_
    rw [applyRope_dist]
    exact mul_le_mul_of_nonneg_left (ha j) (div_nonneg (by norm_num) heps.le)
  have hclose : ∀ j : Fin T,
      |preScore cfg alpha eps
          (fun j' => applyRope cfg.head_dim cfg.rope_theta (positions j') (q j'))
          (fun j' => applyRope cfg.head_dim cfg.rope_theta (positions j') (k j')) i j
        - preScore cfg alpha eps
          (fun j' => applyRope cfg.head_dim cfg.rope_theta (positions j') (q' j'))
          (fun j' => applyRope cfg.head_dim cfg.rope_theta (positions j') (k' j')) i j|
      ≤ Real.exp alpha * (4 / eps * D) := by
    intro j
    simp only [preScore]
    refine (score_lipschitz alpha eps heps.le _ _ _ _).trans ?_
    refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos alpha).le
    have h1 := hrope q q' i hq
    have h2 := hrope k k' j hk
    have hsplit : (4 : ℝ) / eps * D = 2 / eps * D + 2 / eps * D := by ring
    rw [hsplit]
    linarith
  have hvhat : ‖normL2 eps (v i) - normL2 eps (v' i)‖ ≤ 2 / eps * D :=
    (normL2_lipschitz eps heps _ _).trans
      (mul_le_mul_of_nonneg_left (hv i) (div_nonneg (by norm_num) heps.le))
  have hxsa : ∀ Z Z' : EucSpace cfg.head_dim,
      ‖xsaProjection cfg eps v (fun i' => if i' = i then Z else 0) i
        - xsaProjection cfg eps v' (fun i' => if i' = i then Z' else 0) i‖
        ≤ 2 * ‖Z - Z'‖ + 2 * ‖Z'‖ * ‖normL2 eps (v i) - normL2 eps (v' i)‖ := by
    intro Z Z'
    simpa using xsaProjection_dist_le cfg eps heps.le v (fun i' => if i' = i then Z else 0)
      v' (fun i' => if i' = i then Z' else 0) i
  rw [attentionHead_eq, attentionHead_eq]
  set Y := attnOutput cfg alpha eps
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i with hYdef
  set Y' := attnOutput cfg alpha eps
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q' j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k' j)) v' i with hY'def
  have hYY : ‖Y - Y'‖
      ≤ 2 * (Real.exp (2 * (Real.exp alpha * (4 / eps * D))) - 1) * B + D := by
    rw [hYdef, hY'def]
    exact attnOutput_dist_le cfg alpha eps _ B D _ _ v _ _ v' i hclose hB hv
  have hY'B : ‖Y'‖ ≤ B := by
    rw [hY'def]
    exact attnOutput_norm_le cfg _ _ _ _ _ _ B hB
  have hprod : 2 * ‖Y'‖ * ‖normL2 eps (v i) - normL2 eps (v' i)‖ ≤ 4 * B * D / eps := by
    calc 2 * ‖Y'‖ * ‖normL2 eps (v i) - normL2 eps (v' i)‖
        ≤ 2 * B * (2 / eps * D) :=
          mul_le_mul (by linarith) hvhat (norm_nonneg _) (by linarith)
      _ = 4 * B * D / eps := by field_simp; ring
  exact (hxsa Y Y').trans (by linarith)

/-- The hypotheses are satisfiable: two copies of the same head are `0` apart,
at the `eps = 10⁻⁶` of `reference/model.py`. -/
example (cfg : Config) (alpha : ℝ)
    (q k v : Fin 3 → EucSpace cfg.head_dim) (positions : Fin 3 → ℝ)
    (i : Fin 3) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attentionHead cfg alpha 1e-6 q k v positions i
        - attentionHead cfg alpha 1e-6 q k v positions i‖
      ≤ 4 * (Real.exp (2 * (Real.exp alpha * (4 / 1e-6 * 0))) - 1) * B
          + 2 * 0 + 4 * B * 0 / 1e-6 :=
  attentionHead_dist_le cfg alpha 1e-6 (by norm_num) q k v q k v positions i B 0
    hB (fun _ => by simp) (fun _ => by simp) (fun _ => by simp)

end GPTMini
end Transformer
