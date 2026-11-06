module

public import Transformer.MHC.PeriRMS.CurrentModel

/-!
# Current-model bounds with square-root attention gain

Requested companion to `CurrentModel.lean`: change only the attention gain
from `1/head_width` to `1/sqrt(head_width)`. Both retain per-head Q/K RMS,
fixed RoPE, SSMax's log(visible keys), centered XSA, SwiGLU, both Peri-RMS
sites, and the same four-stream finite mHC connections. These statements
provide each gain with and without the residual coefficient `1/sqrt(2L)`.

The normalized residual bounds are unchanged because they use output RMS.
The score envelope becomes `sqrt(head_width)*log(N)`. This comparison of
real-arithmetic upper bounds does not compare trained models, gradients,
or finite-precision behavior. Source: arXiv:2512.24880v2, Eqs. (3), (8), (9),
and arXiv:2501.19399v1, Eq. (2), with the local post-RMS depth scaling.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Transformer

/-- Square-root attention gain plus scaled residual additions. The feature
width and attention head width remain distinct, as in the inverse-width model.
Local combination of arXiv:2512.24880v2, Eq. (3), and arXiv:2501.19399v1, Eq. (2). -/
theorem inverse_sqrt_current_model_bounds {width headWidth : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (hlayers : 0 < layers) (n : ℕ)
    (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) :
    (∀ i, ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    (‖collapse (run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers)‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    |Real.log (n + 1) * ((1 / Real.sqrt (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * Real.sqrt (headWidth : ℝ) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := scaled_model_rms_bound eps heps hwidth attention mlp embedding
    layers hlayers
  refine ⟨hstreams, ?_, inverse_sqrt_score_bound eps heps hhead n a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

/-- Square-root attention gain with unit residual coefficients, preserving
every normalization and mHC gate from the scaled variant.
Local combination of arXiv:2512.24880v2, Eq. (3), and arXiv:2501.19399v1, Eq. (2). -/
theorem unscaled_inverse_sqrt_current_model_bounds {width headWidth : ℕ}
    (eps : ℝ) (heps : 0 < eps) (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (n : ℕ) (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) :
    (∀ i, ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    (‖collapse (run eps 1 attention mlp (embeddingState eps embedding) layers)‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    |Real.log (n + 1) * ((1 / Real.sqrt (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * Real.sqrt (headWidth : ℝ) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := unscaled_model_rms_bound eps heps hwidth attention mlp embedding layers
  refine ⟨hstreams, ?_, inverse_sqrt_score_bound eps heps hhead n a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  inverse_sqrt_current_model_bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    attention mlp embedding 6 (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  unscaled_inverse_sqrt_current_model_bounds (1 / 8388608) (by norm_num)
    (by norm_num) (by norm_num) attention mlp embedding 6 63 (fun _ => 2) (fun _ => 3) q k

end Transformer.MHC.PeriRMS
