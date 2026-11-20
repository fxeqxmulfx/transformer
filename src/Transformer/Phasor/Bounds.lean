module

public import Transformer.Phasor.Forward

/-!
# The main model's forward pass is bounded

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
*Architecture*, with the abstract blocks of `Transformer.MHC.PeriRMS` (arXiv:2512.24880v2, Eq. (3);
arXiv:2501.19399v1, Eq. (2)). Exact real arithmetic, `ε > 0` at every RMS site, depth `L > 0`, the
depth scale `1/√(2L)`, any parameters: arbitrary real-linear projections, routers, phase
coordinates, frequency scales and score coefficients, any token sequence and any positions.

* `Model.stream_rms_le`, `Model.collapse_rms_le`: after `L` layers every stream and their mean have
  RMS at most `1 + 2√(2L)`, by `Model.states_eq_run` and `scaled_model_rms_bound`;
* `Model.feature_rms_le_one`: the final feature has RMS at most one (`final_feature_rms_le_one`),
  so `Model.logits_abs_le`: a logit has modulus at most `√D ‖E v‖`;
* `Model.head_logit_abs_le`: **every attention logit of every layer, head, position and distance is
  at most `(5/4) ln (i + 1)`** (`Attention.logit_abs_le`), for the query position `i`;
* `Model.bounds`: the four together.

They are bounds of the forward values, not of any gradient, and not floating-point statements.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS

variable {D d H F V : ℕ}

/-- **Every stream has RMS at most `1 + 2√(2L)` after `L` layers**, at every position. -/
theorem Model.stream_rms_le (M : Model D d H F V) (eps : ℝ) (heps : 0 < eps) (hD : 0 < D)
    (tokens : ℕ → Fin V) (L : ℕ) (hL : 0 < L) (i : ℕ) (s : Fin 4) :
    ‖M.states eps (1 / Real.sqrt (2 * (L : ℝ))) tokens L i s‖ / Real.sqrt (D : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (L : ℝ)) := by
  rw [M.states_eq_run]
  exact scaled_model_rms_bound eps heps hD _ _ _ L hL s

/-- **The mean of the streams** has RMS at most `1 + 2√(2L)` too. -/
theorem Model.collapse_rms_le (M : Model D d H F V) (eps : ℝ) (heps : 0 < eps) (hD : 0 < D)
    (tokens : ℕ → Fin V) (L : ℕ) (hL : 0 < L) (i : ℕ) :
    ‖collapse (M.states eps (1 / Real.sqrt (2 * (L : ℝ))) tokens L i)‖ / Real.sqrt (D : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (L : ℝ)) := by
  have hs : 0 < Real.sqrt (D : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast hD)
  rw [div_le_iff₀ hs]
  exact collapse_norm_le _ _ fun s =>
    (div_le_iff₀ hs).1 (M.stream_rms_le eps heps hD tokens L hL i s)

/-- **The final feature has RMS at most one**, for every depth scale. -/
theorem Model.feature_rms_le_one (M : Model D d H F V) (eps scale : ℝ) (heps : 0 < eps)
    (hD : 0 < D) (tokens : ℕ → Fin V) (L i : ℕ) :
    ‖M.feature eps scale tokens L i‖ / Real.sqrt (D : ℝ) ≤ 1 := by
  rw [M.feature_eq_finalFeature]
  exact final_feature_rms_le_one eps scale heps hD _ _ _ L

/-- **A logit is at most `√D ‖E v‖`**: Cauchy–Schwarz and the final feature's norm `≤ √D`. -/
theorem Model.logits_abs_le (M : Model D d H F V) (eps scale : ℝ) (heps : 0 < eps) (hD : 0 < D)
    (tokens : ℕ → Fin V) (L i : ℕ) (v : Fin V) :
    |M.logits eps scale tokens L i v| ≤ ‖M.embedding v‖ * Real.sqrt (D : ℝ) := by
  have hs : 0 < Real.sqrt (D : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast hD)
  have hf := (div_le_one hs).1 (M.feature_rms_le_one eps scale heps hD tokens L i)
  calc |M.logits eps scale tokens L i v|
      ≤ ‖M.embedding v‖ * ‖M.feature eps scale tokens L i‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖M.embedding v‖ * Real.sqrt (D : ℝ) := by gcongr

/-- **Every attention logit of every layer is at most `(5/4) ln (i + 1)`**: the layer `k`, the head
`h`, the query position `i`, any key states and any distance. -/
theorem Model.head_logit_abs_le (M : Model D d H F V) (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (k : ℕ) (h : Fin H) (i : ℕ) (x y : State D) (δ : ℕ) :
    |(M.layer k).attention.logit (M.layer k).attentionRouter eps M.alpha h i x y δ| ≤
      5 / 4 * Real.log ((i : ℝ) + 1) :=
  Attention.logit_abs_le _ _ eps heps hd _ h i x y δ

/-- **The bounds of the forward pass**: streams and their mean at RMS `≤ 1 + 2√(2L)`, the final
feature at RMS `≤ 1`, and every attention logit `≤ (5/4) ln (i + 1)`. -/
theorem Model.bounds (M : Model D d H F V) (eps : ℝ) (heps : 0 < eps) (hD : 0 < D) (hd : 0 < d)
    (tokens : ℕ → Fin V) (L : ℕ) (hL : 0 < L) :
    (∀ i s, ‖M.states eps (1 / Real.sqrt (2 * (L : ℝ))) tokens L i s‖ / Real.sqrt (D : ℝ) ≤
        1 + 2 * Real.sqrt (2 * (L : ℝ))) ∧
      (∀ i, ‖collapse (M.states eps (1 / Real.sqrt (2 * (L : ℝ))) tokens L i)‖ /
        Real.sqrt (D : ℝ) ≤ 1 + 2 * Real.sqrt (2 * (L : ℝ))) ∧
      (∀ i, ‖M.feature eps (1 / Real.sqrt (2 * (L : ℝ))) tokens L i‖ / Real.sqrt (D : ℝ) ≤ 1) ∧
      ∀ k h i x y δ,
        |(M.layer k).attention.logit (M.layer k).attentionRouter eps M.alpha h i x y δ| ≤
          5 / 4 * Real.log ((i : ℝ) + 1) :=
  ⟨fun i s => M.stream_rms_le eps heps hD tokens L hL i s,
    fun i => M.collapse_rms_le eps heps hD tokens L hL i,
    fun i => M.feature_rms_le_one eps _ heps hD tokens L i,
    fun k h i x y δ => M.head_logit_abs_le eps heps hd k h i x y δ⟩

/-- The model with every parameter zero, of DiffGPT's sizes in the examples: zero embedding,
`α = 0`, zero attention and feed-forward projections, routers at their initialization with zero
projections. -/
noncomputable def Model.zero (D d H F V : ℕ) : Model D d H F V where
  embedding := fun _ => 0
  alpha := fun _ => 0
  layer := fun _ =>
    { attention := Attention.zero D d H
      attentionRouter := Router.initial 0 0 0
      ffn := FFN.zero D F
      ffnRouter := Router.initial 0 0 0 }

/-- **DiffGPT's sizes**, `D = 128`, `d = 32`, `H = 4`, vocabulary `97`, `171` hidden units, one
layer, `ε = 2⁻²³`: the bounds hold at every position of any sequence. -/
example (tokens : ℕ → Fin 97) :=
  (Model.zero 128 32 4 171 97).bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    tokens 1 (by norm_num)

end Transformer.Phasor
