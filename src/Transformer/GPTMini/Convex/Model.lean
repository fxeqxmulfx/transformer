/-
# The gpt-mini stack with causal sparsemax heads

New modification of `Block.forward` and `GPTMini.forward` in
`experiments/gpt_mini.py` (restored from commit f11b6e2). It uses the
original trainable parameter records, RMSNorm, head reshapes, ReLU² FFN,
residuals, and tied embeddings. Each attention head uses the solved
quadratic simplex program from `Convex.Attention`. RMSNorm and QK/XSA
have separate epsilon arguments, matching the Python defaults `1e-5`
and `1e-6`, respectively. Inference convexity
does not establish joint training convexity. Motivation:
arXiv:2211.11052v1, §3.1, with depth outside its single-block guarantees.
-/

import Transformer.GPTMini.Convex.Attention
import Transformer.GPTMini.Model

open scoped BigOperators

noncomputable section

namespace Transformer.GPTMini.Convex

/-- The original attention sub-layer with replacement heads. Source:
`Block.forward` and `CausalMHA.forward` in `experiments/gpt_mini.py`. -/
def attnSubLayer (cfg : Config) (params : AttnParams cfg) (rmsEps qkEps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model :=
  let qkv := fun i => params.W_qkv (rmsNormEps rmsEps (x i))
  let q := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvQ cfg) (qkv i)) h
  let k := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvK cfg) (qkv i)) h
  let v := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvV cfg) (qkv i)) h
  fun i => params.W_o (headMerge cfg (fun h =>
    attentionHead cfg (params.log_alpha h) qkEps (q h) (k h) (v h) positions i))

/-- Pre-LN attention and FFN with sequential residual additions. Source:
`Block.forward` in `experiments/gpt_mini.py`; only the head weights change. -/
def blockForward (cfg : Config) (params : BlockParams cfg) (rmsEps qkEps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model :=
  let x1 := fun i => x i + attnSubLayer cfg params.attn rmsEps qkEps positions x i
  fun i => x1 i + GPTMini.ffnSubLayer cfg params.ffn rmsEps x1 i

/-- The residual stream through the replacement blocks, with the original
learned token embeddings. Source: `GPTMini.forward`, `for block in self.blocks`. -/
def hidden (cfg : Config) (params : ModelParams cfg) (rmsEps qkEps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size) :
    ℕ → Fin T → EucSpace cfg.d_model
  | 0 => embed cfg params tokens
  | L + 1 =>
      if h : L < cfg.n_layers then
        blockForward cfg (params.blocks ⟨L, h⟩) rmsEps qkEps positions
          (hidden cfg params rmsEps qkEps positions tokens L)
      else hidden cfg params rmsEps qkEps positions tokens L

/-- Replacement model logits, with the original final RMSNorm and tied
unembedding. Source: `GPTMini.forward` in `experiments/gpt_mini.py`. -/
def forward (cfg : Config) (params : ModelParams cfg) (rmsEps qkEps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) : ℝ :=
  unembed cfg params
    (fun j => rmsNormEps rmsEps
      (hidden cfg params rmsEps qkEps positions tokens cfg.n_layers j)) i v

/-- The same parameter-dependent attention bound as the source block.
The convex row remains a probability vector for arbitrary learned Q/K.
Source: `Block.forward`, Pre-LN attention, in `experiments/gpt_mini.py`. -/
theorem attnSubLayer_bounded (cfg : Config) (params : AttnParams cfg)
    (rmsEps qkEps : ℝ) (hrms : 0 < rmsEps) (hqk : 0 ≤ qkEps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖attnSubLayer cfg params rmsEps qkEps positions x i‖ ≤
      2 * ‖params.W_o‖ * ‖params.W_qkv‖ *
        Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) := by
  set B := ‖params.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ) with hB
  have hvalue : ∀ (h : Fin cfg.n_heads) (j : Fin T),
      ‖headSlice cfg (qkvSlice cfg (qkvV cfg)
        (params.W_qkv (rmsNormEps rmsEps (x j)))) h‖ ≤ B := by
    intro h j
    refine (headSlice_norm_le cfg _ h).trans ?_
    refine (qkvSlice_norm_le cfg _ (qkvV_injective cfg) _).trans ?_
    refine (params.W_qkv.le_opNorm _).trans ?_
    exact mul_le_mul_of_nonneg_left
      (rmsNormEps_norm_le rmsEps hrms cfg.d_model_pos _) (norm_nonneg _)
  have hhead : ∀ h : Fin cfg.n_heads,
      ‖attentionHead cfg (params.log_alpha h) qkEps
        (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
          (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
        (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
          (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
        (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
          (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
        positions i‖ ≤ 2 * B := fun h =>
    attentionHead_norm_le cfg _ qkEps hqk _ _ _ positions i B (hvalue h)
  have hmerge := headMerge_norm_le cfg _ (2 * B) hhead
  calc
    ‖attnSubLayer cfg params rmsEps qkEps positions x i‖ ≤
        ‖params.W_o‖ * ‖headMerge cfg (fun h =>
          attentionHead cfg (params.log_alpha h) qkEps
            (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
              (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
            (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
              (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
            (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
              (params.W_qkv (rmsNormEps rmsEps (x j)))) h)
            positions i)‖ := params.W_o.le_opNorm _
    _ ≤ ‖params.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * (2 * B)) :=
      mul_le_mul_of_nonneg_left hmerge (norm_nonneg _)
    _ = _ := by rw [hB]; ring

/-- A replacement block has the source growth estimate, including the
trainable ReLU² FFN. Source: `Block.forward` in `experiments/gpt_mini.py`. -/
theorem blockForward_growth (cfg : Config) (params : BlockParams cfg)
    (rmsEps qkEps : ℝ) (hrms : 0 < rmsEps) (hqk : 0 ≤ qkEps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖blockForward cfg params rmsEps qkEps positions x i‖ ≤ ‖x i‖ +
      2 * ‖params.attn.W_o‖ * ‖params.attn.W_qkv‖ *
        Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) +
      ‖params.ffn.W_out‖ * ‖params.ffn.W_in‖ ^ 2 * (cfg.d_model : ℝ) := by
  let x1 := fun j => x j + attnSubLayer cfg params.attn rmsEps qkEps positions x j
  have hattn := attnSubLayer_bounded cfg params.attn rmsEps qkEps hrms hqk positions x i
  have hffn := GPTMini.ffnSubLayer_bounded cfg params.ffn rmsEps hrms x1 i
  have hstep := norm_add_le (x i) (attnSubLayer cfg params.attn rmsEps qkEps positions x i)
  have hlast := norm_add_le (x1 i) (GPTMini.ffnSubLayer cfg params.ffn rmsEps x1 i)
  change ‖x1 i + GPTMini.ffnSubLayer cfg params.ffn rmsEps x1 i‖ ≤ _
  change ‖x1 i‖ ≤ _ at hstep
  linarith

/-- The bound premises include the source's RMSNorm epsilon. Source:
`RMSNorm.__init__`, default `eps=1e-5` in `experiments/gpt_mini.py`. -/
example : (0 : ℝ) < 1 / 100000 ∧ (0 : ℝ) ≤ 1 / 1000000 := by norm_num

end Transformer.GPTMini.Convex
