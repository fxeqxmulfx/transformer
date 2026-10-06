import Transformer.GPTMini.Semantics.Routing
import Transformer.GPTMini.ReshapeDist
import Transformer.GPTMini.Model

/-!
# Semantic head errors through the actual multi-head residual block

Source: Block.forward and CausalMHA.forward in archived gpt_mini.py at
f11b6e2. headAt extracts an existing head from the original fused QKV
projection after RMSNorm. It neither adds an oracle nor changes attention.

Errors relative to semantic head codes propagate through the actual head
merge and W_o. The residual is retained exactly, and the actual ReLU²
FFN contribution has its own error budget. This separates representation
and routing obligations from the final readout; correct final logits are
not a hypothesis of any result in this module. Semantic codes must be
supplied by the task's independently specified data computation; the
bounds do not assert that training has found such internal states.
-/

namespace Transformer.GPTMini.Semantics

/-- Evaluate one actual head inside attnSubLayer, including its original prenorm and fused QKV.
Source: CausalMHA.forward at f11b6e2, before merging and applying W_o. -/
noncomputable def headAt (cfg : Config) (params : AttnParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) (i : Fin T) : EucSpace cfg.head_dim :=
  let projected := fun j => params.W_qkv (rmsNormEps eps (x j))
  attentionHead cfg (params.log_alpha h) eps
    (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg) (projected j)) h)
    (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg) (projected j)) h)
    (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg) (projected j)) h) positions i

/-- Extracted heads reconstruct the original attention sublayer exactly.
Source: attnSubLayer at f11b6e2; this is an implementation identity, not a semantic assumption. -/
theorem attnSubLayer_eq_heads (cfg : Config) (params : AttnParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    attnSubLayer cfg params eps positions x i =
      params.W_o (headMerge cfg (fun h => headAt cfg params eps positions x h i)) := by
  unfold attnSubLayer headAt
  rfl

/-- Semantic head and FFN approximation errors bound the actual residual block's state error.
Source: the original prenorm block at f11b6e2, including both residual additions. -/
theorem block_error_le (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) (i : Fin T)
    (codes : Fin cfg.n_heads → EucSpace cfg.head_dim) (ffnCode : EucSpace cfg.d_model)
    (headError ffnError : ℝ)
    (hheads : ∀ h, ‖headAt cfg params.attn eps positions x h i - codes h‖ ≤ headError)
    (hffn : ‖ffnSubLayer cfg params.ffn eps
        (fun j => x j + attnSubLayer cfg params.attn eps positions x j) i - ffnCode‖ ≤ ffnError) :
    ‖blockForward cfg params eps positions x i -
      (x i + params.attn.W_o (headMerge cfg codes) + ffnCode)‖ ≤
        ‖params.attn.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * headError) + ffnError := by
  have hmerge := headMerge_dist_le cfg
    (fun h => headAt cfg params.attn eps positions x h i) codes headError hheads
  have hattn : ‖attnSubLayer cfg params.attn eps positions x i -
      params.attn.W_o (headMerge cfg codes)‖ ≤
        ‖params.attn.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * headError) := by
    rw [attnSubLayer_eq_heads, ← params.attn.W_o.map_sub]
    exact (params.attn.W_o.le_opNorm _).trans
      (mul_le_mul_of_nonneg_left hmerge (norm_nonneg _))
  have hsplit : blockForward cfg params eps positions x i -
      (x i + params.attn.W_o (headMerge cfg codes) + ffnCode) =
      (attnSubLayer cfg params.attn eps positions x i - params.attn.W_o (headMerge cfg codes)) +
        (ffnSubLayer cfg params.ffn eps
          (fun j => x j + attnSubLayer cfg params.attn eps positions x j) i - ffnCode) := by
    unfold blockForward
    module
  rw [hsplit]
  exact (norm_add_le _ _).trans (add_le_add hattn hffn)

example (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    (positions : Fin 2 → ℝ) (x : Fin 2 → EucSpace cfg.d_model) :
    (∀ h, ‖headAt cfg params.attn eps positions x h 1 -
      headAt cfg params.attn eps positions x h 1‖ ≤ (0 : ℝ)) ∧
    ‖ffnSubLayer cfg params.ffn eps
        (fun j => x j + attnSubLayer cfg params.attn eps positions x j) 1 -
      ffnSubLayer cfg params.ffn eps
        (fun j => x j + attnSubLayer cfg params.attn eps positions x j) 1‖ ≤ (0 : ℝ) := by
  exact ⟨fun _ => by simp, by simp⟩

/-- Without a separate FFN code, its actual operator norms provide a conservative semantic error budget.
Source: ffnSubLayer_bounded for the original RMSNorm/ReLU² FFN at f11b6e2. -/
theorem block_error_le_bounded_ffn (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    (heps : 0 < eps) {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (i : Fin T) (codes : Fin cfg.n_heads → EucSpace cfg.head_dim) (headError : ℝ)
    (hheads : ∀ h, ‖headAt cfg params.attn eps positions x h i - codes h‖ ≤ headError) :
    ‖blockForward cfg params eps positions x i -
      (x i + params.attn.W_o (headMerge cfg codes))‖ ≤
        ‖params.attn.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * headError) +
          ‖params.ffn.W_out‖ * ‖params.ffn.W_in‖ ^ 2 * (cfg.d_model : ℝ) := by
  have hffn := ffnSubLayer_bounded cfg params.ffn eps heps
    (fun j => x j + attnSubLayer cfg params.attn eps positions x j) i
  have h := block_error_le cfg params eps positions x i codes 0 headError
    (‖params.ffn.W_out‖ * ‖params.ffn.W_in‖ ^ 2 * (cfg.d_model : ℝ)) hheads
    (by simpa only [sub_zero] using hffn)
  simpa only [add_zero] using h

example (cfg : Config) (params : BlockParams cfg) (positions : Fin 2 → ℝ)
    (x : Fin 2 → EucSpace cfg.d_model) : (0 : ℝ) < 1 ∧
    ∀ h, ‖headAt cfg params.attn 1 positions x h 1 -
      headAt cfg params.attn 1 positions x h 1‖ ≤ (0 : ℝ) :=
  ⟨by norm_num, fun _ => by simp⟩

/-- Zero fused QKV produces zero actual head outputs at every position, even with arbitrary RoPE.
Source: the original bias-free QKV projection at f11b6e2; this is the zero-parameter control. -/
theorem headAt_zero_qkv (cfg : Config) (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (h : Fin cfg.n_heads) (i : Fin T) :
    headAt cfg { W_qkv := 0, W_o := 0, log_alpha := fun _ => 0 } eps positions x h i = 0 := by
  have hqkv : ∀ j, (0 : EucSpace cfg.d_model →L[ℝ] EucSpace (3 * cfg.d_model))
      (rmsNormEps eps (x j)) = 0 := by simp
  have hslice : ∀ selector, qkvSlice cfg selector (0 : EucSpace (3 * cfg.d_model)) = 0 := by
    intro selector
    ext j
    simp
  have hhead : ∀ h, headSlice cfg (0 : EucSpace cfg.d_model) h = 0 := by
    intro h
    ext j
    simp
  simp [headAt, hqkv, hslice, hhead, attentionHead, rope_zero, xsaProjection,
    attnOutput, normL2]

/-- Exact local head and FFN codes give the exact semantic residual state.
Source: the zero-error specialization of the actual block transport law at f11b6e2. -/
theorem block_exact_codes (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) (i : Fin T)
    (codes : Fin cfg.n_heads → EucSpace cfg.head_dim) (ffnCode : EucSpace cfg.d_model)
    (hheads : ∀ h, headAt cfg params.attn eps positions x h i = codes h)
    (hffn : ffnSubLayer cfg params.ffn eps
      (fun j => x j + attnSubLayer cfg params.attn eps positions x j) i = ffnCode) :
    blockForward cfg params eps positions x i =
      x i + params.attn.W_o (headMerge cfg codes) + ffnCode := by
  apply sub_eq_zero.mp
  apply norm_eq_zero.mp
  apply le_antisymm _ (norm_nonneg _)
  have h := block_error_le cfg params eps positions x i codes ffnCode 0 0
    (fun h => by rw [hheads h]; simp) (by rw [hffn]; simp)
  simpa only [mul_zero, add_zero] using h

example (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    (positions : Fin 2 → ℝ) (x : Fin 2 → EucSpace cfg.d_model) :
    (∀ h, headAt cfg params.attn eps positions x h 1 = headAt cfg params.attn eps positions x h 1) ∧
    ffnSubLayer cfg params.ffn eps
        (fun j => x j + attnSubLayer cfg params.attn eps positions x j) 1 =
      ffnSubLayer cfg params.ffn eps
        (fun j => x j + attnSubLayer cfg params.attn eps positions x j) 1 :=
  ⟨fun _ => rfl, rfl⟩

end Transformer.GPTMini.Semantics
