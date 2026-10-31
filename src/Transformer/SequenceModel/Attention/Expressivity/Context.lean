module

public import Transformer.SequenceModel.Attention.Expressivity.Layout

/-!
# Fixed attention weights generate every ordered-pair code

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E5.
The single head has zero Q/K, identity values after normalization compensation,
and a context-copy output matrix. The source's XSA and both attention branch
normalizers are retained exactly, at arbitrary supplied rotary angles.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini

/-- Common raw embedding amplitude, also used by the tied readout. -/
def amplitude (cfg : Config) (eps : ℝ) : ℝ :=
  Real.sqrt ((cfg.d_model : ℝ) * eps / ((cfg.d_model : ℝ) - 1))

/-- Attention matrices that copy the previous token to a disjoint bank. -/
def contextBlock (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) : BlockParams cfg := {
  qkv := (normScale cfg.d_model eps 1)⁻¹ • packValue cfg
  projection := (2 * amplitude cfg eps) • ((merge cfg h1).toContinuousLinearMap.comp
    ((shift e).comp (slice cfg)))
  gate := 0
  value := 0
  output := 0 }

/-- E5: an embedded token's pre-attention normalizer has the known positive scale. -/
theorem rms_merge_unit (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ) (i : Fin cfg.head_dim) :
    rmsNormEps eps (merge cfg h1 (unit i)) =
      normScale cfg.d_model eps 1 • merge cfg h1 (unit i) := by
  apply rms_of_normSq
  rw [LinearIsometry.norm_map, norm_unit, one_pow]

/-- E5: the fused projection exactly undoes the pre-attention input scale. -/
theorem context_qkv (cfg : Config) (h1 : cfg.n_heads = 1) {eps : ℝ} (heps : 0 < eps)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (a : Fin cfg.vocab_size) :
    (contextBlock cfg h1 eps e).qkv (rmsNormEps eps (merge cfg h1 (unit (e (.inl a))))) =
      packValue cfg (merge cfg h1 (unit (e (.inl a)))) := by
  rw [rms_merge_unit]
  change (normScale cfg.d_model eps 1)⁻¹ • packValue cfg
    (normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl a)))) = _
  rw [map_smul, smul_smul, inv_mul_cancel₀
    (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num)).ne', one_smul]

/-- E5: the actual attention output before its output RMSNorm is the exact
preimage of a unit context code; equal tokens produce zero. -/
theorem context_attention (cfg : Config) (h1 : cfg.n_heads = 1) {eps xeps : ℝ}
    (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (angle : Fin 2 → Fin (cfg.head_dim / 2) → ℝ) (a b : Fin cfg.vocab_size) :
    attention cfg (contextBlock cfg h1 eps e) eps xeps angle
      (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (![a, b] j)))))) 1 =
      if a = b then 0 else amplitude cfg eps • merge cfg h1 (unit (e (.inr a))) := by
  have hq (j : Fin 2) := context_qkv cfg h1 heps e (![a, b] j)
  simp only [attention, hq, qkvQ_pack, qkvK_pack, qkvV_pack, headSlice_merge]
  have hz (h : Fin cfg.n_heads) : headSlice cfg 0 h = 0 := by ext c; rfl
  simp only [hz]
  have hv : (fun j : Fin 2 => unit (e (.inl (![a, b] j)))) =
      ![unit (e (.inl a)), unit (e (.inl b))] := by funext j; fin_cases j <;> rfl
  rw [hv]
  simp only [headOutput_pair eps xeps hx, e.injective.eq_iff, Sum.inl.injEq]
  by_cases hab : a = b
  · simp only [hab, ↓reduceIte]
    have hm : headMerge cfg (fun _ => (0 : EucSpace cfg.head_dim)) = 0 := by ext i; rfl
    rw [hm, map_zero]
  · simp only [hab, ↓reduceIte]
    change (2 * amplitude cfg eps) • merge cfg h1 (shift e
      (slice cfg (merge cfg h1 ((1 / 2 : ℝ) • unit (e (.inl a)))))) = _
    rw [slice_merge, map_smul, shift_unit, map_smul, smul_smul]
    congr 1
    ring

/-- E5: after output RMSNorm and the residual, the real attention branch
produces the ordered-pair code, including equal operands. -/
theorem context_residual (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (angle : Fin 2 → Fin (cfg.head_dim / 2) → ℝ) (a b : Fin cfg.vocab_size) :
    merge cfg h1 (unit (e (.inl b))) + rmsNormEps eps
      (attention cfg (contextBlock cfg h1 eps e) eps xeps angle
        (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (![a, b] j)))))) 1) =
      merge cfg h1 (pairCode e a b) := by
  rw [context_attention cfg h1 heps hx]
  have hi (i : Fin cfg.head_dim) :
      rmsNormEps eps (amplitude cfg eps • merge cfg h1 (unit i)) = merge cfg h1 (unit i) := by
    have hn : ‖merge cfg h1 (unit i)‖ ^ 2 = 1 := by rw [LinearIsometry.norm_map, norm_unit, one_pow]
    have hpre : amplitude cfg eps • merge cfg h1 (unit i) =
        unnormalize cfg.d_model eps (merge cfg h1 (unit i)) := by
      simp only [unnormalize, hn, amplitude]
    rw [hpre]
    exact rms_unnormalize _ cfg.d_model_pos eps heps _ (by rw [hn]; exact_mod_cast (show 1 < cfg.d_model by omega))
  by_cases hab : a = b
  · simp [hab, pairCode, rmsNormEps]
  · simp only [hab, ↓reduceIte, hi, pairCode, map_add]

end Transformer.SequenceModel.Attention.Expressivity
