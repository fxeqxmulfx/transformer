module

public import Transformer.SequenceModel.Attention.Finite.Packing
public import Transformer.SequenceModel.Attention.Finite.Head
public import Transformer.SequenceModel.Attention.Finite.NormGeometry

/-!
# A lossless actual normalized Transformer context encoder

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F5.
The head result is connected to fused QKV, the attention projection, its
RMSNorm and the residual. Input code injectivity is proved from matrices and
phases, with no hypothesis about the final answer or its logits.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped RealInnerProductSpace

variable {N : ℕ}

/-- F5: the actual attention matrix pipeline computes the selected context
at every operand tuple, with the true softmax denominator. -/
theorem encoder_attention (cfg : Config) (h1 : cfg.n_heads = 1) (hd : 1 < cfg.head_dim)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) (τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) (zero : Fin cfg.vocab_size) (w : EucSpace N)
    (hw : ∀ j, 1 < w j ∧ w j < Real.exp (1 / Real.sqrt (cfg.head_dim : ℝ)))
    (x : Fin N → Fin cfg.vocab_size) :
    attention cfg (encoderBlock cfg h1 eps τ e k) eps xeps (codePhases cfg.head_dim w)
      (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (withReadout zero x j)))))) (Fin.last N) =
      τ • merge cfg h1 ((codeDenominator cfg.head_dim w)⁻¹ • contextCode e zero w x) := by
  have hq (j : Fin (N + 1)) := encoder_qkv cfg h1 heps τ e k (withReadout zero x j)
  simp only [attention, hq, qkvQ_packMaps, qkvK_packMaps, qkvV_packMaps,
    tokenQuery_unit, ContinuousLinearMap.id_apply, headSlice_merge]
  change τ • merge cfg h1 (shift e (slice cfg (merge cfg h1
    (headOutput eps xeps (codePhases cfg.head_dim w) _ _ _ (Fin.last N))))) = _
  rw [slice_merge, shifted_head_code hd heps hx k e zero w hw]

/-- Readout residual plus the actual normalized projected context. -/
def contextState (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (zero : Fin cfg.vocab_size) (w : EucSpace N) (x : Fin N → Fin cfg.vocab_size) : EucSpace cfg.d_model :=
  merge cfg h1 (unit (e (.inl zero))) + rmsNormEps eps (τ • merge cfg h1 (contextCode e zero w x))

/-- The retained readout coordinate supplies the FFN's common positive channel. -/
def readoutChannel (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (zero : Fin cfg.vocab_size) : EucSpace cfg.d_model →L[ℝ] ℝ :=
  innerSL ℝ (merge cfg h1 (unit (e (.inl zero))))

/-- F5: context injection leaves the common readout coordinate exactly one. -/
theorem context_common (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (zero : Fin cfg.vocab_size) (w : EucSpace N) (x : Fin N → Fin cfg.vocab_size) :
    readoutChannel cfg h1 e zero (contextState cfg h1 eps τ e zero w x) = 1 := by
  change ⟪merge cfg h1 (unit (e (.inl zero))),
    merge cfg h1 (unit (e (.inl zero))) + rmsNormEps eps (τ • merge cfg h1 (contextCode e zero w x))⟫ = 1
  rw [inner_add_right, LinearIsometry.inner_map_map, inner_unit]
  simp only [↓reduceIte, rmsNormEps, real_inner_smul_right,
    LinearIsometry.inner_map_map, context_orthogonal, mul_zero, add_zero]

/-- F5: neither output RMSNorm nor the readout residual destroys the
ordered context, at any strictly positive fixed projection scale. -/
theorem contextState_injective (cfg : Config) (h1 : cfg.n_heads = 1)
    {eps τ : ℝ} (heps : 0 < eps) (hτ : 0 < τ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (zero : Fin cfg.vocab_size) (w : EucSpace N) (hi : Function.Injective (contextCode e zero w)) :
    Function.Injective (contextState cfg h1 eps τ e zero w) := by
  intro x y h
  change _ + rmsNormEps eps (τ • merge cfg h1 (contextCode e zero w x)) =
    _ + rmsNormEps eps (τ • merge cfg h1 (contextCode e zero w y)) at h
  have hr := rms_injective cfg.d_model cfg.d_model_pos heps (add_left_cancel h)
  apply hi
  apply (merge cfg h1).injective
  have hc := congrArg (fun v : EucSpace cfg.d_model => τ⁻¹ • v) hr
  simpa only [smul_smul, inv_mul_cancel₀ hτ.ne', one_smul] using hc

end Transformer.SequenceModel.Attention.Finite
