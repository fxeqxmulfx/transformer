module

public import Transformer.SequenceModel.Attention.Finite.Encoder

/-!
# One full normalized rotary block computes any finite-arity table

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F6.
A fixed appended readout token allows nonzero rotary Q/K to encode operand
order. The actual finite SwiGLU branch then decodes the context. All matrices
and phases are shared across the entire input table, with every source
normalizer retained. Width is sufficient, not asserted necessary.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity

/-- F6: a single complete source block realizes every finite ordered task,
with three genuine hidden channels per input tuple and source normalization.
The phase table is fixed by position; it is constructed, not assumed to
provide an injective code. No desired logits occur among the hypotheses. -/
theorem exists_sequence_block (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ N ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (zero : Fin cfg.vocab_size) (f : (Fin N → Fin cfg.vocab_size) → Fin cfg.vocab_size) :
    ∃ angle : Fin (N + 1) → Fin (cfg.head_dim / 2) → ℝ,
      ∃ C : BlockParams cfg, ∀ x : Fin N → Fin cfg.vocab_size,
        block cfg C eps xeps angle
          (fun j => merge cfg h1 (unit (e (.inl (withReadout zero x j))))) (Fin.last N) =
          merge cfg h1 (unit (e (.inl (f x)))) := by
  let : NeZero cfg.vocab_size := ⟨by have := cfg.vocab_pos; omega⟩
  have hd : 1 < cfg.head_dim := by simpa only [Config.head_dim, h1, Nat.div_one] using (show 1 < cfg.d_model by omega)
  have hs : 0 < Real.sqrt (cfg.head_dim : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast (show 0 < cfg.head_dim by omega))
  obtain ⟨w, hbound, hcode⟩ := exists_bounded_context e zero N
    (lo := 1) (hi := Real.exp (1 / Real.sqrt (cfg.head_dim : ℝ)))
    (Real.one_lt_exp_iff.mpr (div_pos (by norm_num) hs))
  let k : Fin (cfg.head_dim / 2) := ⟨0, by omega⟩
  obtain ⟨τ, hτ, hsmall⟩ := exists_small_normalized_context cfg.d_model cfg.d_model_pos heps
    (fun x : Fin N → Fin cfg.vocab_size => merge cfg h1 (contextCode e zero w x))
  let D := codeDenominator cfg.head_dim w
  have hD : D ≠ 0 := (codeDenominator_pos cfg.head_dim w (fun j => by linarith [(hbound j).1])).ne'
  let B := encoderBlock cfg h1 eps (τ * D) e k
  let state := contextState cfg h1 eps τ e zero w
  let u := readoutChannel cfg h1 e zero
  let target := fun x => merge cfg h1 (unit (e (.inl (f x))))
  have hu (x) : u (state x) = 1 := context_common cfg h1 eps τ e zero w x
  have hi : Function.Injective (fun x => (u (state x))⁻¹ • state x) := by
    simp only [hu, inv_one, one_smul]
    exact contextState_injective cfg h1 heps hτ e zero w hcode
  have hr (x) : ‖target x - state x‖ ^ 2 < (cfg.d_model : ℝ) := by
    apply reachable_unit_answer hw
    · rw [LinearIsometry.norm_map, norm_unit]
    · rw [LinearIsometry.norm_map, norm_unit]
    · exact hsmall x
  have hH : 3 * Fintype.card (Fin N → Fin cfg.vocab_size) ≤ cfg.d_ff := by
    simpa only [Fintype.card_fun, Fintype.card_fin] using hwidth
  obtain ⟨C, hQ, hP, hff⟩ := exists_normalized_ffn cfg hH B heps state target u
    (fun x => by rw [hu]; norm_num) hi hr
  refine ⟨codePhases cfg.head_dim w, C, ?_⟩
  intro x
  have ha : attention cfg C eps xeps (codePhases cfg.head_dim w)
      (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (withReadout zero x j)))))) (Fin.last N) =
      τ • merge cfg h1 (contextCode e zero w x) := by
    have he : attention cfg C eps xeps (codePhases cfg.head_dim w)
        (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (withReadout zero x j)))))) (Fin.last N) =
        attention cfg B eps xeps (codePhases cfg.head_dim w)
        (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (withReadout zero x j)))))) (Fin.last N) := by
      simp only [attention, hQ, hP]
    rw [he, encoder_attention cfg h1 hd heps hx (τ * D) e k zero w hbound]
    change (τ * D) • merge cfg h1 (D⁻¹ • contextCode e zero w x) = _
    rw [map_smul, smul_smul, mul_assoc, mul_inv_cancel₀ hD, mul_one]
  dsimp only [block]
  rw [ha]
  simpa only [state, contextState, target, withReadout, Fin.lastCases_last] using hff x

end Transformer.SequenceModel.Attention.Finite
