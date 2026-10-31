module

public import Transformer.SequenceModel.Attention.Finite.Block
public import Transformer.SequenceModel.Attention.Expressivity.Model

/-!
# Complete finite-arity classification with tied embeddings and finite depth

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F6.
The constructed active block is installed in the actual source residual
stack. Additional blocks are identity residuals. The actual embedding and
final normalizers and the tied raw readout yield a positive strict margin.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped RealInnerProductSpace

/-- One active rotary block, identity residual padding, and the source tied embedding. -/
def sequenceModel (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (C : BlockParams cfg) : ModelParams cfg where
  embedding := taskEmbedding cfg h1 eps e
  blocks d := if d.val = 0 then C else (decode cfg 0).blocks d

/-- F6: the proved active block output survives the full finite-depth
source stack. The block premise is supplied by `exists_sequence_block`. -/
theorem sequence_hidden (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) {N : ℕ}
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (zero : Fin cfg.vocab_size)
    (angle : Fin (N + 1) → Fin (cfg.head_dim / 2) → ℝ) (C : BlockParams cfg)
    (f : (Fin N → Fin cfg.vocab_size) → Fin cfg.vocab_size)
    (hC : ∀ x, block cfg C eps xeps angle
      (fun j => merge cfg h1 (unit (e (.inl (withReadout zero x j))))) (Fin.last N) =
      merge cfg h1 (unit (e (.inl (f x))))) (x : Fin N → Fin cfg.vocab_size) (n : ℕ) :
    hidden cfg (sequenceModel cfg h1 eps e C) eps xeps (fun _ => angle)
      (withReadout zero x) (n + 1) (Fin.last N) = merge cfg h1 (unit (e (.inl (f x)))) := by
  induction n with
  | zero =>
    simp only [hidden, cfg.n_layers_pos, ↓reduceDIte]
    change block cfg C eps xeps angle
      (fun j => rmsNormEps eps (taskEmbedding cfg h1 eps e (withReadout zero x j))) (Fin.last N) = _
    simp only [embedding_normalize cfg h1 hw heps]
    exact hC x
  | succ n ih =>
    rw [hidden]
    split_ifs with hn
    · have hz : n + 1 ≠ 0 := by omega
      simp only [sequenceModel, hz, ↓reduceIte, block_zero]
      exact ih
    · exact ih

/-- F6: every finite ordered table is a strict classifier of the complete
source-shaped model whenever the explicit finite capacity bounds hold.
The theorem provides fixed weights and active rotary phases, not assumptions
about trained logits, a rank factorization, or an idealized normalization. -/
theorem exists_classifier (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ N ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (zero : Fin cfg.vocab_size)
    (f : (Fin N → Fin cfg.vocab_size) → Fin cfg.vocab_size) :
    ∃ angle : Fin (N + 1) → Fin (cfg.head_dim / 2) → ℝ,
      ∃ weights : Parameters cfg, ∀ x q, q ≠ f x →
        score cfg eps xeps (fun _ => angle) (Fin.last N) weights (withReadout zero x) q <
        score cfg eps xeps (fun _ => angle) (Fin.last N) weights (withReadout zero x) (f x) := by
  obtain ⟨angle, C, hC⟩ := exists_sequence_block cfg h1 hw N hwidth heps hx e zero f
  refine ⟨angle, encode cfg (sequenceModel cfg h1 eps e C), ?_⟩
  intro x q hq
  have hhidden := sequence_hidden cfg h1 (by omega) heps e zero angle C f hC x (cfg.n_layers - 1)
  have hn : cfg.n_layers - 1 + 1 = cfg.n_layers := by have := cfg.n_layers_pos; omega
  rw [hn] at hhidden
  have hscore (v : Fin cfg.vocab_size) :
      score cfg eps xeps (fun _ => angle) (Fin.last N) (encode cfg (sequenceModel cfg h1 eps e C))
        (withReadout zero x) v =
      (normScale cfg.d_model eps 1 * amplitude cfg eps) * (if v = f x then 1 else 0) := by
    rw [score, decode_encode, hhidden, rms_merge_unit]
    change ⟪normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl (f x)))),
      amplitude cfg eps • merge cfg h1 (unit (e (.inl v)))⟫ = _
    rw [real_inner_smul_left, real_inner_smul_right, LinearIsometry.inner_map_map, inner_unit]
    simp only [e.injective.eq_iff, Sum.inl.injEq]
    by_cases hv : v = f x
    · simp [hv]
    · simp [hv, Ne.symm hv]
  rw [hscore, hscore]
  simp only [hq, ↓reduceIte, mul_zero, mul_one]
  exact mul_pos (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num))
    (amplitude_pos cfg (by omega) heps)

end Transformer.SequenceModel.Attention.Finite
