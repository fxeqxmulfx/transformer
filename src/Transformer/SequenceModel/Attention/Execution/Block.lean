module

public import Transformer.SequenceModel.Attention.Execution.Attention
public import Transformer.SequenceModel.Attention.Execution.FFN

/-!
# Both executed residual branches

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C6;
diff_gpt `b98c665b`, `block.py`. The FFN sees the actually recorded first
residual, not the ideal one. Its movement is propagated with the proved
source FFN-residual constant, alongside new operation errors.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- C6: every operation in a full block contributes to an explicit radius;
the input stream is held fixed in this local execution comparison. -/
theorem block_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (t : BlockTrace cfg T) (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : BlockAccurate cfg B eps xeps a x t δ τ) (i : Fin T) :
    ‖t.output i - block cfg B eps xeps a x i‖ ≤ blockRadius cfg B eps xeps δ τ := by
  rcases ht with ⟨ha, hm, hf, ho⟩
  let y := x i + rmsNormEps eps (attention cfg B eps xeps a (fun j => rmsNormEps eps (x j)) i)
  let z := t.mid i + rmsNormEps eps (feedForward cfg B (rmsNormEps eps (t.mid i)))
  have hav := attention_error cfg B heps hxeps a x t.attn δ τ hδ hτ ha i
  have hmid : ‖t.mid i - y‖ ≤ (attentionBudget cfg B eps xeps δ τ).output + δ := by
    have heq : x i + t.attn.output i - y = t.attn.output i -
        rmsNormEps eps (attention cfg B eps xeps a (fun j => rmsNormEps eps (x j)) i) := by
      dsimp [y]
      abel
    have hb : ‖x i + t.attn.output i - y‖ ≤ (attentionBudget cfg B eps xeps δ τ).output := by
      rw [heq]
      exact hav
    exact (norm_sub_le_norm_sub_add_norm_sub (t.mid i) (x i + t.attn.output i) y).trans
      ((add_le_add (hm i) hb).trans_eq (by ring))
  have hff := ffn_error cfg B heps t.mid t.ffn δ hδ hf i
  have hout : ‖t.output i - z‖ ≤ (ffnBudget cfg B eps δ).output + δ := by
    have heq : t.mid i + t.ffn.output i - z = t.ffn.output i -
        rmsNormEps eps (feedForward cfg B (rmsNormEps eps (t.mid i))) := by
      dsimp [z]
      abel
    have hb : ‖t.mid i + t.ffn.output i - z‖ ≤ (ffnBudget cfg B eps δ).output := by
      rw [heq]
      exact hff
    exact (norm_sub_le_norm_sub_add_norm_sub (t.output i) (t.mid i + t.ffn.output i) z).trans
      ((add_le_add (ho i) hb).trans_eq (by ring))
  have href : ‖z - block cfg B eps xeps a x i‖ ≤
      ffnGain cfg B eps * ((attentionBudget cfg B eps xeps δ τ).output + δ) :=
    (ffnResidual_error cfg B heps (t.mid i) y).trans
      (mul_le_mul_of_nonneg_left hmid (by unfold ffnGain ffnLip rmsLip; positivity))
  exact (norm_sub_le_norm_sub_add_norm_sub (t.output i) z _).trans
    ((add_le_add hout href).trans_eq (by unfold blockRadius; ring))

/-- C6: new execution error adds to amplification of an already perturbed
residual stream; the block Lipschitz factor is derived from the actual source. -/
theorem block_movement_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x y : Fin T → EucSpace cfg.d_model)
    (t : BlockTrace cfg T) (δ τ E : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : BlockAccurate cfg B eps xeps a x t δ τ) (hE : ∀ j, ‖x j - y j‖ ≤ E) (i : Fin T) :
    ‖t.output i - block cfg B eps xeps a y i‖ ≤
      blockLip cfg B eps xeps * E + blockRadius cfg B eps xeps δ τ :=
  (norm_sub_le_norm_sub_add_norm_sub (t.output i) (block cfg B eps xeps a x i) _).trans
    ((add_le_add (block_error cfg B heps hxeps a x t δ τ hδ hτ ht i)
      (Stability.block_error cfg B heps hxeps a x y E hE i)).trans_eq (by ring))

/-- Record the source's sequential attention and FFN residual additions. -/
def exactBlock (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) : BlockTrace cfg T :=
  let attn := exactAttention cfg B eps xeps a x
  let mid := fun i => x i + attn.output i
  let ffn := exactFFN cfg B eps mid
  ⟨attn, mid, ffn, fun i => mid i + ffn.output i⟩

/-- Structural source correspondence: recording the real operations leaves the block unchanged. -/
theorem exactBlock_output (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    (exactBlock cfg B eps xeps a x).output = block cfg B eps xeps a x := rfl

/-- C6: the actual source furnishes consistent operation checks for a whole block. -/
theorem exactBlock_accurate (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    BlockAccurate cfg B eps xeps a x (exactBlock cfg B eps xeps a x) 0 0 := by
  exact ⟨exactAttention_accurate _ _ _ _ _ _, by intro i; simp [exactBlock],
    exactFFN_accurate _ _ _ _, by intro i; simp [exactBlock]⟩

example (B : BlockParams witnessConfig) := block_error witnessConfig B
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  (exactBlock witnessConfig B (1 / 8388608) (1 / 1000000000000)
    (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]))
  0 0 (by norm_num) (by norm_num) (exactBlock_accurate _ _ _ _ _ _) 2

example (B : BlockParams witnessConfig) := block_movement_error witnessConfig B
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  (exactBlock witnessConfig B (1 / 8388608) (1 / 1000000000000)
    (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]))
  0 0 0 (by norm_num) (by norm_num) (exactBlock_accurate _ _ _ _ _ _)
  (by intro j; simp) 2

end Transformer.SequenceModel.Attention.Execution
