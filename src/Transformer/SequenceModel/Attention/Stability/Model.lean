module

public import Transformer.SequenceModel.Attention.Stability.Continuity

/-!
# Continuity of the complete normalized source in RoPE frequencies

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A3;
`diff_gpt/domain/model/{gpt,block,mha,feed_forward}.py` at `b98c665b`.
All weights are fixed but arbitrary. Phases vary through the entire residual
stack; Q/K/V in later blocks are not held fixed in this result.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini Landscape.Neural

variable {P : Type*} [TopologicalSpace P]

private theorem continuous_slice {n m : ℕ} (f : Fin m → Fin n)
    (x : P → EucSpace n) (hx : Continuous x) :
    Continuous (fun w => WithLp.toLp 2 (fun i => x w (f i))) := by
  fun_prop

/-- A3: the actual fused QKV, head layout, XSA and output projection are
jointly continuous in the input stream and all positional phases. -/
theorem continuous_attention (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : P → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : P → Fin T → EucSpace cfg.d_model)
    (ha : ∀ j c, Continuous (fun w => a w j c))
    (hx : ∀ j, Continuous (fun w => x w j)) (i : Fin T) :
    Continuous (fun w => attention cfg B eps xeps (a w) (x w) i) := by
  have hc (part : Fin cfg.d_model → Fin (3 * cfg.d_model)) (h : Fin cfg.n_heads)
      (j : Fin T) : Continuous (fun w =>
        headSlice cfg (qkvSlice cfg part (B.qkv (x w j))) h) :=
    continuous_slice _ _ (continuous_slice _ _ (B.qkv.continuous.comp (hx j)))
  have hh (h : Fin cfg.n_heads) := continuous_headOutput cfg.head_dim_pos heps hxeps a
    (fun w j => headSlice cfg (qkvSlice cfg (qkvQ cfg) (B.qkv (x w j))) h)
    (fun w j => headSlice cfg (qkvSlice cfg (qkvK cfg) (B.qkv (x w j))) h)
    (fun w j => headSlice cfg (qkvSlice cfg (qkvV cfg) (B.qkv (x w j))) h)
    ha (hc _ h) (hc _ h) (hc _ h) i
  apply B.projection.continuous.comp
  apply (PiLp.continuous_toLp 2 _).comp
  apply continuous_pi
  intro c
  exact (PiLp.continuous_apply (p := 2) (β := fun _ : Fin cfg.head_dim => ℝ)
    (headSplit cfg c).2).comp (hh (headSplit cfg c).1)

/-- A3: continuity of the source SwiGLU, with separate learned gate/value
matrices and a positive sigmoid denominator at every finite real input. -/
theorem continuous_feedForward (cfg : Config) (B : BlockParams cfg) :
    Continuous (feedForward cfg B) := by
  apply B.output.continuous.comp
  apply (PiLp.continuous_toLp 2 _).comp
  apply continuous_pi
  intro c
  have hg : Continuous (fun x => B.gate x c) := by fun_prop
  have hv : Continuous (fun x => B.value x c) := by fun_prop
  exact (hg.div (continuous_const.add hg.neg.rexp)
    (fun x => (show (0 : ℝ) < 1 + Real.exp (-(B.gate x c)) by positivity).ne')).mul hv

/-- A3: all four branch normalizers and both residual additions remain in
the block when its phases and input residuals vary. -/
theorem continuous_block (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : P → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : P → Fin T → EucSpace cfg.d_model)
    (ha : ∀ j c, Continuous (fun w => a w j c))
    (hx : ∀ j, Continuous (fun w => x w j)) (i : Fin T) :
    Continuous (fun w => block cfg B eps xeps (a w) (x w) i) := by
  have hn := continuous_rms cfg.d_model_pos heps
  have hatt := continuous_attention cfg B heps hxeps a
    (fun w j => rmsNormEps eps (x w j)) ha (fun j => hn.comp (hx j)) i
  have hx1 := (hx i).add (hn.comp hatt)
  exact hx1.add (hn.comp ((continuous_feedForward cfg B).comp (hn.comp hx1)))

/-- A3: continuity of every finite source stack in its entire phase table.
Depth may be arbitrary and every subsequent residual is recomputed. -/
theorem continuous_hidden (cfg : Config) (p : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : P → Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (ha : ∀ l j c, Continuous (fun w => a w l j c))
    (tokens : Fin T → Fin cfg.vocab_size) (n : ℕ) (i : Fin T) :
    Continuous (fun w => hidden cfg p eps xeps (a w) tokens n i) := by
  induction n generalizing i with
  | zero => exact continuous_const
  | succ n ih =>
      simp only [hidden]
      split_ifs with hn
      · exact continuous_block cfg (p.blocks ⟨n, hn⟩) heps hxeps _ _ (ha ⟨n, hn⟩) ih i
      · exact ih i

/-- A3: the final RMSNorm and tied raw embedding readout are continuous
in the phases, with arbitrary fixed source weights and positive epsilons. -/
theorem continuous_score (cfg : Config) (w : Parameters cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : P → Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (ha : ∀ l j c, Continuous (fun f => a f l j c))
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) (v : Fin cfg.vocab_size) :
    Continuous (fun f => score cfg eps xeps (a f) i w tokens v) :=
  ((continuous_rms cfg.d_model_pos heps).comp
    (continuous_hidden cfg (decode cfg w) heps hxeps a ha tokens cfg.n_layers i)).inner
      continuous_const

private def examplePhases (t : ℝ) : Fin 2 → Fin 3 → Fin 2 → ℝ :=
  fun l j c => t * (l.val + j.val + c.val)

example := continuous_slice (fun c : Fin 2 => c.castAdd 2)
  (fun t : ℝ => WithLp.toLp 2 (fun _ : Fin 4 => t)) (by fun_prop)
example := continuous_attention witnessConfig ((decode witnessConfig 0).blocks 0)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun t => examplePhases t 0) (fun t : ℝ => fun _ : Fin 3 => WithLp.toLp 2 (fun _ => t))
  (by intro j c; unfold examplePhases; fun_prop) (by intro j; fun_prop) 2
example := continuous_block witnessConfig ((decode witnessConfig 0).blocks 0)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun t => examplePhases t 0) (fun t : ℝ => fun _ : Fin 3 => WithLp.toLp 2 (fun _ => t))
  (by intro j c; unfold examplePhases; fun_prop) (by intro j; fun_prop) 2
example := continuous_hidden witnessConfig (decode witnessConfig 0)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  examplePhases (by intro l j c; unfold examplePhases; fun_prop) (fun _ => 1) 2 2
example := continuous_score witnessConfig 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  examplePhases (by intro l j c; unfold examplePhases; fun_prop) (fun _ => 1) 2 1

end Transformer.SequenceModel.Attention.Stability
