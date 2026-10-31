module

public import Transformer.SequenceModel.Attention.Model

/-!
# What the actual normalizers bound

Source: `review/flat_spectrum.md`, *General positional models*, G5, and the
supplied diff_gpt forward at `b98c665b`. Positive numerical epsilon is kept.
Peri output normalization bounds each residual branch independently of its
weight norms. Q/K RMSNorm and standard attention scaling bound scores by
sqrt(head width), rather than an unbounded learned temperature.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini

/-- G5: Q/K RMSNorm, arbitrary rotations and standard 1/sqrt(d) scaling
give a uniform head-score bound, at every finite positive head width. -/
theorem headScore_bound (d : ℕ) (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |headScore d eps a b q k| ≤ Real.sqrt (d : ℝ) := by
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  have hq : ‖rotateHead d a (rmsNormEps eps q)‖ ≤ Real.sqrt (d : ℝ) := by
    rw [rotateHead_isometry]
    exact rmsNormEps_norm_le eps heps hd q
  have hk : ‖rotateHead d b (rmsNormEps eps k)‖ ≤ Real.sqrt (d : ℝ) := by
    rw [rotateHead_isometry]
    exact rmsNormEps_norm_le eps heps hd k
  rw [headScore, abs_div, abs_of_pos hs, div_le_iff₀ hs]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul hq hk (norm_nonneg _) hs.le)

/-- G5: each of the two normalized residual branches adds at most sqrt(W)
to a token's norm, regardless of attention/FFN matrix size or input count. -/
theorem block_norm_le (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    (heps : 0 < eps) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖block cfg B eps xeps angle x i‖ ≤ ‖x i‖ + 2 * Real.sqrt (cfg.d_model : ℝ) := by
  let a := rmsNormEps eps (attention cfg B eps xeps angle (fun j => rmsNormEps eps (x j)) i)
  let f := rmsNormEps eps (feedForward cfg B (rmsNormEps eps (x i + a)))
  have ha : ‖a‖ ≤ Real.sqrt (cfg.d_model : ℝ) := rmsNormEps_norm_le eps heps cfg.d_model_pos _
  have hf : ‖f‖ ≤ Real.sqrt (cfg.d_model : ℝ) := rmsNormEps_norm_le eps heps cfg.d_model_pos _
  change ‖(x i + a) + f‖ ≤ _
  have h1 := norm_add_le (x i + a) f
  have h2 := norm_add_le (x i) a
  linarith

/-- G5: normalized embeddings and both normalized branches give the complete
finite stack bound (1+2L)*sqrt(W), without bounding the learned matrices. -/
theorem hidden_norm_le (cfg : Config) (p : ModelParams cfg) (eps xeps : ℝ)
    (heps : 0 < eps) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (n : ℕ) (i : Fin T) :
    ‖hidden cfg p eps xeps angle tokens n i‖ ≤
      (1 + 2 * (n : ℝ)) * Real.sqrt (cfg.d_model : ℝ) := by
  induction n with
  | zero => simpa only [hidden, Nat.cast_zero, mul_zero, add_zero, one_mul] using
      rmsNormEps_norm_le eps heps cfg.d_model_pos (p.embedding (tokens i))
  | succ n ih =>
      simp only [hidden]
      split_ifs with hn
      · have hb := block_norm_le cfg (p.blocks ⟨n, hn⟩) eps xeps heps (angle ⟨n, hn⟩)
          (hidden cfg p eps xeps angle tokens n) i
        push_cast
        nlinarith
      · have hs := Real.sqrt_nonneg (cfg.d_model : ℝ)
        push_cast
        nlinarith

/-- G5: final RMSNorm bounds each tied score by sqrt(W) times its embedding
row norm. Unlike hidden-state bounds this still depends on the tied readout. -/
theorem score_bound (cfg : Config) (eps xeps : ℝ) (heps : 0 < eps) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (readout : Fin T) (w : Parameters cfg) (tokens : Fin T → Fin cfg.vocab_size)
    (v : Fin cfg.vocab_size) :
    |score cfg eps xeps angle readout w tokens v| ≤
      Real.sqrt (cfg.d_model : ℝ) * ‖(decode cfg w).embedding v‖ := by
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_right (rmsNormEps_norm_le eps heps cfg.d_model_pos _) (norm_nonneg _))

example := headScore_bound 4 (by decide) (1 / 8388608) (by norm_num)
  (fun k => (k.val + 1 : ℝ) / 7) (fun k => (k.val + 3 : ℝ) / 11)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![2, 1, 4, 3])
private def exampleWeights : Parameters witnessConfig := WithLp.toLp 2 (fun _ => 1)
private def exampleAngles : Fin 2 → Fin 5 → Fin 2 → ℝ :=
  sourceAngles witnessConfig (fun _ => true) (fun i => i.val)

example := block_norm_le witnessConfig ((decode witnessConfig exampleWeights).blocks 0)
  (1 / 8388608) (1 / 1000000000000) (by norm_num) (exampleAngles 0)
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) 4
example := hidden_norm_le witnessConfig (decode witnessConfig exampleWeights)
  (1 / 8388608) (1 / 1000000000000) (by norm_num) exampleAngles (fun _ => 1) 2 4
example := score_bound witnessConfig (1 / 8388608) (1 / 1000000000000)
  (by norm_num) exampleAngles 4 exampleWeights (fun _ => 1) 2

end Transformer.SequenceModel.Attention
