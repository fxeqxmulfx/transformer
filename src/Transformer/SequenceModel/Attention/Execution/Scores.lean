module

public import Transformer.SequenceModel.Attention.Execution.Budget

/-!
# Source attention scores from recorded operations

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C4;
diff_gpt `b98c665b`, `mha.py` and `rope.py`. Input RMS, fused QKV, head RMS,
stored tables, table application and scaled dot products are all included.
The cross term from perturbing both query and key is retained.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini CRASP Stability

/-- C4: exact slicing and reshape propagate the fused matrix error without amplification. -/
theorem trace_component_error (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) (t : AttentionTrace cfg T) (δ : ℝ)
    (hpre : ∀ i, ‖t.pre i - rmsNormEps eps (x i)‖ ≤ δ)
    (hfused : ∀ i, ‖t.fused i - B.qkv (t.pre i)‖ ≤ δ)
    (part : Fin cfg.d_model → Fin (3 * cfg.d_model)) (hpart : Function.Injective part)
    (i : Fin T) (h : Fin cfg.n_heads) :
    ‖t.component part i h - Normalized.component cfg B eps part (x i) h‖ ≤ ‖B.qkv‖ * δ + δ :=
  (headSlice_dist_le cfg _ _ h).trans ((qkvSlice_dist_le cfg part hpart _ _).trans
    (linear_error B.qkv (rmsNormEps eps (x i)) (t.pre i) (t.fused i) δ δ (hpre i) (hfused i)))

/-- C4: table perturbation, movement of the normalized input and arithmetic
applying the table contribute separately to the rotated query/key error. -/
theorem rotary_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (a c s : Fin (d / 2) → ℝ) (x z y : EucSpace d) (E δ τ : ℝ) (hτ : 0 ≤ τ)
    (hz : ‖z - rmsNormEps eps x‖ ≤ E)
    (hy : ‖y - tableRotate d c s z‖ ≤ δ)
    (hc : ∀ k, |c k - Real.cos (a k)| ≤ τ) (hs : ∀ k, |s k - Real.sin (a k)| ≤ τ) :
    ‖y - rotateHead d a (rmsNormEps eps x)‖ ≤ (2 * τ) * (Real.sqrt (d : ℝ) + E) + E + δ := by
  have hzn := norm_of_error z (rmsNormEps eps x) E _ hz (rmsNormEps_norm_le eps heps hd x)
  have ht := (tableRotate_error d c s a z τ hτ hc hs).trans
    (mul_le_mul_of_nonneg_left hzn (by positivity))
  have hm : ‖rotateHead d a z - rotateHead d a (rmsNormEps eps x)‖ ≤ E := by
    rw [rotation_dist]
    exact hz
  exact (norm_sub_le_norm_sub_add_norm_sub y (tableRotate d c s z) _).trans
    ((add_le_add hy ((norm_sub_le_norm_sub_add_norm_sub _ (rotateHead d a z) _).trans
      (add_le_add ht hm))).trans_eq (by ring))

/-- C4: the actual source scaled score, compared with its recorded computation.
The sufficient bound includes errors in both Q/K vectors and their interaction. -/
theorem attention_scores_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    (xeps : ℝ) {T : ℕ} (a : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (t : AttentionTrace cfg T)
    (δ τ : ℝ) (hτ : 0 ≤ τ) (ht : AttentionAccurate cfg B eps xeps a x t δ τ)
    (h : Fin cfg.n_heads) (i j : Fin T) :
    |t.scores h i j - headScore cfg.head_dim eps (a i) (a j)
      (Normalized.component cfg B eps (qkvQ cfg) (x i) h)
      (Normalized.component cfg B eps (qkvK cfg) (x j) h)| ≤
        (attentionBudget cfg B eps xeps δ τ).score := by
  rcases ht with ⟨hpre, hfused, hqn, hkn, hc, hs, hq, hk, hscore, hrest⟩
  let b := attentionBudget cfg B eps xeps δ τ
  let q₀ := Normalized.component cfg B eps (qkvQ cfg) (x i) h
  let k₀ := Normalized.component cfg B eps (qkvK cfg) (x j) h
  let qr := rotateHead cfg.head_dim (a i) (rmsNormEps eps q₀)
  let kr := rotateHead cfg.head_dim (a j) (rmsNormEps eps k₀)
  have hqe : ‖t.qnorm h i - rmsNormEps eps q₀‖ ≤ b.qk :=
    rms_error cfg.head_dim_pos heps q₀ (t.component (qkvQ cfg) i h) (t.qnorm h i) _ δ
      (trace_component_error cfg B eps x t δ hpre hfused _ (qkvQ_injective cfg) i h) (hqn h i)
  have hke : ‖t.knorm h j - rmsNormEps eps k₀‖ ≤ b.qk :=
    rms_error cfg.head_dim_pos heps k₀ (t.component (qkvK cfg) j h) (t.knorm h j) _ δ
      (trace_component_error cfg B eps x t δ hpre hfused _ (qkvK_injective cfg) j h) (hkn h j)
  have hqr : ‖t.query h i - qr‖ ≤ b.rotary := rotary_error cfg.head_dim_pos heps
    (a i) (t.cosTable i) (t.sinTable i) q₀ (t.qnorm h i) (t.query h i) b.qk δ τ hτ
    hqe (hq h i) (hc i) (hs i)
  have hkr : ‖t.key h j - kr‖ ≤ b.rotary := rotary_error cfg.head_dim_pos heps
    (a j) (t.cosTable j) (t.sinTable j) k₀ (t.knorm h j) (t.key h j) b.qk δ τ hτ
    hke (hk h j) (hc j) (hs j)
  have hqrn : ‖qr‖ ≤ Real.sqrt (cfg.head_dim : ℝ) := by
    rw [rotateHead_isometry]
    exact rmsNormEps_norm_le eps heps cfg.head_dim_pos q₀
  have hkrn : ‖kr‖ ≤ Real.sqrt (cfg.head_dim : ℝ) := by
    rw [rotateHead_isometry]
    exact rmsNormEps_norm_le eps heps cfg.head_dim_pos k₀
  have hi := inner_pair_error (t.query h i) (t.key h j) qr kr b.rotary
    (Real.sqrt (cfg.head_dim : ℝ)) hqr hkr hqrn hkrn
  have hscale : |inner ℝ (t.query h i) (t.key h j) / Real.sqrt (cfg.head_dim : ℝ) -
      headScore cfg.head_dim eps (a i) (a j) q₀ k₀| ≤
      (2 * Real.sqrt (cfg.head_dim : ℝ) + b.rotary) * b.rotary / Real.sqrt (cfg.head_dim : ℝ) := by
    rw [headScore, ← sub_div, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
    exact div_le_div_of_nonneg_right hi (Real.sqrt_nonneg _)
  exact (abs_sub_le _ (inner ℝ (t.query h i) (t.key h j) / Real.sqrt (cfg.head_dim : ℝ)) _).trans
    ((add_le_add (hscore h i j) hscale).trans_eq (by dsimp [b, attentionBudget]; ring))

example := rotary_error (d := 2) (by decide) (eps := 1) (by norm_num)
  (fun _ => 0) (fun _ => 1) (fun _ => 0) 0 0 0 0 0 0 (by norm_num)
  (by simp [rmsNormEps]) (by rw [tableRotate_zero]; simp)
  (by intro k; norm_num) (by intro k; norm_num)

end Transformer.SequenceModel.Attention.Execution
