module

public import Transformer.SequenceModel.Attention.Regularity.Head

/-!
# Every trainable matrix participates in the regularity proof

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, R1.
The source QKV/reshape/XSA/projection path and both Peri-RMSNorm branches
are differentiated jointly in all scalar parameters and their incoming state.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural

variable (cfg : Config) {w₀ : Parameters cfg} {d : Fin cfg.n_layers}

/-- R1: the fused QKV map is analytic in every weight and input coordinate. -/
theorem analyticAt_qkv {v : Parameters cfg → EucSpace cfg.d_model}
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) w₀) (i : Fin (3 * cfg.d_model)) :
    AnalyticAt ℝ (fun w => (decode cfg w).blocks d |>.qkv (v w) i) w₀ := by
  exact analyticAt_matrixApply (fun a b => analyticAt_parameter (I := Index cfg) (.inr (d, .inl (a, b))) w₀) hv i

/-- R1: all three SwiGLU matrices vary, including the output projection. -/
theorem analyticAt_feedForward {v : Parameters cfg → EucSpace cfg.d_model}
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) w₀) (i : Fin cfg.d_model) :
    AnalyticAt ℝ (fun w => feedForward cfg ((decode cfg w).blocks d) (v w) i) w₀ := by
  have hg := analyticAt_matrixApply
    (fun a b => analyticAt_parameter (I := Index cfg) (.inr (d, .inr (.inr (.inl (a, b))))) w₀) hv
  have hu := analyticAt_matrixApply
    (fun a b => analyticAt_parameter (I := Index cfg) (.inr (d, .inr (.inr (.inr (.inl (a, b)))))) w₀) hv
  exact analyticAt_matrixApply
    (fun a b => analyticAt_parameter (I := Index cfg) (.inr (d, .inr (.inr (.inr (.inr (a, b)))))) w₀)
    (fun j => analyticAt_swiglu_comp (hg j) (hu j)) i

/-- The exact value supplied to XSA before its clipped own-value normalization. -/
def layerValue (cfg : Config) (d : Fin cfg.n_layers) (w : Parameters cfg)
    (v : EucSpace cfg.d_model) (h : Fin cfg.n_heads) : EucSpace cfg.head_dim :=
  headSlice cfg (qkvSlice cfg (qkvV cfg) ((decode cfg w).blocks d |>.qkv v)) h

/-- R1: the whole attention branch is analytic at weights avoiding the own-value
clipping sphere. The hypothesis concerns actual activations at these weights. -/
theorem analyticAt_attention {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    {v : Parameters cfg → Fin T → EucSpace cfg.d_model}
    (hv : ∀ j i, AnalyticAt ℝ (fun w => v w j i) w₀) (j : Fin T)
    (hboundary : ∀ h, ‖layerValue cfg d w₀ (v w₀ j) h‖ ≠ xeps) (i : Fin cfg.d_model) :
    AnalyticAt ℝ (fun w => attention cfg ((decode cfg w).blocks d)
      eps xeps angle (v w) j i) w₀ := by
  let qkv := fun w t => ((decode cfg w).blocks d).qkv (v w t)
  let q := fun w h t => headSlice cfg (qkvSlice cfg (qkvQ cfg) (qkv w t)) h
  let k := fun w h t => headSlice cfg (qkvSlice cfg (qkvK cfg) (qkv w t)) h
  let u := fun w h t => headSlice cfg (qkvSlice cfg (qkvV cfg) (qkv w t)) h
  let out := fun w h => headOutput eps xeps angle (q w h) (k w h) (u w h) j
  have hqkv : ∀ t c, AnalyticAt ℝ (fun w => qkv w t c) w₀ :=
    fun t => analyticAt_qkv cfg (d := d) (hv t)
  have ho : ∀ h c, AnalyticAt ℝ (fun w => out w h c) w₀ := by
    intro h c
    exact analyticAt_headOutput cfg.head_dim_pos heps hxeps angle
      (q := fun w => q w h) (k := fun w => k w h) (v := fun w => u w h) (x := w₀)
      (fun t c => hqkv t (qkvQ cfg ((headSplit cfg).symm (h, c))))
      (fun t c => hqkv t (qkvK cfg ((headSplit cfg).symm (h, c))))
      (fun t c => hqkv t (qkvV cfg ((headSplit cfg).symm (h, c))))
      j (show ‖u w₀ h j‖ ≠ xeps from hboundary h) c
  have hm : ∀ c, AnalyticAt ℝ (fun w => headMerge cfg (out w) c) w₀ :=
    fun c => ho (headSplit cfg c).1 (headSplit cfg c).2
  change AnalyticAt ℝ (fun w => matrixEquiv cfg.d_model cfg.d_model
    (fun a b => w.ofLp (.inr (d, .inr (.inl (a, b))))) (headMerge cfg (out w)) i) w₀
  exact analyticAt_matrixApply
    (fun a b => analyticAt_parameter (I := Index cfg) (.inr (d, .inr (.inl (a, b)))) w₀) hm i

/-- R1: both normalized branches and the sequential residual additions are
analytic jointly in all source weights and an analytic incoming stream. -/
theorem analyticAt_block {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    {v : Parameters cfg → Fin T → EucSpace cfg.d_model}
    (hv : ∀ j i, AnalyticAt ℝ (fun w => v w j i) w₀)
    (hboundary : ∀ j h, ‖layerValue cfg d w₀ (rmsNormEps eps (v w₀ j)) h‖ ≠ xeps)
    (j : Fin T) (i : Fin cfg.d_model) :
    AnalyticAt ℝ (fun w => block cfg ((decode cfg w).blocks d)
      eps xeps angle (v w) j i) w₀ := by
  let v1 := fun w j => v w j + rmsNormEps eps
    (attention cfg ((decode cfg w).blocks d) eps xeps angle
      (fun k => rmsNormEps eps (v w k)) j)
  have ha := fun j => analyticAt_attention cfg heps hxeps angle
    (fun k => analyticAt_rmsNormEps cfg.d_model_pos heps (hv k)) j (hboundary j)
  have h1 : ∀ j i, AnalyticAt ℝ (fun w => v1 w j i) w₀ :=
    fun j i => (hv j i).fun_add (analyticAt_rmsNormEps cfg.d_model_pos heps (ha j) i)
  have hf := analyticAt_feedForward cfg (d := d)
    (analyticAt_rmsNormEps cfg.d_model_pos heps (h1 j))
  exact (h1 j i).fun_add (analyticAt_rmsNormEps cfg.d_model_pos heps hf i)

/-- Zero QKV weights give zero values even when the input is nonzero. This
supplies genuine positive-width/depth regularity witnesses. -/
theorem layerValue_zero (d : Fin cfg.n_layers) (v : EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) : layerValue cfg d 0 v h = 0 := by
  ext c
  change matrixEquiv _ _ (0 : Matrix (Fin (3 * cfg.d_model)) (Fin cfg.d_model) ℝ)
    v (qkvV cfg ((headSplit cfg).symm (h, c))) = 0
  simp

example := analyticAt_qkv witnessConfig (d := 0) (w₀ := 0)
  (v := fun w => (decode witnessConfig w).embedding 1)
  (fun i => analyticAt_parameter (I := Index witnessConfig) (.inl (1, i)) 0) 0
example := analyticAt_feedForward witnessConfig (d := 0) (w₀ := 0)
  (v := fun w => (decode witnessConfig w).embedding 1)
  (fun i => analyticAt_parameter (I := Index witnessConfig) (.inl (1, i)) 0) 0
example := analyticAt_attention witnessConfig (d := 0) (w₀ := 0)
  (eps := 1) (xeps := 1) (by norm_num) (by norm_num)
  (fun _ : Fin 3 => fun _ => 1 / 7)
  (v := fun w _ => (decode witnessConfig w).embedding 1)
  (fun _ i => analyticAt_parameter (I := Index witnessConfig) (.inl (1, i)) 0) 2
  (by intro h; rw [layerValue_zero]; simp) 0
example := analyticAt_block witnessConfig (d := 0) (w₀ := 0)
  (eps := 1) (xeps := 1) (by norm_num) (by norm_num)
  (fun _ : Fin 3 => fun _ => 1 / 7)
  (v := fun w _ => (decode witnessConfig w).embedding 1)
  (fun _ i => analyticAt_parameter (I := Index witnessConfig) (.inl (1, i)) 0)
  (by intro j h; rw [layerValue_zero]; simp) 2 0

end Transformer.SequenceModel.Attention
