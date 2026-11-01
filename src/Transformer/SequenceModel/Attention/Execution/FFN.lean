module

public import Transformer.SequenceModel.Attention.Execution.Budget

/-!
# Error through the actual SwiGLU execution branch

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C5;
diff_gpt `b98c665b`, `feed_forward.py` and `block.py`. The proof starts from
each operation's own recorded inputs. Approximate input normalization,
independent gate/value matrices, SiLU, multiplication, output projection
and output normalization all contribute to the bound.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability Landscape.Neural

/-- C5: the complete computed FFN branch error follows from its individual
operation contracts, including a normalized input that has itself moved. -/
theorem ffn_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    {T : ℕ} (x : Fin T → EucSpace cfg.d_model) (t : FFNTrace cfg T) (δ : ℝ) (hδ : 0 ≤ δ)
    (ht : FFNAccurate cfg B eps x t δ) (i : Fin T) :
    ‖t.output i - rmsNormEps eps (feedForward cfg B (rmsNormEps eps (x i)))‖ ≤
      (ffnBudget cfg B eps δ).output := by
  rcases ht with ⟨hpre, hgate, hvalue, hact, hprod, hproj, hout⟩
  let y := rmsNormEps eps (x i)
  let Eg := ‖B.gate‖ * δ + δ
  let Ev := ‖B.value‖ * δ + δ
  let Rg := ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ)
  let Rv := ‖B.value‖ * Real.sqrt (cfg.d_model : ℝ) + Ev
  let z := WithLp.toLp 2 (fun c => swiglu (B.gate y c) (B.value y c))
  have hg : ‖t.gate i - B.gate y‖ ≤ Eg := linear_error B.gate y (t.pre i) (t.gate i) δ δ (hpre i) (hgate i)
  have hv : ‖t.value i - B.value y‖ ≤ Ev := linear_error B.value y (t.pre i) (t.value i) δ δ (hpre i) (hvalue i)
  have hRg : ‖B.gate y‖ ≤ Rg := (B.gate.le_opNorm y).trans
    (mul_le_mul_of_nonneg_left (rmsNormEps_norm_le eps heps cfg.d_model_pos _) (norm_nonneg _))
  have hRv : ‖t.value i‖ ≤ Rv := norm_of_error _ _ Ev _ hv ((B.value.le_opNorm y).trans
    (mul_le_mul_of_nonneg_left (rmsNormEps_norm_le eps heps cfg.d_model_pos _) (norm_nonneg _)))
  have hRv0 : 0 ≤ Rv := (norm_nonneg _).trans hRv
  have hc c : |t.product i c - z c| ≤ δ + δ * Rv + Rv * (1 + Rg) * Eg + Rg * Ev := by
    have hpc : |t.product i c - t.activated i c * t.value i c| ≤ δ :=
      (PiLp.norm_apply_le _ c).trans (hprod i)
    have hac : |t.activated i c - silu (t.gate i c)| ≤ δ :=
      (PiLp.norm_apply_le _ c).trans (hact i)
    have hav : |t.activated i c * t.value i c - swiglu (t.gate i c) (t.value i c)| ≤ δ * Rv := by
      unfold swiglu
      rw [← sub_mul, abs_mul]
      exact mul_le_mul hac ((PiLp.norm_apply_le _ c).trans hRv) (abs_nonneg _) hδ
    have hsw := swiglu_error (t.gate i c) (t.value i c) (B.gate y c) (B.value y c) Rg Rv Eg Ev
      ((PiLp.norm_apply_le _ c).trans hRg) ((PiLp.norm_apply_le _ c).trans hRv)
      ((PiLp.norm_apply_le _ c).trans hg) ((PiLp.norm_apply_le _ c).trans hv)
    exact (abs_sub_le _ (t.activated i c * t.value i c) _).trans
      ((add_le_add hpc ((abs_sub_le _ (swiglu (t.gate i c) (t.value i c)) _).trans
        (add_le_add hav hsw))).trans_eq (by ring))
  have hp : ‖t.product i - z‖ ≤ (ffnBudget cfg B eps δ).product := by
    apply norm_le_coordinates _ _ (by positivity)
    exact hc
  have ho : ‖t.projected i - B.output z‖ ≤ (ffnBudget cfg B eps δ).projected :=
    linear_error B.output z (t.product i) (t.projected i) _ δ hp (hproj i)
  exact rms_error cfg.d_model_pos heps (B.output z) (t.projected i) (t.output i) _ δ ho (hout i)

/-- Exact real evaluation recorded at every source FFN operation. -/
def exactFFN (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) : FFNTrace cfg T :=
  let pre := fun i => rmsNormEps eps (x i)
  let gate := fun i => B.gate (pre i)
  let value := fun i => B.value (pre i)
  let activated := fun i => WithLp.toLp 2 (fun c => silu (gate i c))
  let product := fun i => WithLp.toLp 2 (fun c => activated i c * value i c)
  let projected := fun i => B.output (product i)
  ⟨pre, gate, value, activated, product, projected, fun i => rmsNormEps eps (projected i)⟩

/-- C5: exact execution supplies all operation premises, for arbitrary real weights. -/
theorem exactFFN_accurate (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) : FFNAccurate cfg B eps x (exactFFN cfg B eps x) 0 := by
  simp [FFNAccurate, exactFFN]

example (B : BlockParams witnessConfig) := ffn_error witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (fun i : Fin 3 => WithLp.toLp 2 ![i.val, 1, 2, 3])
  (exactFFN witnessConfig B (1 / 8388608) (fun i : Fin 3 => WithLp.toLp 2 ![i.val, 1, 2, 3]))
  0 (by norm_num) (exactFFN_accurate _ _ _ _) 2

end Transformer.SequenceModel.Attention.Execution
