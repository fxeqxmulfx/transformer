module

public import Transformer.SequenceModel.Attention.Stability.Gates
public import Transformer.SequenceModel.Attention.Stability.Averages

/-!
# Sensitivity of the complete normalized source block

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B4;
diff_gpt `b98c665b`, `block.py`, `mha.py`, `feed_forward.py`. Both residuals,
all four branch normalizers, moving XSA directions and actual SwiGLU are
retained. Constants are sufficient bounds, independent of context length.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini

/-- Amplification by the pre/post-normalized FFN residual. -/
def ffnGain (cfg : Config) (B : BlockParams cfg) (eps : ℝ) : ℝ :=
  1 + rmsLip eps * ffnLip cfg B * rmsLip eps

/-- Head merge, learned attention projection and post-normalization. -/
def attentionGain (cfg : Config) (B : BlockParams cfg) (eps : ℝ) : ℝ :=
  rmsLip eps * ‖B.projection‖ * Real.sqrt (cfg.n_heads : ℝ)

/-- Sensitivity of XSA's normalized own-value direction. -/
def directionGain (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) : ℝ :=
  2 * valueBound cfg B * (2 / xeps * componentLip cfg B eps)

/-- B4: the actual normalized SwiGLU residual is globally Lipschitz.
Its input normalization supplies the bounded ball required by B2. -/
theorem ffnResidual_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ}
    (heps : 0 < eps) (x y : EucSpace cfg.d_model) :
    ‖(x + rmsNormEps eps (feedForward cfg B (rmsNormEps eps x))) -
      (y + rmsNormEps eps (feedForward cfg B (rmsNormEps eps y)))‖ ≤
        ffnGain cfg B eps * ‖x - y‖ := by
  have hL : 0 ≤ rmsLip eps := by unfold rmsLip; positivity
  have hF : 0 ≤ ffnLip cfg B := by unfold ffnLip; positivity
  have hn := rmsNormEps_lipschitz eps heps cfg.d_model_pos x y
  have hf := feedForward_error cfg B (rmsNormEps eps x) (rmsNormEps eps y)
    (rmsNormEps_norm_le eps heps cfg.d_model_pos x)
    (rmsNormEps_norm_le eps heps cfg.d_model_pos y)
  have ho := rmsNormEps_lipschitz eps heps cfg.d_model_pos
    (feedForward cfg B (rmsNormEps eps x)) (feedForward cfg B (rmsNormEps eps y))
  have h := ho.trans (mul_le_mul_of_nonneg_left
    (hf.trans (mul_le_mul_of_nonneg_left hn hF)) hL)
  rw [add_sub_add_comm]
  exact (norm_add_le _ _).trans ((add_le_add le_rfl h).trans_eq
    (by unfold ffnGain rmsLip; ring))

/-- B4: joint error in the residual and the softmax averages, including
the change of XSA's own-value direction and the first-position zero rule. -/
theorem localUpdate_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (first : Bool)
    (x y : EucSpace cfg.d_model) (u v : Fin cfg.n_heads → EucSpace cfg.head_dim)
    (E A : ℝ) (hx : ‖x - y‖ ≤ E) (hu : ∀ h, ‖u h - v h‖ ≤ A)
    (hv : ∀ h, ‖v h‖ ≤ valueBound cfg B) :
    ‖Normalized.localUpdate cfg B eps xeps first x u -
      Normalized.localUpdate cfg B eps xeps first y v‖ ≤
      ffnGain cfg B eps * (E + attentionGain cfg B eps *
        (2 * A + directionGain cfg B eps xeps * E)) := by
  have hE : 0 ≤ E := (norm_nonneg _).trans hx
  have hA : 0 ≤ A := (norm_nonneg _).trans (hu ⟨0, cfg.n_heads_pos⟩)
  have hL : 0 ≤ rmsLip eps := by unfold rmsLip; positivity
  have hC : 0 ≤ componentLip cfg B eps := by unfold componentLip; positivity
  have hV : 0 ≤ valueBound cfg B := by unfold valueBound; positivity
  have hX : 0 ≤ directionGain cfg B eps xeps := by unfold directionGain; positivity
  let heads (z : EucSpace cfg.d_model) (avg : Fin cfg.n_heads → EucSpace cfg.head_dim) := fun h =>
    let dir := normL2 xeps (Normalized.component cfg B eps (qkvV cfg) z h)
    if first then 0 else avg h - inner ℝ (avg h) dir • dir
  have hh h : ‖heads x u h - heads y v h‖ ≤ 2 * A + directionGain cfg B eps xeps * E := by
    dsimp only [heads]
    split
    · simpa only [sub_self, norm_zero] using
        add_nonneg (mul_nonneg (by norm_num) hA) (mul_nonneg hX hE)
    · have hd := (normL2_lipschitz xeps hxeps
          (Normalized.component cfg B eps (qkvV cfg) x h)
          (Normalized.component cfg B eps (qkvV cfg) y h)).trans
          (mul_le_mul_of_nonneg_left
            ((component_error cfg B heps _ (qkvV_injective cfg) x y h).trans
              (mul_le_mul_of_nonneg_left hx hC)) (by positivity))
      have hp := GPTMini.norm_proj_sub_proj_le (u h) (v h)
        (normL2 xeps (Normalized.component cfg B eps (qkvV cfg) x h))
        (normL2 xeps (Normalized.component cfg B eps (qkvV cfg) y h))
        (normL2_norm_le xeps hxeps.le _) (normL2_norm_le xeps hxeps.le _)
      refine hp.trans ?_
      have ht := mul_le_mul (mul_le_mul_of_nonneg_left (hv h) (by norm_num)) hd
        (norm_nonneg _) (by positivity : 0 ≤ 2 * valueBound cfg B)
      have ha := mul_le_mul_of_nonneg_left (hu h) (by norm_num : (0 : ℝ) ≤ 2)
      exact (add_le_add ha ht).trans_eq (by unfold directionGain; ring)
  have hm := headMerge_dist_le cfg (heads x u) (heads y v) _ hh
  have hp : ‖B.projection (headMerge cfg (heads x u)) -
      B.projection (headMerge cfg (heads y v))‖ ≤
      ‖B.projection‖ * (Real.sqrt (cfg.n_heads : ℝ) *
        (2 * A + directionGain cfg B eps xeps * E)) := by
    rw [← map_sub]
    exact (B.projection.le_opNorm _).trans (mul_le_mul_of_nonneg_left hm (norm_nonneg _))
  have hn := (rmsNormEps_lipschitz eps heps cfg.d_model_pos
      (B.projection (headMerge cfg (heads x u)))
      (B.projection (headMerge cfg (heads y v)))).trans (mul_le_mul_of_nonneg_left hp hL)
  have hr : ‖(x + rmsNormEps eps (B.projection (headMerge cfg (heads x u)))) -
      (y + rmsNormEps eps (B.projection (headMerge cfg (heads y v))))‖ ≤
      E + attentionGain cfg B eps * (2 * A + directionGain cfg B eps xeps * E) := by
    rw [add_sub_add_comm]
    exact (norm_add_le _ _).trans ((add_le_add hx hn).trans_eq
      (by unfold attentionGain rmsLip; ring))
  exact (ffnResidual_error cfg B heps _ _).trans
    (mul_le_mul_of_nonneg_left hr (by unfold ffnGain ffnLip; positivity))

/-- Explicit sensitivity of one full source block to its input stream. -/
def blockLip (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) : ℝ :=
  ffnGain cfg B eps * (1 + attentionGain cfg B eps *
    (2 * averageLip cfg B eps + directionGain cfg B eps xeps))

/-- B4: a global bound for the actual source block, using the same arbitrary
RoPE phases on each side. No bound on earlier residual magnitudes is assumed. -/
theorem block_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x y : Fin T → EucSpace cfg.d_model) (E : ℝ) (hE : ∀ j, ‖x j - y j‖ ≤ E) (i : Fin T) :
    ‖block cfg B eps xeps a x i - block cfg B eps xeps a y i‖ ≤ blockLip cfg B eps xeps * E := by
  rw [← Normalized.local_sourceAverage, ← Normalized.local_sourceAverage]
  exact (localUpdate_error cfg B heps hxeps _ (x i) (y i) _ _ E (averageLip cfg B eps * E)
    (hE i) (sourceAverage_error cfg B heps a x y E hE i)
    (sourceAverage_norm cfg B heps a y i)).trans_eq (by unfold blockLip; ring)

example (B : BlockParams witnessConfig) := ffnResidual_error witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (WithLp.toLp 2 ![1, 2, 3, 4]) 0
example (B : BlockParams witnessConfig) := localUpdate_error witnessConfig B
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  false 0 0 (fun _ => 0) (fun _ => 0) 0 0 (by simp) (by intro h; simp)
  (by intro h; simp only [norm_zero]; unfold valueBound; positivity)
example (B : BlockParams witnessConfig) := block_error witnessConfig B
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun i : Fin 3 => fun _ => (i.val : ℝ))
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  0 (by intro j; simp) 2

end Transformer.SequenceModel.Attention.Stability
