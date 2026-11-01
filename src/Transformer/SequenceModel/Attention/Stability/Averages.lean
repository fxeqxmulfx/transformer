module

public import Transformer.SequenceModel.Attention.Stability.Vector

/-!
# Source softmax averages when residual states change

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B3;
diff_gpt `b98c665b`, `mha.py`. Unlike the earlier phase-only estimate, both
the attention distribution and all projected values vary. The same arbitrary
exact RoPE phases are used on both sides.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini
open scoped BigOperators

/-- B3: causal attention is a convex average of its actual value vectors. -/
theorem average_norm {T d : ℕ} (s : Fin T → Fin T → ℝ) (v : Fin T → EucSpace d)
    (i : Fin T) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖∑ j, causalWeight s i j • v j‖ ≤ B := by
  calc
    _ ≤ ∑ j, ‖causalWeight s i j • v j‖ := norm_sum_le _ _
    _ ≤ ∑ j, causalWeight s i j * B := by
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (weight_nonneg s i j)]
      exact mul_le_mul_of_nonneg_left (hB j) (weight_nonneg s i j)
    _ = B := by rw [← Finset.sum_mul, causalWeight_sum, one_mul]

/-- B3: movement of values adds to movement of the attention distribution. -/
theorem average_joint_error {T d : ℕ} (s t : Fin T → Fin T → ℝ)
    (v v' : Fin T → EucSpace d) (i : Fin T) (r B D : ℝ)
    (hs : ∀ j ≤ i, |s i j - t i j| ≤ r)
    (hv : ∀ j, ‖v j - v' j‖ ≤ D) (hB : ∀ j, ‖v' j‖ ≤ B) :
    ‖(∑ j, causalWeight s i j • v j) - ∑ j, causalWeight t i j • v' j‖ ≤ 8 * r * B + D := by
  have hval : ‖(∑ j, causalWeight s i j • v j) - ∑ j, causalWeight s i j • v' j‖ ≤ D := by
    rw [← Finset.sum_sub_distrib]
    simp only [← smul_sub]
    exact average_norm s (fun j => v j - v' j) i D hv
  exact (norm_sub_le_norm_sub_add_norm_sub _ (∑ j, causalWeight s i j • v' j) _).trans
    (by linarith [average_error s t v' i r B hs hB])

/-- B3: for unchanged exact phases, RMS-normalized source head logits
have input error at most 2*rmsLip(eps)*D when Q/K each change by at most D. -/
theorem headScore_input_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (a b : Fin (d / 2) → ℝ) (q k q' k' : EucSpace d) (D : ℝ)
    (hq : ‖q - q'‖ ≤ D) (hk : ‖k - k'‖ ≤ D) :
    |headScore d eps a b q k - headScore d eps a b q' k'| ≤ 2 * rmsLip eps * D := by
  let u := rotateHead d a (rmsNormEps eps q)
  let v := rotateHead d b (rmsNormEps eps k)
  let u' := rotateHead d a (rmsNormEps eps q')
  let v' := rotateHead d b (rmsNormEps eps k')
  have hL : 0 ≤ rmsLip eps := by unfold rmsLip; positivity
  have hu : ‖u - u'‖ ≤ rmsLip eps * D := by
    rw [rotation_dist]
    exact (rmsNormEps_lipschitz eps heps hd _ _).trans (mul_le_mul_of_nonneg_left hq hL)
  have hv : ‖v - v'‖ ≤ rmsLip eps * D := by
    rw [rotation_dist]
    exact (rmsNormEps_lipschitz eps heps hd _ _).trans (mul_le_mul_of_nonneg_left hk hL)
  have hvnorm : ‖v‖ ≤ Real.sqrt (d : ℝ) := by
    rw [rotateHead_isometry]; exact rmsNormEps_norm_le eps heps hd k
  have hunorm : ‖u'‖ ≤ Real.sqrt (d : ℝ) := by
    rw [rotateHead_isometry]; exact rmsNormEps_norm_le eps heps hd q'
  have heq : inner ℝ u v - inner ℝ u' v' = inner ℝ (u - u') v + inner ℝ u' (v - v') := by
    rw [inner_sub_left, inner_sub_right]; ring
  have hab : |inner ℝ u v - inner ℝ u' v'| ≤
      (2 * rmsLip eps * D) * Real.sqrt (d : ℝ) := by
    rw [heq]
    refine (abs_add_le _ _).trans ((add_le_add (abs_real_inner_le_norm _ _)
      (abs_real_inner_le_norm _ _)).trans ?_)
    have hD : 0 ≤ D := (norm_nonneg _).trans hq
    have h1 := mul_le_mul hu hvnorm (norm_nonneg _) (mul_nonneg hL hD)
    have h2 := mul_le_mul hunorm hv (norm_nonneg _) (Real.sqrt_nonneg _)
    nlinarith
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  unfold headScore
  rw [← sub_div, abs_div, abs_of_pos hs, div_le_iff₀ hs]
  exact hab

/-- Sensitivity of one actual source softmax average to the residual stream. -/
def averageLip (cfg : Config) (B : BlockParams cfg) (eps : ℝ) : ℝ :=
  (16 * rmsLip eps * valueBound cfg B + 1) * componentLip cfg B eps

/-- B3: values in a source average have a uniform bound from pre-RMSNorm. -/
theorem sourceAverage_norm (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    {T : ℕ} (a : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) (h : Fin cfg.n_heads) :
    ‖Normalized.sourceAverage cfg B eps a x i h‖ ≤ valueBound cfg B :=
  average_norm _ _ i _ (fun j => component_norm cfg B heps _ (qkvV_injective cfg) (x j) h)

/-- B3: the source average changes with both its RMS-normalized logits
and its unnormalized values. The bound is independent of context length. -/
theorem sourceAverage_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    {T : ℕ} (a : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x y : Fin T → EucSpace cfg.d_model) (E : ℝ) (hE : ∀ j, ‖x j - y j‖ ≤ E)
    (i : Fin T) (h : Fin cfg.n_heads) :
    ‖Normalized.sourceAverage cfg B eps a x i h - Normalized.sourceAverage cfg B eps a y i h‖ ≤
      averageLip cfg B eps * E := by
  have hC : 0 ≤ componentLip cfg B eps := by unfold componentLip rmsLip; positivity
  have hc part hp j :
      ‖Normalized.component cfg B eps part (x j) h - Normalized.component cfg B eps part (y j) h‖ ≤
        componentLip cfg B eps * E :=
    (component_error cfg B heps part hp _ _ h).trans (mul_le_mul_of_nonneg_left (hE j) hC)
  have hs j (hj : j ≤ i) := headScore_input_error cfg.head_dim_pos heps (a i) (a j) _ _ _ _
    (componentLip cfg B eps * E) (hc (qkvQ cfg) (qkvQ_injective cfg) i)
    (hc (qkvK cfg) (qkvK_injective cfg) j)
  exact (average_joint_error _ _ _ _ i _ _ _ hs (hc (qkvV cfg) (qkvV_injective cfg))
    (fun j => component_norm cfg B heps _ (qkvV_injective cfg) (y j) h)).trans_eq
      (by unfold averageLip; ring)

example := average_norm (fun _ _ : Fin 3 => (0 : ℝ))
  (fun _ => WithLp.toLp 2 ![1, 0]) 2 1
  (by intro j; norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_succ])
example := average_joint_error (fun _ _ : Fin 3 => (0 : ℝ)) (fun _ _ => 1 / 10)
  (fun _ => (0 : EucSpace 2)) (fun _ => 0) 2 (1 / 10) 1 0
  (by intro j hj; norm_num) (by intro j; simp) (by intro j; simp)
example := headScore_input_error (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
  (fun _ => 1 / 7) (fun _ => 2 / 7) 0 0 0 0 0 (by simp) (by simp)
example (B : BlockParams witnessConfig) := sourceAverage_norm witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (fun i : Fin 3 => fun _ => (i.val : ℝ))
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) 2 0
example (B : BlockParams witnessConfig) := sourceAverage_error witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (fun i : Fin 3 => fun _ => (i.val : ℝ))
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  0 (by intro j; simp) 2 0

end Transformer.SequenceModel.Attention.Stability
