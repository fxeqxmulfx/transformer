module

public import Transformer.PhasorV2.FFN

/-!
# Phasor v2 tied readout

DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*,
uses `logits_i = E · R_D(mean_s h^L_{i,s})`. Its `model.json` specifies the unweighted stream
mean and uses the learned embedding table directly as the classifier weights.

In Phasor v2, the mean is `streamMean` and the final RMS uses the complex channels. The score
for token `v` is the real part of the complex inner product of `E v` with the normalized
mean. The full forward pass shares the same table with `embed`, which normalizes the selected
row and copies it to all four streams.

The skip preserves the phase-aligned mode `h_s = r_s • v` (`phaseSkip_aligned`). The mean of
that mode is `(∑_s r_s / 4) • v`. The first phase is one and the other three have positive
real parts, so the coefficient has real part at least `1/4`. The mean has norm at least
`‖v‖ / 4` and its final RMS feature is nonzero whenever `v` is nonzero, the width is positive
and epsilon is nonnegative.
This concerns the feature supplied to the classifier. Each logit then depends on its learned
embedding row through the output projection.

The logit bound uses the norm of the original embedding row and the width of the normalized
feature (`abs_readout_le`). Both normalization steps are visible at depth zero
(`forward_zero`), when all four streams hold the same embedded token. At any depth the
logits are causal (`isCausal_forward`).
-/

@[expose] public section

open Transformer.Sandwich

namespace Transformer.PhasorV2

variable {c m H p : ℕ} {V : Type*}

/-- The unweighted mean of the four streams in DiffGPT's *Architecture*. -/
noncomputable def streamMean (h : State c) : CSpace c :=
  ((1 / 4 : ℝ) : ℂ) • ∑ s, h s

/-- The tied output projection of DiffGPT's *Architecture*, using RMS over complex channels.
The learned row `E v` is paired with the normalized mean by the real part of the inner product. -/
noncomputable def readout (eps : ℝ) (E : V → CSpace c) (h : State c) : V → ℝ :=
  let z := rms eps (streamMean h)
  fun v => (inner ℂ (E v) z).re

/-- The full forward pass after `L` attention and feed-forward layers, with the embedding table
`E` shared by `embed` and `readout`. The write scale `γ` is a parameter. -/
noncomputable def forward (eps γ : ℝ) (R : ℕ → Router c) (α : Fin p → ℝ)
    (A : ℕ → Attention c H p) (F : ℕ → FFN c m) (L : ℕ) (E : V → CSpace c)
    (tok : ℕ → V) (i : ℕ) : V → ℝ :=
  readout eps E (stack eps γ R (layerBranch eps α A F) (2 * L) (embed eps E tok) i)

/-- The unweighted mean of phase-aligned streams scales their common vector by the mean
phase. This is the stream mean of DiffGPT's *Architecture* applied to its conserved mode. -/
theorem streamMean_aligned (r : Fin 4 → ℂ) (v : CSpace c) :
    streamMean (fun s => r s • v) = (((1 / 4 : ℝ) : ℂ) * ∑ s, r s) • v := by
  rw [streamMean, ← Finset.sum_smul, smul_smul]

/-- The unweighted mean gives the common vector of four identical streams, as at the
embedding of DiffGPT's *Architecture*. -/
theorem streamMean_const (x : CSpace c) : streamMean (fun _ => x) = x := by
  have hphase : ((1 / 4 : ℝ) : ℂ) * (∑ _ : Fin 4, (1 : ℂ)) = 1 := by
    rw [Fin.sum_univ_four]
    norm_num
  simpa only [one_smul, hphase] using streamMean_aligned (fun _ => 1) x

/-- The final mean retains at least a quarter of the length of the phase-aligned mode in
DiffGPT's *Architecture*. Stream zero provides the real part one in the sum of the phases. -/
theorem Router.norm_streamMean_aligned_ge (R : Router c) (v : CSpace c) :
    ‖v‖ / 4 ≤ ‖streamMean (fun s => R.rot s • v)‖ := by
  have hreal : 1 ≤ (∑ s, R.rot s).re := by
    rw [Complex.re_sum, Fin.sum_univ_four, R.rot_zero, Complex.one_re]
    linarith [R.rot_re_pos 1, R.rot_re_pos 2, R.rot_re_pos 3]
  have hnorm : 1 ≤ ‖∑ s, R.rot s‖ := hreal.trans (Complex.re_le_norm _)
  rw [streamMean_aligned, norm_smul, norm_mul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 4)]
  calc
    ‖v‖ / 4 = (1 / 4 * 1) * ‖v‖ := by ring
    _ ≤ (1 / 4 * ‖∑ s, R.rot s‖) * ‖v‖ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hnorm (by norm_num)) (norm_nonneg _)

/-- A nonzero phase-aligned mode of the DiffGPT skip has a nonzero stream mean. -/
theorem Router.streamMean_aligned_ne_zero (R : Router c) {v : CSpace c} (hv : v ≠ 0) :
    streamMean (fun s => R.rot s • v) ≠ 0 :=
  norm_pos_iff.mp ((div_pos (norm_pos_iff.mpr hv) (by norm_num)).trans_le
    (R.norm_streamMean_aligned_ge v))

/-- A mode on the first channel has a nonzero mean at the model width. -/
example (R : Router 64) :=
  R.streamMean_aligned_ne_zero (v := PiLp.single 2 (0 : Fin 64) (1 : ℂ)) (by simp)

/-- The final RMS of DiffGPT's *Architecture*, over complex channels, keeps the mean of a
nonzero phase-aligned mode nonzero. -/
theorem Router.rms_streamMean_aligned_ne_zero (R : Router c) {eps : ℝ} (heps : 0 ≤ eps)
    (hc : 0 < c) {v : CSpace c} (hv : v ≠ 0) :
    rms eps (streamMean (fun s => R.rot s • v)) ≠ 0 := by
  have hmean := R.streamMean_aligned_ne_zero hv
  have hnorm := norm_pos_iff.mpr hmean
  have hwidth : (0 : ℝ) < c := Nat.cast_pos.mpr hc
  unfold rms
  exact smul_ne_zero (Complex.ofReal_ne_zero.mpr (ne_of_gt (by positivity))) hmean

/-- The first-channel mode survives the router's skip, then the mean and final RMS. -/
example (R : Router 64) (h : State 64) :
    rms (1 / 2 ^ 23) (streamMean (phaseSkip R.rot (R.skip (1 / 2 ^ 23) h)
      (fun s => R.rot s • PiLp.single 2 (0 : Fin 64) (1 : ℂ)))) ≠ 0 := by
  rw [phaseSkip_aligned R.norm_rot (R.skip_row_sum (1 / 2 ^ 23) h)]
  exact R.rms_streamMean_aligned_ne_zero (by norm_num) (by norm_num) (by simp)

/-- A tied logit of DiffGPT's readout has magnitude at most the norm of its embedding row times
`√c`, because the final RMS is over `c` complex channels. -/
theorem abs_readout_le {eps : ℝ} (heps : 0 ≤ eps) (E : V → CSpace c) (h : State c) (v : V) :
    |readout eps E h v| ≤ ‖E v‖ * Real.sqrt c := by
  let z := rms eps (streamMean h)
  calc
    |readout eps E h v| ≤ ‖inner ℂ (E v) z‖ := Complex.abs_re_le_norm _
    _ ≤ ‖E v‖ * ‖z‖ := norm_inner_le_norm (𝕜 := ℂ) _ _
    _ ≤ ‖E v‖ * Real.sqrt c := mul_le_mul_of_nonneg_left (norm_rms_le heps _) (norm_nonneg _)

/-- The tied logits for the main model's width and epsilon. -/
example (E : V → CSpace 64) (h : State 64) (v : V) :=
  abs_readout_le (eps := 1 / 2 ^ 23) (by norm_num) E h v

/-- With an empty stack the mean returns the normalized lookup. The final RMS acts on it
and the tied projection still uses the learned row, as in DiffGPT's *Architecture*. -/
theorem forward_zero (eps γ : ℝ) (R : ℕ → Router c) (α : Fin p → ℝ)
    (A : ℕ → Attention c H p) (F : ℕ → FFN c m) (E : V → CSpace c) (tok : ℕ → V) (i : ℕ) :
    forward eps γ R α A F 0 E tok i =
      fun v => (inner ℂ (E v) (rms eps (rms eps (E (tok i))))).re := by
  unfold forward
  rw [Nat.mul_zero, stack]
  unfold readout embed
  rw [streamMean_const]

/-- The tied logit vector at a position depends only on the token prefix through that position,
since the final mean, RMS and output projection act at a single position. -/
theorem isCausal_forward (eps γ : ℝ) (R : ℕ → Router c) (α : Fin p → ℝ)
    (A : ℕ → Attention c H p) (F : ℕ → FFN c m) (L : ℕ) (E : V → CSpace c) :
    IsCausal (forward eps γ R α A F L E) := by
  intro tok tok' i htok
  exact congrArg (readout eps E) (stack_layers_congr eps γ R α A F (2 * L) E htok)

/-- After four layers, tokens after position five may be replaced without changing its logits. -/
example (E : V → CSpace 64) (R : ℕ → Router 64) (α : Fin 16 → ℝ)
    (A : ℕ → Attention 64 4 16) (F : ℕ → FFN 64 171) (tok : ℕ → V) (v : V) :=
  isCausal_forward (1 / 2 ^ 23) (1 / Real.sqrt 8) R α A F 4 E
    tok (fun j => if j ≤ 5 then tok j else v) 5 (fun j hj => by simp [hj])

end Transformer.PhasorV2
