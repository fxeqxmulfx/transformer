module

public import Transformer.MHC.PeriRMS.Dynamics
public import Transformer.MHC.PeriRMS.PhaseScoreHead
public import Transformer.MHC.PeriRMS.Scores

/-!
# Bounds specialized to the main model

The main model of DiffGPT (`~/python_projects/gpt`, `fe1d528`, `research/main_model/`, arm
`phase-score-lerope`) is a complex Peri-RMS transformer: embedding RMS, per-head Q/K RMS, LeRoPE
angles, SSMax with the gain `ln N / head_width` on the score `Re Z + β (Im Z)² / head_width` of the
Hermitian product `Z` of query and key, ordinary values, the plain complex-product feed-forward
network, input/output Peri-RMS, the depth coefficient `1/√(2L)` after the branch output RMS, final
RMS, and a tied embedding/readout matrix. Its parent, the former base of this module, is
`peri-rms-scaled-ssmax-inverse-width-lr2`: real score, fixed RoPE, centered XSA, SwiGLU. These
bounds do not depend on the optimizer.

`run` replaces each of the two residual additions per block by four-stream finite mHC whose skip
map is conjugated by the stream phases, `D P D*` (`step`, `phaseMHCStep`). The raw attention
branch includes Q/K projections, normalization, LeRoPE, SSMax with the phase score, ordinary
values and output projection; the raw MLP is the plain complex FFN and output projection. Their
exact internal operations need no extra hypothesis for the output-RMS bound. The final stream
mean is passed through the final RMS and the tied head.

The score theorems state two scores for the same normalized, arbitrarily rotated query and key:
the parent's real score, at most `log N`, which is the model's at `β = 0` (its initialization),
and the phase score with the implemented `β = c / √(1 + c²)`, at most `5/4 · log N`. The projected
query, key and rotary phases are arbitrary, so routing before those projections cannot invalidate
these pointwise bounds. Conjoining the bounds is not a formal translation of the PyTorch forward,
a bound on learned readout weights, or a theorem about training dynamics.

Migrated from diff-gpt c6c517b, `research/mhc_scaled/CurrentModel.lean`; updated to the main model.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Transformer Transformer.Residual Transformer.MHC
open scoped BigOperators

/-- The mHC proposal's final features for either residual coefficient,
immediately before the unchanged tied readout: stream mean, then final RMS. -/
noncomputable def finalFeature {d : ℕ} (eps scale : ℝ)
    (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d) (layers : ℕ) : EucSpace d :=
  GPTMini.rmsNormEps eps (collapse
    (run eps scale attention mlp
      (embeddingState eps embedding) layers))

/-- Main-model contract: each stream and their mean have RMS at most `1 + 2*sqrt(2L)`; for the
same Q/K-normalized, arbitrarily rotated query and key, the inverse-width SSMax real score has
magnitude at most `log(N)` and the phase score with the implemented `β = c / √(1 + c²)` at most
`5/4 * log(N)`. Feature width and attention head width are distinct. The same positive epsilon
is used at the model's RMS sites. The skip maps are phase-conjugated, with arbitrary angles.
Local combination of arXiv:2512.24880v2, Eq. (3), arXiv:2501.19399v1, Eq. (2), and DiffGPT `fe1d528`,
`research/main_model/README.txt` (*Architecture*). -/
theorem current_model_bounds {width headWidth : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (hlayers : 0 < layers) (n : ℕ)
    (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) (c : ℝ) :
    (∀ i, ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    (‖collapse (run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers)‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    |Real.log (n + 1) * ((1 / (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) ∧
    |headPhaseScore headWidth (Real.log (n + 1) / headWidth) (phaseCoefficient c)
      (rotatedQK eps a q) (rotatedQK eps b k)| ≤ 5 / 4 * Real.log (n + 1) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := scaled_model_rms_bound eps heps hwidth attention mlp embedding
    layers hlayers
  have hn : (1 : ℝ) ≤ (n : ℝ) + 1 := le_add_of_nonneg_left (Nat.cast_nonneg n)
  refine ⟨hstreams, ?_, score_bound eps heps hhead n a b q k,
    headPhaseScore_rotatedQK_abs_le eps heps hhead _ _ hn (phaseCoefficient_abs_lt_one c).le
      a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

/-- Matched unscaled contract: set only the post-RMS depth coefficient to one.
The four streams, phase-conjugated finite Sinkhorn skip, sigmoid / twice-sigmoid gates, branch
normalizations, and inverse-width Q/K-normalized SSMax scores are identical.
Local combination of arXiv:2512.24880v2, Eq. (3), arXiv:2501.19399v1, Eq. (2), and DiffGPT `fe1d528`,
`research/main_model/README.txt` (*Architecture*). -/
theorem unscaled_current_model_bounds {width headWidth : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (n : ℕ) (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) (c : ℝ) :
    (∀ i, ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    (‖collapse (run eps 1 attention mlp (embeddingState eps embedding) layers)‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    |Real.log (n + 1) * ((1 / (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) ∧
    |headPhaseScore headWidth (Real.log (n + 1) / headWidth) (phaseCoefficient c)
      (rotatedQK eps a q) (rotatedQK eps b k)| ≤ 5 / 4 * Real.log (n + 1) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := unscaled_model_rms_bound eps heps hwidth attention mlp embedding layers
  have hn : (1 : ℝ) ≤ (n : ℝ) + 1 := le_add_of_nonneg_left (Nat.cast_nonneg n)
  refine ⟨hstreams, ?_, score_bound eps heps hhead n a b q k,
    headPhaseScore_rotatedQK_abs_le eps heps hhead _ _ hn (phaseCoefficient_abs_lt_one c).le
      a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

/-- Positive depth gives a strictly smaller uniform upper bound with scaling.
This compares the two proved bounds, not actual activations, gradients,
or learning outcomes of the scaled and unscaled models.
Comparison of the local post-RMS coefficients in arXiv:2512.24880v2, Eq. (3). -/
theorem scaled_bound_lt_unscaled_bound (layers : ℕ) (hlayers : 0 < layers) :
    1 + 2 * Real.sqrt (2 * (layers : ℝ)) < 1 + 4 * (layers : ℝ) := by
  have hl : (1 : ℝ) ≤ (layers : ℝ) := by exact_mod_cast hlayers
  have hs : Real.sqrt (2 * (layers : ℝ)) < 2 * (layers : ℝ) :=
    Real.sqrt_lt_self_iff.mpr (by linarith)
  linarith

/-- The current final RMS bounds the features supplied to the tied matrix,
independently of depth and the chosen residual coefficient. This does not
bound logits independently of the readout matrix or full-network gradients.
Local final-RMS readout extension of arXiv:2512.24880v2, Eq. (3). -/
theorem final_feature_rms_le_one {d : ℕ} (eps scale : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d) (layers : ℕ) :
    ‖finalFeature eps scale attention mlp embedding layers‖ / Real.sqrt (d : ℝ) ≤ 1 := by
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  simpa only [finalFeature, one_mul] using GPTMini.rmsNormEps_norm_le eps heps hd
    (collapse (run eps scale attention mlp
      (embeddingState eps embedding) layers))

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  current_model_bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    attention mlp embedding 6 (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k 3

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  unscaled_current_model_bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    attention mlp embedding 6 63 (fun _ => 2) (fun _ => 3) q k 3

example : 1 + 2 * Real.sqrt (2 * (6 : ℝ)) < 1 + 4 * (6 : ℝ) := by
  simpa only [Nat.cast_ofNat] using scaled_bound_lt_unscaled_bound 6 (by norm_num)

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128) :
    ‖finalFeature (1 / 8388608) (1 / Real.sqrt 12) attention mlp embedding 6‖ /
        Real.sqrt (128 : ℝ) ≤ 1 := by
  exact final_feature_rms_le_one (1 / 8388608) (1 / Real.sqrt 12) (by norm_num) (by norm_num)
    attention mlp embedding 6

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128) :
    ‖finalFeature (1 / 8388608) 1 attention mlp embedding 6‖ / Real.sqrt (128 : ℝ) ≤ 1 := by
  exact final_feature_rms_le_one (1 / 8388608) 1 (by norm_num) (by norm_num)
    attention mlp embedding 6

end Transformer.MHC.PeriRMS
