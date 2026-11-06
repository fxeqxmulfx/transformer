module

public import Transformer.MHC.PeriRMS.Basic

/-!
# Scaled and unscaled finite mHC depth bounds

Local extension of arXiv:2512.24880v2, Eq. (3), around the two Peri-RMS
sublayers of the diff-gpt model frozen at c6c517b. Migrated from
`research/mhc_scaled/Bounds.lean`, with the same hypotheses and constants.

There are attention and SwiGLU updates per decoder block. Normalized
embeddings and output-normalized branches give per-stream RMS bounds
`1 + 2*sqrt(2L)` for coefficient `1/sqrt(2L)`, and `1 + 4L` for coefficient
one. The factor two retains the paper's `2*sigmoid` write gates. Branches
need not be independent, and routing may depend on current streams.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Transformer Transformer.Residual Transformer.MHC
open scoped BigOperators

/-- Current decoder order: attention addition, then SwiGLU addition.
Layer count counts decoder blocks, not the two residual sublayers. -/
noncomputable def run {d : ℕ} (eps scale : ℝ)
    (attention mlp : ℕ → Sublayer d) (initial : State d) : ℕ → State d
  | 0 => initial
  | k + 1 =>
    step eps scale (mlp k) (step eps scale (attention k)
      (run eps scale attention mlp initial k))

/-- Both sublayers contribute at most `2 * scale * sqrt(width)` per block.
The bound permits correlated branches and input-dependent routing.
Local depth iteration of arXiv:2512.24880v2, Eq. (3), with Eq. (8) write gates. -/
theorem run_norm_le {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (attention mlp : ℕ → Sublayer d)
    (initial : State d) (R : ℝ) (hx : ∀ i, ‖initial i‖ ≤ R) (layers : ℕ) :
    ∀ i, ‖run eps scale attention mlp initial layers i‖ ≤
      R + (layers : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) := by
  induction layers with
  | zero => simpa only [run, Nat.cast_zero, zero_mul, add_zero] using hx
  | succ k ih =>
    have ha := step_norm_le eps scale heps hd hscale (attention k)
      (run eps scale attention mlp initial k) _ ih
    have hm := step_norm_le eps scale heps hd hscale (mlp k)
      (step eps scale (attention k) (run eps scale attention mlp initial k)) _ ha
    intro i
    calc
      ‖run eps scale attention mlp initial (k + 1) i‖ ≤
          R + (k : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) +
            2 * scale * Real.sqrt (d : ℝ) + 2 * scale * Real.sqrt (d : ℝ) := hm i
      _ = R + ((k + 1 : ℕ) : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) := by
        push_cast
        ring

/-- Specialization to `1/sqrt(2L)` after output RMS and twice-sigmoid write-back.
The per-stream RMS bound is `1 + 2*sqrt(2L)`, not the single-stream
unit-write bound `1 + sqrt(2L)`. Positive width, epsilon and depth are explicit.
Local post-RMS depth coefficient applied to arXiv:2512.24880v2, Eq. (3). -/
theorem scaled_model_rms_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d)
    (layers : ℕ) (hlayers : 0 < layers) (i : Fin 4) :
    ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ / Real.sqrt (d : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ)) := by
  have hpos : (0 : ℝ) < 2 * (layers : ℝ) := by positivity
  have hspos : 0 < Real.sqrt (2 * (layers : ℝ)) := Real.sqrt_pos.mpr hpos
  have hsquare := Real.sq_sqrt hpos.le
  have hcoef : (2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))) =
      Real.sqrt (2 * (layers : ℝ)) := by
    field_simp
    nlinarith
  have hr := run_norm_le eps (1 / Real.sqrt (2 * (layers : ℝ))) heps hd
    (by positivity) attention mlp (embeddingState eps embedding) (Real.sqrt (d : ℝ))
    (fun _ => GPTMini.rmsNormEps_norm_le eps heps hd embedding) layers i
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  calc
    ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ ≤
        Real.sqrt (d : ℝ) + (layers : ℝ) *
          (4 * (1 / Real.sqrt (2 * (layers : ℝ))) * Real.sqrt (d : ℝ)) := hr
    _ = (1 + 2 * ((2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))))) *
        Real.sqrt (d : ℝ) := by ring
    _ = (1 + 2 * Real.sqrt (2 * (layers : ℝ))) * Real.sqrt (d : ℝ) := by rw [hcoef]

/-- The same mHC / Peri-RMS model without depth scaling: `scale = 1`.
Two sublayers per block and write gates bounded by two give `1 + 4L`.
Depth zero is permitted; all raw branches and routing predictors remain arbitrary.
Local output-RMS extension of arXiv:2512.24880v2, Eqs. (3), (8). -/
theorem unscaled_model_rms_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d)
    (layers : ℕ) (i : Fin 4) :
    ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ /
        Real.sqrt (d : ℝ) ≤ 1 + 4 * (layers : ℝ) := by
  have hr := run_norm_le eps 1 heps hd (by norm_num) attention mlp
    (embeddingState eps embedding) (Real.sqrt (d : ℝ))
    (fun _ => GPTMini.rmsNormEps_norm_le eps heps hd embedding) layers i
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  calc
    ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ ≤
        Real.sqrt (d : ℝ) + (layers : ℝ) * (4 * 1 * Real.sqrt (d : ℝ)) := hr
    _ = (1 + 4 * (layers : ℝ)) * Real.sqrt (d : ℝ) := by ring

example : ∀ i,
    ‖run (1 / 8388608) (1 / 2) (fun _ => witnessBranch) (fun _ => witnessBranch)
        (fun _ => witness) 2 i‖ ≤ (1 : ℝ) + (2 : ℝ) * (4 * (1 / 2) * Real.sqrt 1) := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using
    run_norm_le (d := 1) (1 / 8388608) (1 / 2) (by norm_num) (by norm_num)
      (by norm_num) (fun _ => witnessBranch) (fun _ => witnessBranch)
      (fun _ => witness) 1 (by intro i; simp [witness, PiLp.norm_single]) 2

example :=
  scaled_model_rms_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => witnessBranch) (fun _ => witnessBranch) witness 2 (by norm_num) 0

example :=
  unscaled_model_rms_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => witnessBranch) (fun _ => witnessBranch) witness 2 0

end Transformer.MHC.PeriRMS
