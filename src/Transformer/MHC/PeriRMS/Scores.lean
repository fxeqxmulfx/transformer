module

public import Transformer.GPTMini.RMSNorm
public import Transformer.SequenceModel.Rotary

/-!
# Q/K RMS, rotary, and both SSMax attention gains

Source: the diff-gpt model and proofs frozen at c6c517b,
`research/early99/guarantees/NormalizedScores.lean` and the `score_bound`
statement of `research/early99_dimension_gain/Guarantees.lean`.
SSMax is Nakanishi, arXiv:2501.19399v1, Eq. (2), with s=1. The fixed
inverse-head-width gain is the local experiment's explicit adaptation.
The square-root gain is also retained as a separately requested comparison.

The projected query/key vectors and rotary phases are arbitrary. Their
finite-epsilon normalized norms bound the exact real score by log(N),
independently of upstream mHC routing. This gives neither a positive margin
nor a bound on the full input-dependent network Jacobian. No dependency on
the external Python repository or its proof modules is retained.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Transformer
open scoped BigOperators

open Transformer.SequenceModel

/-- Per-head finite-epsilon RMS followed by the actual arbitrary-phase rotary map. -/
noncomputable def rotatedQK {d : ℕ} (eps : ℝ) (angles : Fin (d / 2) → ℝ)
    (x : EucSpace d) : EucSpace d :=
  rotateHead d angles (GPTMini.rmsNormEps eps x)

/-- Post-RMS rotations preserve the uniform sqrt(d) query/key norm bound.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem normalized_norm {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (angles : Fin (d / 2) → ℝ) (x : EucSpace d) :
    ‖rotatedQK eps angles x‖ ≤ Real.sqrt (d : ℝ) := by
  rw [rotatedQK, rotateHead_isometry]
  exact GPTMini.rmsNormEps_norm_le eps heps hd x

/-- Cauchy–Schwarz controls every finite-epsilon normalized Q/K dot product.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem normalized_dot {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |inner ℝ (rotatedQK eps a q) (rotatedQK eps b k)| ≤ (d : ℝ) := by
  calc
    _ ≤ ‖rotatedQK eps a q‖ * ‖rotatedQK eps b k‖ := abs_real_inner_le_norm _ _
    _ ≤ Real.sqrt (d : ℝ) * Real.sqrt (d : ℝ) :=
      mul_le_mul (normalized_norm eps heps hd a q) (normalized_norm eps heps hd b k)
        (norm_nonneg _) (Real.sqrt_nonneg _)
    _ = (d : ℝ) := Real.mul_self_sqrt (Nat.cast_nonneg d)

/-- A fixed multiplier g bounds each ordinary attention score by |g|*d.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem normalized_score {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k)| ≤ |gain| * (d : ℝ) := by
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (normalized_dot eps heps hd a b q k) (abs_nonneg gain)

/-- Fixed s=1 SSMax changes this uniform bound by log(N), for N visible keys.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem ssmax_score {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) *
      (gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * (|gain| * (d : ℝ)) := by
  have hn : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  rw [abs_mul, abs_of_nonneg hn]
  exact mul_le_mul_of_nonneg_left (normalized_score eps gain heps hd a b q k) hn

/-- The declared log(N)/d coefficient gives a width-independent log(N) envelope.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem score_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) * ((1 / (d : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have h := ssmax_score eps (1 / (d : ℝ)) heps hd n a b q k
  simpa [abs_of_pos (div_pos zero_lt_one hd'), ne_of_gt hd'] using h

/-- With the standard gain `1/sqrt(d)`, Q/K RMS and SSMax give the envelope
`sqrt(d)*log(N)`. This specializes the same Eq. (2) and normalization;
the inverse-width theorem above is the separately requested gain `1/d`.
Local Q/K RMS and rotary specialization for arXiv:2501.19399v1, Eq. (2). -/
theorem inverse_sqrt_score_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) * ((1 / Real.sqrt (d : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * Real.sqrt (d : ℝ) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd'
  have hcoef : |1 / Real.sqrt (d : ℝ)| * (d : ℝ) = Real.sqrt (d : ℝ) := by
    rw [abs_of_pos (div_pos zero_lt_one hs)]
    field_simp [ne_of_gt hs]
    nlinarith [Real.sq_sqrt hd'.le]
  simpa only [hcoef] using ssmax_score eps (1 / Real.sqrt (d : ℝ)) heps hd n a b q k

example (q : EucSpace 32) := normalized_norm (1 / 8388608)
  (by norm_num) (by norm_num) (fun _ => 2) q

example (q k : EucSpace 32) := normalized_dot (1 / 8388608)
  (by norm_num) (by norm_num) (fun _ => 2) (fun _ => 3) q k

example (q k : EucSpace 32) := normalized_score (1 / 8388608) (1 / 32)
  (by norm_num) (by norm_num) (fun _ => 2) (fun _ => 3) q k

example (q k : EucSpace 32) := ssmax_score (1 / 8388608) (1 / 32)
  (by norm_num) (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

example (q k : EucSpace 32) := score_bound (1 / 8388608)
  (by norm_num) (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

example (q k : EucSpace 32) := inverse_sqrt_score_bound (1 / 8388608)
  (by norm_num) (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

end Transformer.MHC.PeriRMS
