module

public import Transformer.SequenceModel.Attention.Model

/-!
# Exact inversion of a normalized residual branch

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E1.
Positive-epsilon RMSNorm maps onto the open ball of radius sqrt(width).
The explicit inverse below lets a finite interpolation construction retain
the actual Peri-RMSNorm output, rather than replace normalization by identity.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini

/-- Exact positive radial multiplier for a vector with squared norm n. -/
def normScale (d : ℕ) (eps n : ℝ) : ℝ := Real.sqrt (d : ℝ) / Real.sqrt (n + d * eps)

/-- E1: known squared norm gives the precise scale including epsilon. -/
theorem rms_of_normSq (d : ℕ) (eps : ℝ) (v : EucSpace d) (n : ℝ)
    (hv : ‖v‖ ^ 2 = n) : rmsNormEps eps v = normScale d eps n • v := by
  unfold rmsNormEps normScale
  rw [hv]

/-- E1: the source scale is strictly positive at every positive width and epsilon. -/
theorem normScale_pos (d : ℕ) (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (n : ℝ) (hn : 0 ≤ n) : 0 < normScale d eps n := by
  exact div_pos (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd))
    (Real.sqrt_pos.mpr (add_pos_of_nonneg_of_pos hn (mul_pos (Nat.cast_pos.mpr hd) heps)))

/-- Raw branch value whose normalized output is the requested vector. -/
def unnormalize (d : ℕ) (eps : ℝ) (v : EucSpace d) : EucSpace d :=
  Real.sqrt ((d : ℝ) * eps / ((d : ℝ) - ‖v‖ ^ 2)) • v

/-- E1: every vector strictly inside the RMSNorm radius has an exact preimage.
This is a full positive-epsilon identity, not a zero-epsilon limit. -/
theorem rms_unnormalize (d : ℕ) (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (v : EucSpace d) (hv : ‖v‖ ^ 2 < (d : ℝ)) :
    rmsNormEps eps (unnormalize d eps v) = v := by
  let t := Real.sqrt ((d : ℝ) * eps / ((d : ℝ) - ‖v‖ ^ 2))
  have hd' : 0 < (d : ℝ) := Nat.cast_pos.mpr hd
  have hden : 0 < (d : ℝ) - ‖v‖ ^ 2 := sub_pos.mpr hv
  have ht : 0 < t := Real.sqrt_pos.mpr (div_pos (mul_pos hd' heps) hden)
  have ht2 : t ^ 2 = (d : ℝ) * eps / ((d : ℝ) - ‖v‖ ^ 2) :=
    Real.sq_sqrt (div_pos (mul_pos hd' heps) hden).le
  have hbalance : t ^ 2 * ((d : ℝ) - ‖v‖ ^ 2) = (d : ℝ) * eps := by
    rw [ht2]
    exact div_mul_cancel₀ _ hden.ne'
  have hr : ‖t • v‖ ^ 2 + (d : ℝ) * eps = (d : ℝ) * t ^ 2 := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos ht, mul_pow]
    nlinarith
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd'
  change (Real.sqrt (d : ℝ) / Real.sqrt (‖t • v‖ ^ 2 + (d : ℝ) * eps)) • (t • v) = v
  rw [hr, Real.sqrt_mul hd'.le, Real.sqrt_sq ht.le, smul_smul]
  have hc : Real.sqrt (d : ℝ) / (Real.sqrt (d : ℝ) * t) * t = 1 := by
    field_simp
  rw [hc, one_smul]

/-- E1: a residual target can be hit exactly when its displacement lies in
the true branch-output ball. Later interpolation supplies this raw output. -/
theorem residual_target (d : ℕ) (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (x target : EucSpace d) (h : ‖target - x‖ ^ 2 < (d : ℝ)) :
    x + rmsNormEps eps (unnormalize d eps (target - x)) = target := by
  rw [rms_unnormalize d hd eps heps _ h]
  abel

example := rms_of_normSq 4 (1 / 8388608) (WithLp.toLp 2 ![1, 0, 0, 0]) 1
  (by norm_num [EuclideanSpace.norm_sq_eq, Fin.sum_univ_succ])
example := normScale_pos 4 (by decide) (1 / 8388608) (by norm_num) 2 (by norm_num)
example := rms_unnormalize 4 (by decide) (1 / 8388608) (by norm_num)
  (WithLp.toLp 2 ![1, -1, -1, 0])
  (by norm_num [EuclideanSpace.norm_sq_eq, Fin.sum_univ_succ])
example := residual_target 4 (by decide) (1 / 8388608) (by norm_num)
  (WithLp.toLp 2 ![1, 0, 1, 0]) (WithLp.toLp 2 ![0, 1, 0, 0])
  (by norm_num [EuclideanSpace.norm_sq_eq, Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Expressivity
