/-
# AdaFisher: convergence with corrected step-size conditions

arXiv:2405.16397v3, Proposition 3.3, Appendix A.2.
This proves both objective convergence and convergence in squared Euclidean
distance for the corrected theorem. The source's printed step is refuted.
-/

import Transformer.AdaFisher.Section3_ConvexCorrected
import Mathlib.Topology.MetricSpace.Pseudo.Lemmas

open scoped BigOperators
open Filter Topology

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- Corrected convergence of the unmomented AdaFisher iteration,
Proposition 3.3, Appendix A.2. Smoothness and strong convexity are given
by their standard two-sided first-order models; `star` is a stationary
point. Fisher bounds `[δ,B]` are separate, and η L≤δ replaces η≤1/L.
The squared Euclidean distance to `star` and the objective gap both tend
to zero. All matrices may vary at every iteration within these bounds. -/
theorem corrected_convex_convergence (J : (Fin d → ℝ) → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (L μ η δ B : ℝ)
    (f : ℕ → Fin d → ℝ) (initial star : Fin d → ℝ)
    (hupper : SmoothUpperModel J g L) (hlower : StrongLowerModel J g μ)
    (hstar : g star = 0) (hμ : 0 < μ) (hη : 0 < η) (hδ : 0 < δ) (hB : 0 < B)
    (hstep : L * η ≤ δ) (hrate : η * μ ≤ B)
    (hf : ∀ t i, δ ≤ f t i ∧ f t i ≤ B) :
    Tendsto (fun t => J (gradientRun η f g initial t) - J star) atTop (𝓝 0) ∧
    Tendsto (fun t => squaredNorm (gradientRun η f g initial t - star)) atTop (𝓝 0) := by
  have hlowerGap (x : Fin d → ℝ) : μ / 2 * squaredNorm (x - star) ≤ J x - J star := by
    have h := hlower star x
    rw [hstar] at h
    simp only [pairing, Pi.zero_apply, zero_mul, Finset.sum_const_zero, add_zero] at h
    linarith
  have hgap (x : Fin d → ℝ) : 0 ≤ J x - J star := by
    have hn := squaredNorm_nonneg (x - star)
    nlinarith [hlowerGap x]
  have hq : 0 ≤ 1 - η * μ / B := sub_nonneg.mpr ((div_le_one hB).mpr hrate)
  have hq1 : 1 - η * μ / B < 1 := by
    have hp : 0 < η * μ / B := div_pos (mul_pos hη hμ) hB
    linarith
  have hp := (tendsto_pow_atTop_nhds_zero_of_abs_lt_one
    (by rw [abs_of_nonneg hq]; exact hq1)).mul_const (J initial - J star)
  simp only [zero_mul] at hp
  have hgapLimit := squeeze_zero
    (fun t => hgap (gradientRun η f g initial t))
    (corrected_convex_rate J g L μ η δ B f initial star hupper hlower
      hμ hη.le hδ hB hstep hrate hf) hp
  refine ⟨hgapLimit, ?_⟩
  have hμ0 : μ ≠ 0 := ne_of_gt hμ
  have hid : (2 / μ) * (μ / 2) = (1 : ℝ) := by field_simp
  have hdist (x : Fin d → ℝ) : squaredNorm (x - star) ≤ (2 / μ) * (J x - J star) := by
    have h := mul_le_mul_of_nonneg_left (hlowerGap x) (by positivity : 0 ≤ 2 / μ)
    rwa [← mul_assoc, hid, one_mul] at h
  apply squeeze_zero (fun t => squaredNorm_nonneg (gradientRun η f g initial t - star))
    (fun t => hdist (gradientRun η f g initial t))
  simpa only [mul_zero] using hgapLimit.const_mul (2 / μ)

example :
    SmoothUpperModel (vectorQuadratic (d := 1)) id 1 ∧
    StrongLowerModel (vectorQuadratic (d := 1)) id 1 ∧
    id (0 : Fin 1 → ℝ) = 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 1 * (1 / 2 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) * 1 ≤ 1 ∧
    (∀ (t : ℕ) (i : Fin 1), 1 ≤ (fun _ _ => (1 : ℝ)) t i ∧
      (fun _ _ => (1 : ℝ)) t i ≤ 1) := by
  refine ⟨vectorQuadratic_models.1, vectorQuadratic_models.2, rfl, ?_⟩
  norm_num

end Transformer.AdaFisher
