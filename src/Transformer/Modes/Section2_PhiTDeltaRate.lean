/-
# The number of modes of a Gaussian KDE — `lem:phi-t`, `α_t δ_t²` at the rate

The check on `δ_t` in the proof of `lem:phi-t` bounds `α_t δ_t²` by a constant,
which needs `n ≲ β^{5/2}` (`Section2_PhiTDelta.lean`).  What holds with no bound
on `n` at all is the bound with the rate kept:
`α_t δ_t² ≲ n β^{-5/2} e^{-t²/2} (1 + t²)²`, uniformly on `T`.  It is what
`eq:int-phi-final` needs on `T'`, where `n β^{-3/2} e^{-t²/2} ≥ 1` can be huge
and the shift `δ_t` is paid for by the factor `e^{-A_t/2}` and the width of the
integral in `t` (`Section2_ProxyKRTPrime.lean`, `Section2_WidthIntegral.lean`).

Source: arXiv:2412.09080v3, proof of `lem:phi-t`.
-/

import Transformer.Modes.Section2_PhiTAsymp

open Real Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-- **Lemma (lem:phi-t), the check on `δ_t`, at the rate.**
`α_t δ_t² ≤ C n β^{-5/2} e^{-t²/2} (4 + 6t²)²` uniformly on `T`, with no bound
on `n`.  With `n ≲ β^{5/2}` this is `α_t δ_t² = O(1)`
(`phiAlpha_mul_phiDelta_sq_le`).

Source: arXiv:2412.09080v3, proof of `lem:phi-t`. -/
theorem phiAlpha_mul_phiDelta_sq_le_rate {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 ≤
        C * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * (4 + 6 * t ^ 2) ^ 2 := by
  obtain ⟨C₁, C₂, -, hC₂, hα⟩ := phiAlpha_isTheta hreg hω
  refine ⟨C₂, hC₂, ?_⟩
  filter_upwards [hα, eventually_quot_mem_Ioo hreg hω,
    hreg.tendsto_B.eventually_ge_atTop 1] with k hαk hk hB t ht
  have hβ := hreg.B_pos k
  obtain ⟨⟨a1, b1⟩, ⟨a2, b2⟩, ⟨a3, b3⟩, ⟨a4, b4⟩, -⟩ := hk t ht
  have hε0 : 0 ≤ (B k)⁻¹ := inv_nonneg.2 hβ.le
  have hε1 : (B k)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hB
  set E := gMean' ((B k)⁻¹, t ^ 2 / B k) * (1 - t ^ 2 + (B k)⁻¹)
      + t ^ 2 * (qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t)) with hE
  have hR0 : 0 < qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t) := by
    apply div_pos <;> nlinarith
  have hR4 : qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t) < 4 := by
    rw [div_lt_iff₀ (by linarith)]; nlinarith [mul_lt_mul b4 b1.le (by linarith) (by norm_num)]
  have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
  have hEu : E ≤ 4 + 6 * t ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hR4.le ht2,
      mul_nonneg (by linarith : (0 : ℝ) ≤ gMean' ((B k)⁻¹, t ^ 2 / B k)) ht2,
      mul_le_mul_of_nonneg_right b2.le (by linarith : (0 : ℝ) ≤ 1 + (B k)⁻¹)]
  have hEl : -(4 + 6 * t ^ 2) ≤ E := by
    nlinarith [mul_nonneg ht2 hR0.le, mul_le_mul_of_nonneg_right b2.le ht2,
      mul_nonneg (by linarith : (0 : ℝ) ≤ gMean' ((B k)⁻¹, t ^ 2 / B k)) hε0]
  have hsq : E ^ 2 ≤ (4 + 6 * t ^ 2) ^ 2 := sq_le_sq' hEl hEu
  have h1 : B k ^ ((1 : ℝ) / 2) * (B k ^ (-(3 : ℝ) / 2)) ^ 2 = B k ^ (-(5 : ℝ) / 2) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ.le, ← Real.rpow_add hβ]; norm_num
  have h2 : Real.exp (t ^ 2 / 2) * Real.exp (-(t ^ 2) / 2) ^ 2 = Real.exp (-(t ^ 2) / 2) := by
    rw [show Real.exp (-(t ^ 2) / 2) ^ 2 = Real.exp (-(t ^ 2) / 2) * Real.exp (-(t ^ 2) / 2)
      from sq _, ← mul_assoc, ← Real.exp_add, ← Real.exp_add]
    congr 1; ring
  rw [phiDelta_eq hβ t (by linarith), ← hE]
  calc phiAlpha (B k) t * (√(N k) * momentScale (B k) t * E) ^ 2
      ≤ C₂ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2))
          * (√(N k) * momentScale (B k) t * E) ^ 2 :=
        mul_le_mul_of_nonneg_right (hαk t ht).2 (sq_nonneg _)
    _ = C₂ * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * E ^ 2 := by
        rw [mul_pow, mul_pow, Real.sq_sqrt (Nat.cast_nonneg _), momentScale]
        linear_combination (C₂ * N k * E ^ 2 * (Real.exp (t ^ 2 / 2)
          * Real.exp (-(t ^ 2) / 2) ^ 2)) * h1 + (C₂ * N k * E ^ 2 * B k ^ (-(5 : ℝ) / 2)) * h2
    _ ≤ C₂ * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2))
          * (4 + 6 * t ^ 2) ^ 2 := by
        gcongr

/-- The hypotheses above are satisfiable: `β = n`, `ω(β) = √(log log β)`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Modes
end Transformer
