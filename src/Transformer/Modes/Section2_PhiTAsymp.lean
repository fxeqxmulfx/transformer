/-
# The number of modes of a Gaussian KDE — `lem:phi-t`, the asymptotics

§2.2 of arXiv:2412.09080v3, `lem:phi-t`: the sizes of `A_t`, `α_t`, `δ_t` and
of the inner Kac-Rice integral over the Gaussian proxy, uniformly over `t ∈ T`.

**What the source says and what is carried here.**

* `≍` over `t ∈ T` is read uniformly: two constants `0 < C₁ ≤ C₂` and, for all
  large `k`, the two-sided bound at every `t ∈ T_k`, along a regime sequence
  `(n_k, β_k)`.  `A_t, α_t, δ_t` are the exact coefficients of
  `krQuad_zero_eq`, so the source's "there exist" is discharged by them.

* `A_t ≍ β^{-3/2} n t² e^{-t²/2}` and `α_t ≍ β^{1/2} e^{t²/2}`: as in the
  source.

* **`δ_t` is not carried as the source writes it.**  The source claims
  `δ_t ≍ n^{1/2}β^{-3/2}e^{-t²/2}(1 - t²/2)`.  The right side vanishes at
  `t = √2 ∈ T`; from the closed forms of §5.2,
  `δ_t = n^{1/2}β^{-3/2}e^{-t²/2}(1 - t²/2 + (-3/2 + 3t²/2 - t⁴/4)/β + O(β⁻²))`,
  which is `≈ n^{1/2}β^{-5/2}e^{-1}/2 > 0` there (checked numerically to seven
  digits), so the claim is false at `t = √2`.  The refutation is not carried
  yet.

* Both bounds are read off the exact identities `phiA_eq`, `phiAlpha_eq`
  and the quotients of `Section2_PhiTQuot.lean`, which lie in `(1/2, 2)` on
  `T` for all large `k` (`eventually_quot_mem_Ioo`).  The check on `δ_t` and
  the second display are in `Section2_PhiTDelta.lean`.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`.
-/

import Transformer.Modes.Section2_PhiTUniform

open Real Filter Asymptotics MeasureTheory
open scoped Topology

namespace Transformer
namespace Modes

/-- **Lemma (lem:phi-t), `A_t ≍ β^{-3/2} n t² e^{-t²/2}`, uniformly on `T`.**

Source: arXiv:2412.09080v3, `lem:phi-t`. -/
theorem phiA_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * (B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2))
          ≤ phiA (N k) (B k) t ∧
        phiA (N k) (B k) t
          ≤ C₂ * (B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2)) := by
  have hq : 0 < (2 : ℝ) ^ (-(5 : ℝ) / 2) * 2 := by positivity
  refine ⟨1 / (8 * ((2 : ℝ) ^ (-(5 : ℝ) / 2) * 2)), 8 / ((2 : ℝ) ^ (-(5 : ℝ) / 2) * 2),
    by positivity, by positivity, ?_⟩
  filter_upwards [eventually_quot_mem_Ioo hreg hω] with k hk t ht
  obtain ⟨⟨a1, b1⟩, -, ⟨a3, b3⟩, -, -⟩ := hk t ht
  have hβ := hreg.B_pos k
  rw [phiA_eq hβ t (by linarith)]
  have hX : 0 ≤ B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2) := by positivity
  generalize B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2) = X at hX ⊢
  generalize (2 : ℝ) ^ (-(5 : ℝ) / 2) * 2 = q at hq ⊢
  generalize gMean ((B k)⁻¹, t ^ 2 / B k) = F at a1 b1 ⊢
  generalize qVar (B k) t = G at a3 b3 ⊢
  have hF : 1 / 4 < F ^ 2 := by nlinarith
  have hF' : F ^ 2 < 4 := by nlinarith
  constructor
  · rw [mul_comm]
    refine mul_le_mul_of_nonneg_left ?_ hX
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_lt_mul_of_pos_right b3 hq, mul_lt_mul_of_pos_right hF hq]
  · rw [mul_comm (8 / q)]
    refine mul_le_mul_of_nonneg_left ?_ hX
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_lt_mul_of_pos_right a3 hq, mul_lt_mul_of_pos_right hF' hq]

/-- **Lemma (lem:phi-t), `α_t ≍ β^{1/2} e^{t²/2}`, uniformly on `T`.**

Source: arXiv:2412.09080v3, `lem:phi-t`. -/
theorem phiAlpha_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) ≤ phiAlpha (B k) t ∧
        phiAlpha (B k) t ≤ C₂ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) := by
  have hp : 0 < (2 : ℝ) ^ (-(5 : ℝ) / 2) := by positivity
  refine ⟨1 / (24 * (2 : ℝ) ^ (-(5 : ℝ) / 2)), 4 / (2 : ℝ) ^ (-(5 : ℝ) / 2),
    by positivity, by positivity, ?_⟩
  filter_upwards [eventually_quot_mem_Ioo hreg hω,
    eventually_forall_intervalT_Ioo hreg hω (G := Prod.snd) continuousAt_snd rfl
      (a := -1) (b := 1 / 100) (by norm_num) (by norm_num)] with k hk hx t ht
  obtain ⟨-, -, ⟨a3, b3⟩, ⟨a4, b4⟩, ⟨a5, b5⟩⟩ := hk t ht
  have hx1 := (hx t ht).2
  have hβ := hreg.B_pos k
  have hx0 : 0 ≤ t ^ 2 / B k := by positivity
  have hD1 : 1 < 6 * qVar (B k) t * qVar' (B k) t - t ^ 2 / B k * qCov (B k) t ^ 2 := by
    nlinarith [mul_lt_mul a3 a5.le (by norm_num) (by linarith),
      mul_le_mul_of_nonneg_left (sq_le_sq' (by linarith) b4.le) hx0]
  have hD2 : 6 * qVar (B k) t * qVar' (B k) t - t ^ 2 / B k * qCov (B k) t ^ 2 < 24 := by
    nlinarith [mul_lt_mul b3 b5.le (by linarith) (by norm_num),
      mul_nonneg hx0 (sq_nonneg (qCov (B k) t))]
  rw [phiAlpha_eq hβ t (by linarith) (by linarith)]
  have hX : 0 ≤ B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  generalize B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) = X at hX ⊢
  generalize (2 : ℝ) ^ (-(5 : ℝ) / 2) = q at hp ⊢
  generalize 6 * qVar (B k) t * qVar' (B k) t - t ^ 2 / B k * qCov (B k) t ^ 2 = D at hD1 hD2 ⊢
  generalize qVar (B k) t = F at a3 b3 ⊢
  constructor
  · rw [mul_comm]
    refine mul_le_mul_of_nonneg_left ?_ hX
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_lt_mul_of_pos_left hD2 hp, mul_lt_mul_of_pos_left a3 hp]
  · rw [mul_comm (4 / q)]
    refine mul_le_mul_of_nonneg_left ?_ hX
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_lt_mul_of_pos_left hD1 hp, mul_lt_mul_of_pos_left b3 hp]

/-- The hypotheses above are satisfiable: `β = n`, `ω(β) = √(log log β)`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Modes
end Transformer
