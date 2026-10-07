/-
# The number of modes of a Gaussian KDE — the standardized summand is bounded on `T`

The covariance estimates of `lem:moments-p` give the pointwise bound
`‖Y(t,x)‖ ≲ (β exp(t²))^{1/4}`, uniformly in `t ∈ T` and the sample point.
Unlike the raw-moment calculation in the manuscript, this bound can be
combined with `E‖Y(t)‖² = 2` to prove all the moment bounds of `lem:eta`.

Source: arXiv:2412.09080v3, `eq:Yi`, `lem:eta`, §5.2–5.3.
-/

import Transformer.Modes.Section3_SummandBounds
import Transformer.Modes.Section2_ProxyKRScale

open Real Filter
open scoped Topology

namespace Transformer.Modes

/-- On `T`, the covariance ratio is at most `2√β`, uniformly for large
regime indices. Source: arXiv:2412.09080v3, `lem:moments-p`, §5.2–5.3. -/
theorem eventually_sigmaCov_div_sigmaFst_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      |sigmaCov (B k) t / sigmaFst (B k) t| ≤ 2 * Real.sqrt (B k) := by
  filter_upwards [eventually_quot_mem_Ioo hreg hω,
    eventually_forall_intervalT_Ioo hreg hω (G := Prod.snd) continuousAt_snd rfl
      (a := -1) (b := 1) (by norm_num) (by norm_num)] with k hq hx t ht
  have hβ := hreg.B_pos k
  obtain ⟨-, -, ⟨hvar, -⟩, ⟨hcov, hcov₂⟩, -⟩ := hq t ht
  have hqpos : 0 < qVar (B k) t := by linarith
  have hm : 0 < momentScale (B k) t := by unfold momentScale; positivity
  have hp : (2 : ℝ) ^ (-(5 : ℝ) / 2) ≠ 0 := by positivity
  have heq : sigmaCov (B k) t / sigmaFst (B k) t =
      -t * (qCov (B k) t / (2 * qVar (B k) t)) := by
    rw [sigmaCov_eq hβ, sigmaFst_eq hβ]
    field_simp [hqpos.ne', hm.ne', hp]
  have hratio : |qCov (B k) t / (2 * qVar (B k) t)| ≤ 2 := by
    rw [abs_of_nonneg (by positivity), div_le_iff₀ (by positivity)]
    linarith
  have ht₂ : t ^ 2 ≤ B k := by
    have := (hx t ht).2
    rw [div_lt_iff₀ hβ] at this
    linarith
  have htabs : |t| ≤ Real.sqrt (B k) := by
    nlinarith [Real.sq_sqrt hβ.le, Real.sqrt_nonneg (B k), abs_nonneg t, sq_abs t]
  rw [heq, abs_mul, abs_neg]
  calc
    _ ≤ |t| * 2 := mul_le_mul_of_nonneg_left hratio (abs_nonneg _)
    _ ≤ 2 * Real.sqrt (B k) := by linarith

/-- The covariance-ratio estimate has the standard regime witness. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-- The scale `β^{1/2}exp(t²/2)` is the square of
`(β exp(t²))^{1/4}`. Source: arXiv:2412.09080v3, `lem:eta`, §5.3. -/
theorem etaScale_sq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    ((β * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)) ^ 2 =
      β ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
  norm_num
  rw [Real.mul_rpow hβ.le (Real.exp_pos _).le, ← Real.exp_mul]
  congr 2
  ring

/-- The scale identity holds at a positive bandwidth. -/
example : (0 : ℝ) < 1 := one_pos

/-- Squared Euclidean norms of the standardized summands are bounded by
the square of their claimed scale. Source: arXiv:2412.09080v3, `lem:eta`, §5.3. -/
theorem eventually_singleY_sq_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop,
      ∀ t ∈ intervalT (N k) (B k) (ω (B k)), ∀ x : ℝ,
        eucl (singleY (B k) t x) ^ 2 ≤
          C * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) := by
  obtain ⟨L, _, hL, _, hF⟩ := eventually_sigmaFst_bounds hreg hω
  obtain ⟨_, U, _, hU, hA⟩ := phiAlpha_isTheta hreg hω
  refine ⟨4 / L + 64 * U, by positivity, ?_⟩
  filter_upwards [hF, hA, eventually_sigmaCov_div_sigmaFst_le hreg hω] with
    k hFk hAk hbk t ht x
  have hβ := hreg.B_pos k
  have hf := sigmaFst_pos hβ t
  have hd := sigmaDet_pos hβ t
  let m := momentScale (B k) t
  let S := B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)
  have hm : 0 < m := by dsimp [m, momentScale]; positivity
  have hS : 0 < S := by dsimp [S]; positivity
  have hscale : m * S = (B k)⁻¹ := momentScale_mul_alphaBase hβ t
  have hfirst : 4 / (B k * sigmaFst (B k) t) ≤ (4 / L) * S := by
    calc
      _ ≤ 4 / (B k * (L * m)) := by
        gcongr
        exact (hFk t ht).1
      _ = (4 / L) * S := by
        have he : B k * m * S = 1 := by rw [mul_assoc, hscale]; field_simp
        apply (div_eq_iff (mul_pos hβ (mul_pos hL hm)).ne').mpr
        field_simp [hL.ne']
        nlinarith [he]
  have hsecond : 64 * phiAlpha (B k) t ≤ (64 * U) * S := by
    have := mul_le_mul_of_nonneg_left (hAk t ht).2 (show (0 : ℝ) ≤ 64 by norm_num)
    simpa [S, mul_assoc] using this
  obtain ⟨hz₁, hz₂⟩ := centered_bigG_bounds hβ t x
  calc
    _ ≤ 4 / (B k * sigmaFst (B k) t) + 64 * phiAlpha (B k) t := by
      simpa [singleY, sigmaDet, phiAlpha] using
        eucl_whiten_sq_le_of_bounds hf hd hβ
          (bigG (B k) t x - meanG (B k) t, bigG' (B k) t x - meanG' (B k) t)
          hz₁ hz₂ (hbk t ht)
    _ ≤ (4 / L) * S + (64 * U) * S := add_le_add hfirst hsecond
    _ = (4 / L + 64 * U) * S := by ring

/-- The squared-summand bound has simultaneous regime and slow-growth
witnesses. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-- The uniform pointwise bound for `Y(t,x)` on `T`.
Source: arXiv:2412.09080v3, `lem:eta`, §5.3. -/
theorem eventually_singleY_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop,
      ∀ t ∈ intervalT (N k) (B k) (ω (B k)), ∀ x : ℝ,
        eucl (singleY (B k) t x) ≤
          C * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4) := by
  obtain ⟨C, hC, hbound⟩ := eventually_singleY_sq_le hreg hω
  refine ⟨Real.sqrt C, Real.sqrt_pos.mpr hC, ?_⟩
  filter_upwards [hbound] with k hk t ht x
  have hβ := hreg.B_pos k
  have hsq := hk t ht x
  rw [← etaScale_sq hβ t] at hsq
  have he : (Real.sqrt C * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)) ^ 2 =
      C * ((B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hC.le]
  have hnonneg : 0 ≤ Real.sqrt C * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4) := by positivity
  nlinarith [Real.sqrt_nonneg ((singleY (B k) t x).1 ^ 2 + (singleY (B k) t x).2 ^ 2)]

/-- The pointwise bound has the standard regime witness. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-- The pointwise bound is realized at a positive bandwidth and at `t = 0`
for every sample point, using `β = n` and the slow-growth witness above. -/
example : ∃ β C : ℝ, 0 < β ∧ 0 < C ∧ ∀ x : ℝ,
    eucl (singleY β 0 x) ≤ C * (β * Real.exp (0 ^ 2)) ^ ((1 : ℝ) / 4) := by
  obtain ⟨C, hC, hbound⟩ :=
    eventually_singleY_le isRegime_succ isSlowGrowth_sqrt_log_log
  obtain ⟨k, hk⟩ := hbound.exists
  refine ⟨((k + 1 : ℕ) : ℝ), C, by positivity, hC, ?_⟩
  exact hk 0 (zero_mem_intervalT _ _ _)

end Transformer.Modes
