import Transformer.Modes.Section3_ErrorThird
import Transformer.Modes.Section3_EtaMoment
import Transformer.Modes.Section3_PointwiseFalse

/-
# The number of modes of a Gaussian KDE — higher-order errors, pointwise

§3.2 of arXiv:2412.09080v3: counterexamples to `lem:error-higher`, and the
valid arithmetic estimates `eq:rate` on `T` and `T'`.

**What the source says and what is carried here.**

* **The missing Edgeworth term is accounted for.** The source defines `g₂ = q_t - φ` and
  `g₃ = q_t - φ - n^{-1/2}ψ` and then writes `|q_t - φ|` in the display for both
  `s = 2` and `s = 3`. The proof in `sec:pf-error-higher` bounds `g_s`, and
  `cor:error-higher` uses `g₃`. Both the literal display and its corrected
  version are refuted in `Section3_PointwiseFalse.lean`.

* **The global pointwise bounds are false.** In the allowed regime `β = n`,
  every `n ≥ 5` has no bounded standardized density, at any `t`. A finite
  uniform error from the bounded Gaussian or Edgeworth target would give
  precisely such a density. The negative statements below keep the source's
  rates and quantifiers at this explicit regime; continuity is unnecessary.
  The observation point `0` belongs to every window `T`.

* `eq:rate`: the first `≲` is `lem:eta` at `s + 1` (`etaMoment_le`), proved.
  The rest is arithmetic on `T` and `T'` and is proved: `rate_T`, `rate_T'`.

Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `eq:error-higher`,
`eq:rate`, `sec:pf-error-higher`.
-/

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-! ### `lem:error-higher` -/

/-- **Counterexample to `lem:error-higher`, `s = 2`.** In the allowed regime
`n = β = k+1`, even densities without continuity cannot satisfy the source's
uniform weighted error bound. The failure already occurs at `t = 0 ∈ T`.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `eq:error-higher`. -/
theorem not_error_higher_two (ω : ℝ → ℝ) :
    ¬ ∃ C : ℝ, ∀ᶠ k : ℕ in atTop, ∀ t ∈ intervalT (k + 1) (k + 1) (ω (k + 1)),
      ∃ q : ℝ × ℝ → ℝ,
        IsDensityOf (Measure.pi fun _ : Fin (k + 1) => lawY (k + 1) t) (scaledSum (k + 1)) q ∧
        ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x|
          ≤ C * (((k + 1 : ℕ) : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (k + 1) t 3) := by
  rintro ⟨C, hC⟩
  obtain ⟨k, hk, hk4⟩ := (hC.and (eventually_ge_atTop (4 : ℕ))).exists
  obtain ⟨q, hqP, hbound⟩ := hk 0 (zero_mem_intervalT _ _ _)
  exact not_weighted_gaussian_density_scaledSum (n := k + 1) (by omega)
    (by exact_mod_cast (show 2 < k + 1 by omega)) (by norm_num) 0 ⟨q, hqP, _, hbound⟩

/-- **The corrected `s = 3` claim is false as well.** The source's display
omits `n^{-1/2}ψ`; inserting it does not repair the global bound in `β = n`.
No continuity assumption is needed for this counterexample.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, corrected `eq:error-higher`. -/
theorem not_error_higher_three (ω : ℝ → ℝ) :
    ¬ ∃ C : ℝ, ∀ᶠ k : ℕ in atTop, ∀ t ∈ intervalT (k + 1) (k + 1) (ω (k + 1)),
      ∃ q : ℝ × ℝ → ℝ,
        IsDensityOf (Measure.pi fun _ : Fin (k + 1) => lawY (k + 1) t) (scaledSum (k + 1)) q ∧
        ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt (k + 1 : ℕ))⁻¹ * psiOf (lawY (k + 1) t) x|
          ≤ C * (((k + 1 : ℕ) : ℝ)⁻¹ * etaMoment (k + 1) t 4) := by
  rintro ⟨C, hC⟩
  obtain ⟨k, hk, hk4⟩ := (hC.and (eventually_ge_atTop (4 : ℕ))).exists
  obtain ⟨q, hqP, hbound⟩ := hk 0 (zero_mem_intervalT _ _ _)
  exact not_weighted_edgeworth_density_scaledSum (n := k + 1) (by omega)
    (by exact_mod_cast (show 2 < k + 1 by omega)) (by norm_num) 0 ⟨q, hqP, _, hbound⟩

/-- Both counterexamples satisfy the source's original growth hypotheses. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-! ### `eq:rate` -/

/-- `n^{-1}β^{1/2}e^{t²/2}`, the base of `eq:rate`, as one exponential. -/
theorem rate_base_eq {n : ℕ} (hn : 0 < n) {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    (n : ℝ)⁻¹ * Real.sqrt β * Real.exp (t ^ 2 / 2)
      = Real.exp (-Real.log n + Real.log β / 2 + t ^ 2 / 2) := by
  rw [Real.exp_add, Real.exp_add, Real.exp_neg, Real.exp_log (by positivity),
    Real.sqrt_eq_rpow, Real.rpow_def_of_pos hβ, mul_one_div]

/-- **The identity in `eq:rate`:**
`n^{-(s-1)/2}(β e^{t²})^{(s-1)/4} = (n^{-1}β^{1/2}e^{t²/2})^{(s-1)/2}`, the
bound of `lem:eta` on `η_{s+1}` put in the form the source reads the rate off.

Source: arXiv:2412.09080v3, `eq:rate`. -/
theorem rate_eq {n : ℕ} (hn : 0 < n) {β : ℝ} (hβ : 0 < β) (t e : ℝ) :
    (n : ℝ) ^ (-e) * (β * Real.exp (t ^ 2)) ^ (e / 2)
      = ((n : ℝ)⁻¹ * Real.sqrt β * Real.exp (t ^ 2 / 2)) ^ e := by
  rw [rate_base_eq hn hβ, ← Real.exp_mul, Real.rpow_def_of_pos (by positivity),
    Real.rpow_def_of_pos (by positivity), Real.log_mul hβ.ne' (Real.exp_pos _).ne',
    Real.log_exp, ← Real.exp_add]
  congr 1
  ring

/-- `t² ≤ 2 log n - log β - ω` on `T`, when `T` is not degenerate. -/
theorem sq_le_of_mem_intervalT {n : ℕ} {β w t : ℝ}
    (hT : 0 ≤ 2 * Real.log n - Real.log β - w) (ht : t ∈ intervalT n β w) :
    t ^ 2 ≤ 2 * Real.log n - Real.log β - w := by
  have habs : |t| ≤ Real.sqrt (2 * Real.log n - Real.log β - w) :=
    abs_le.2 ⟨by linarith [ht.1], ht.2⟩
  calc t ^ 2 = |t| ^ 2 := (sq_abs t).symm
    _ ≤ Real.sqrt (2 * Real.log n - Real.log β - w) ^ 2 := pow_le_pow_left₀ (abs_nonneg t) habs 2
    _ = _ := Real.sq_sqrt hT

/-- **Equation (eq:rate), on `T`:**
`(n^{-1}β^{1/2}e^{t²/2})^{(s-1)/2} ≤ e^{-(s-1)ω(β)/4}`.  The source's
"by definition of `T`" needs `T` not to be degenerate,
`2 log n - log β - ω(β) ≥ 0`, which the regime gives eventually.

Source: arXiv:2412.09080v3, `eq:rate`. -/
theorem rate_T {n : ℕ} (hn : 0 < n) {β w t : ℝ} (hβ : 0 < β)
    (hT : 0 ≤ 2 * Real.log n - Real.log β - w) (ht : t ∈ intervalT n β w) (s : ℕ) (hs : 1 ≤ s) :
    ((n : ℝ)⁻¹ * Real.sqrt β * Real.exp (t ^ 2 / 2)) ^ (((s : ℝ) - 1) / 2)
      ≤ Real.exp (-((s : ℝ) - 1) * w / 4) := by
  have he : 0 ≤ ((s : ℝ) - 1) / 2 := by
    have : (1 : ℝ) ≤ s := by exact_mod_cast hs
    linarith
  have h2 := sq_le_of_mem_intervalT hT ht
  calc _ ≤ Real.exp (-w / 2) ^ (((s : ℝ) - 1) / 2) := by
        refine Real.rpow_le_rpow (by positivity) ?_ he
        rw [rate_base_eq hn hβ]
        exact Real.exp_le_exp.2 (by linarith)
    _ = _ := by rw [← Real.exp_mul]; congr 1; ring

/-- **Equation (eq:rate), on `T'`:**
`(n^{-1}β^{1/2}e^{t²/2})^{(s-1)/2} ≤ β^{-(s-1)/2}`.  No hypothesis on `T'` is
needed: it is empty unless `β ≤ n^{2/3}`, and then `2 log n - 3 log β ≥ 0`.

Source: arXiv:2412.09080v3, `eq:rate`. -/
theorem rate_T' {n : ℕ} (hn : 0 < n) {β t : ℝ} (hβ : 0 < β) (ht : t ∈ intervalT' n β)
    (s : ℕ) (hs : 1 ≤ s) :
    ((n : ℝ)⁻¹ * Real.sqrt β * Real.exp (t ^ 2 / 2)) ^ (((s : ℝ) - 1) / 2)
      ≤ β ^ (-(((s : ℝ) - 1) / 2)) := by
  unfold intervalT' at ht
  split_ifs at ht with hβn
  · have he : 0 ≤ ((s : ℝ) - 1) / 2 := by
      have : (1 : ℝ) ≤ s := by exact_mod_cast hs
      linarith
    have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
    have hlog : Real.log β ≤ 2 / 3 * Real.log n := by
      rw [← Real.log_rpow hn0]
      exact Real.log_le_log hβ hβn
    have hT : 0 ≤ 2 * Real.log n - 3 * Real.log β := by linarith
    have habs : |t| ≤ Real.sqrt (2 * Real.log n - 3 * Real.log β) :=
      abs_le.2 ⟨by linarith [ht.1], ht.2⟩
    have h2 : t ^ 2 ≤ 2 * Real.log n - 3 * Real.log β := by
      calc t ^ 2 = |t| ^ 2 := (sq_abs t).symm
        _ ≤ Real.sqrt (2 * Real.log n - 3 * Real.log β) ^ 2 :=
          pow_le_pow_left₀ (abs_nonneg t) habs 2
        _ = _ := Real.sq_sqrt hT
    rw [Real.rpow_neg hβ.le, ← Real.inv_rpow hβ.le]
    refine Real.rpow_le_rpow (by positivity) ?_ he
    rw [rate_base_eq hn hβ, ← Real.exp_log (inv_pos.2 hβ), Real.log_inv]
    exact Real.exp_le_exp.2 (by linarith)
  · exact absurd ht (Set.notMem_empty t)

/-- The hypotheses of `rate_T` and `rate_T'` are satisfiable: `n = 1`, `β = 1`,
`ω = 0`, `t = 0`, `s = 2`. -/
example : 0 < 1 ∧ (0 : ℝ) < 1 ∧ 0 ≤ 2 * Real.log (1 : ℕ) - Real.log 1 - 0 ∧
    (0 : ℝ) ∈ intervalT 1 1 0 ∧ (0 : ℝ) ∈ intervalT' 1 1 ∧ 1 ≤ 2 := by
  refine ⟨one_pos, one_pos, by simp, ⟨by simp, by simp⟩, ?_, by norm_num⟩
  simp [intervalT']

end Modes
end Transformer
