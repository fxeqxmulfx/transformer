import Transformer.Modes.Section3_ErrorThird

/-
# The number of modes of a Gaussian KDE — higher-order errors, pointwise

§3.2 of arXiv:2412.09080v3: `lem:error-higher`, the analogue of `thm:br` for
`q_t` with the dependence on `β` tracked, and `eq:rate`, the rate it gives on
`T` and `T'`.

**What the source says and what is carried here.**

* **`eq:error-higher` is corrected.**  The source defines `g₂ = q_t - φ` and
  `g₃ = q_t - φ - n^{-1/2}ψ` and then writes `|q_t - φ|` in the display for both
  `s = 2` and `s = 3`.  For `s = 3` that claims `|q_t - φ| ≲ n^{-1}η₄`, which
  is false as soon as `n^{-1/2}ψ ≠ 0` — the Edgeworth term is the first-order
  error.  The proof in `sec:pf-error-higher` bounds `g_s`, and `cor:error-higher`
  uses `g₃`; what is stated is `g_s`.

* The source fixes `t` in the proof and uses the bound uniformly in `t ∈ T` in
  `cor:error-higher`; the uniform statement is the one stated.  The proof's
  `ε = sup_{‖z‖>a} |𝓕f(z)| < 1` depends on `t` and `β`, and the uniformity
  needs `ε^{n/5}` to beat a power of `n` uniformly, which the source does not
  address.

* The density `q_t` is asserted to exist, be continuous, and satisfy the bound:
  a continuous density is unique, so this is the source's claim, and it cannot
  hold vacuously for want of a density.

* `eq:rate`: the first `≲` is `lem:eta` at `s + 1` (`etaMoment_le`), unproved.
  The rest is arithmetic on `T` and `T'` and is proved: `rate_T`, `rate_T'`.

Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `eq:error-higher`,
`eq:rate`, `sec:pf-error-higher`.
-/

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-! ### `lem:error-higher` -/

/-- **Lemma (lem:error-higher), `s = 2`, corrected.**  In the regime
`n^c ≲ β ≲ n^{2-c}`, uniformly in `t ∈ T`, the density `q_t` of
`n^{-1/2} Σ Yᵢ(t)` exists, is continuous, and
`sup_x (1 + ‖x‖²) |q_t - φ|(x) ≲ n^{-1/2} η₃`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:error-higher`, `eq:error-higher`. -/
theorem error_higher_two {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
        IsDensityOf (Measure.pi fun _ : Fin (N k) => lawY (B k) t) (scaledSum (N k)) q ∧
        ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x|
          ≤ C * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) := by
  sorry

/-- **Lemma (lem:error-higher), `s = 3`, corrected.**  In the regime
`n^c ≲ β ≲ n^{2-c}`, uniformly in `t ∈ T`, the density `q_t` of
`n^{-1/2} Σ Yᵢ(t)` exists, is continuous, and
`sup_x (1 + ‖x‖³) |q_t - φ - n^{-1/2}ψ|(x) ≲ n^{-1} η₄`.  The source writes
`|q_t - φ|`; see the module docstring.

Not proved here.

Source: arXiv:2412.09080v3, `lem:error-higher`, `eq:error-higher`. -/
theorem error_higher_three {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
        IsDensityOf (Measure.pi fun _ : Fin (N k) => lawY (B k) t) (scaledSum (N k)) q ∧
        ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt (N k))⁻¹ * psiOf (lawY (B k) t) x|
          ≤ C * ((N k : ℝ)⁻¹ * etaMoment (B k) t 4) := by
  sorry

/-- The hypotheses of `error_higher_two` and `error_higher_three` are
satisfiable. -/
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
