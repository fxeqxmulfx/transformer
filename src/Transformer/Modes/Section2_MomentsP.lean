/-
# The number of modes of a Gaussian KDE — `lem:moments-p`

§2.2 of arXiv:2412.09080v3, `lem:moments-p`: the asymptotics of the mean
`μ_t` and covariance `Σ_t` of the vector `(F_n(t), F_n'(t))`.

**What the source says and what is carried here.**

* `~` is read entrywise, as `Asymptotics.IsEquivalent` along a sequence of
  parameters `(n_k, β_k, t_k)` with `β_k → ∞` and `t_k²/β_k → 0`.  The second
  condition is what §5.2 uses: "if `n^c ≲ β ≲ n^{2-c}` and `t ∈ T`, then
  `exp Θ(t²/β) → 1`".  That the regime and `T` do give it is
  `tendsto_sq_div_of_mem_intervalT`, proved.

* The entries of `Σ_t` are `Var G = E G² - (E G)²`, `Cov(G, G') = E GG' - E G·E G'`
  and `Var G' = E G'² - (E G')²`, the moments of `Section5_Moments.lean`.
  They are stated and proved in `Section2_MomentsPCov.lean`; this file has
  the two entries of `μ_t`.

* **The second entry of `μ_t` is corrected.**  The source writes
  `μ_{t,2} ~ n^{1/2}β^{-3/2}e^{-t²/2}(1 - t²)`.  At `t = ±1` the right side is
  `0` while `E G'(±1) = e^{-β/(2(β+1))}/(β+1)^{5/2} > 0` (`meanG'_eq`), so the
  claim fails there, and near `t = ±1` it fails along any `t_k → ±1` fast
  enough.  The closed form of `E G'` gives the correct leading term
  `n^{1/2}β^{-3/2}e^{-t²/2}(1 - t² + 1/β)`, which is what is stated.  The
  other four entries are as the source writes them.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`; §5.2.
-/

import Transformer.Modes.Section5_MomentsAsymp
import Transformer.Modes.Section1_Sketch

open Real Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-! ### The regime makes `t²/β` vanish on `T` -/

/-- **`t²/β → 0` on `T`.**  In the regime `n^c ≲ β ≲ n^{2-c}`, a point `t` of
`T = [-√(2 log n - log β - ω(β)), √(…)]` has `t² ≤ 2 log n ≲ log β / c`, so
`t²/β → 0`.  This is the remark opening §5.2 that makes every exponential
`exp Θ(t²/β)` in the moments tend to `1`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`), first paragraph. -/
theorem tendsto_sq_div_of_mem_intervalT {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {t : ℕ → ℝ}
    (ht : ∀ᶠ k in atTop, t k ∈ intervalT (N k) (B k) (ω (B k))) :
    Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0) := by
  obtain ⟨C, hC, hCw⟩ := hreg.lower.exists_pos
  have hlog : Tendsto (fun x : ℝ => Real.log x ^ 1 / (1 * x + 0)) atTop (𝓝 0) :=
    Real.tendsto_pow_log_div_mul_add_atTop 1 0 1 one_ne_zero
  have hinv : Tendsto (fun x : ℝ => x⁻¹) atTop (𝓝 0) := tendsto_inv_atTop_zero
  have hg : Tendsto (fun k => 2 / c * (Real.log C * (B k)⁻¹ + Real.log (B k) ^ 1 / (1 * B k + 0)))
      atTop (𝓝 0) := by
    have := ((hinv.comp hreg.tendsto_B).const_mul (Real.log C)).add (hlog.comp hreg.tendsto_B)
    simpa using this.const_mul (2 / c)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hg ?_ ?_
  · filter_upwards with k
    exact div_nonneg (sq_nonneg _) (hreg.B_pos k).le
  · filter_upwards [ht, hCw.bound, hreg.tendsto_N.eventually_ge_atTop 1,
      hreg.tendsto_B.eventually_ge_atTop 1, (hω.lower.comp hreg.tendsto_B).eventually_ge_atTop 0]
      with k htk hk hN hB hωk
    have hB0 : 0 < B k := hreg.B_pos k
    have hN0 : 0 < (N k : ℝ) := by linarith
    have hlogN : 0 ≤ Real.log (N k) := Real.log_nonneg hN
    -- `t² ≤ 2 log n`
    have ht2 : t k ^ 2 ≤ 2 * Real.log (N k) := by
      have habs : |t k| ≤ Real.sqrt (2 * Real.log (N k)) := by
        refine abs_le.mpr ⟨?_, ?_⟩ <;>
        · have := htk.1; have := htk.2
          have hle : Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k))
              ≤ Real.sqrt (2 * Real.log (N k)) :=
            Real.sqrt_le_sqrt (by have := Real.log_nonneg hB; simp at hωk; linarith)
          linarith
      calc t k ^ 2 = |t k| ^ 2 := (sq_abs _).symm
        _ ≤ Real.sqrt (2 * Real.log (N k)) ^ 2 := by gcongr
        _ = 2 * Real.log (N k) := Real.sq_sqrt (by positivity)
    -- `c log n ≤ log C + log β`
    have hcN : c * Real.log (N k) ≤ Real.log C + Real.log (B k) := by
      rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg hB0.le] at hk
      rw [← Real.log_rpow hN0, ← Real.log_mul hC.ne' hB0.ne']
      exact Real.log_le_log (by positivity) hk
    have hc := hreg.c_pos
    calc t k ^ 2 / B k ≤ 2 * Real.log (N k) / B k := by gcongr
      _ ≤ 2 * ((Real.log C + Real.log (B k)) / c) / B k := by
          gcongr
          rw [le_div_iff₀ hc]; linarith
      _ = 2 / c * (Real.log C * (B k)⁻¹ + Real.log (B k) ^ 1 / (1 * B k + 0)) := by
          field_simp; ring

/-- The hypotheses of `tendsto_sq_div_of_mem_intervalT` are satisfiable: the
regime `β = n`, the window `ω(β) = √(log log β)`, and `t_k = 0`, which lies in
every `T`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧
    ∀ k : ℕ, (0 : ℝ) ∈ intervalT (k + 1) ((k + 1 : ℕ) : ℝ)
      (Real.sqrt (Real.log (Real.log ((k + 1 : ℕ) : ℝ)))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, fun _ => ⟨by simp, Real.sqrt_nonneg _⟩⟩

/-! ### `eq:moments-p`, entrywise -/

/-- The prefactor `β^{-3/2}e^{-t²/2}` shared by every entry of `eq:moments-p`. -/
noncomputable def momentScale (β t : ℝ) : ℝ := β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)

/-- **Lemma (lem:moments-p), `μ_{t,1}`.**  `√n E G(t) ~ n^{1/2}β^{-3/2}e^{-t²/2} t`.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`. -/
theorem moments_p_mean_fst {N : ℕ → ℕ} {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) :
    (fun k => Real.sqrt (N k) * meanG (B k) (t k))
      ~[atTop] fun k => Real.sqrt (N k) * momentScale (B k) (t k) * t k := by
  have hp := (tendsto_inv_atTop_zero.comp hB).prodMk_nhds ht
  have hF := (show ContinuousAt (fun p : ℝ × ℝ => Real.exp (p.2 * (1 / (1 + p.1)) / 2)
      * (1 / (1 + p.1)) * √(1 / (1 + p.1))) (0, 0) by fun_prop (disch := norm_num)).tendsto.comp hp
  refine isEquivalent_of_eq_mul (F := fun k => Real.exp (t k ^ 2 / B k * (1 / (1 + (B k)⁻¹)) / 2)
    * (1 / (1 + (B k)⁻¹)) * √(1 / (1 + (B k)⁻¹))) (by simpa [Function.comp_def] using hF) ?_
  filter_upwards [hB.eventually_gt_atTop 0] with k hk
  rw [meanG_eq_mul hk, momentScale]
  ring

/-- **Lemma (lem:moments-p), `μ_{t,2}`, corrected.**
`√n E G'(t) ~ n^{1/2}β^{-3/2}e^{-t²/2}(1 - t² + 1/β)`.

The source writes `1 - t²` in place of `1 - t² + 1/β`, which is false at
`t = ±1`: the right side vanishes and `E G'(±1) > 0`.  See the module
docstring.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`; the correction is
read off the closed form of `E G'` in §5.2. -/
theorem moments_p_mean_snd {N : ℕ → ℕ} {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) :
    (fun k => Real.sqrt (N k) * meanG' (B k) (t k))
      ~[atTop] fun k => Real.sqrt (N k) * momentScale (B k) (t k) * (1 - t k ^ 2 + (B k)⁻¹) := by
  have hp := (tendsto_inv_atTop_zero.comp hB).prodMk_nhds ht
  have hF := (show ContinuousAt (fun p : ℝ × ℝ => Real.exp (p.2 * (1 / (1 + p.1)) / 2)
      * (1 / (1 + p.1)) ^ 2 * √(1 / (1 + p.1))) (0, 0) by
        fun_prop (disch := norm_num)).tendsto.comp hp
  refine isEquivalent_of_eq_mul (F := fun k => Real.exp (t k ^ 2 / B k * (1 / (1 + (B k)⁻¹)) / 2)
    * (1 / (1 + (B k)⁻¹)) ^ 2 * √(1 / (1 + (B k)⁻¹))) (by simpa [Function.comp_def] using hF) ?_
  filter_upwards [hB.eventually_gt_atTop 0] with k hk
  rw [meanG'_eq_mul hk, momentScale]
  ring

/-- The hypotheses of `eq:moments-p` are satisfiable: `β_k = k + 1`, `t_k = 0`. -/
example : Tendsto (fun k : ℕ => (k : ℝ) + 1) atTop atTop ∧
    Tendsto (fun k : ℕ => (0 : ℝ) ^ 2 / ((k : ℝ) + 1)) atTop (𝓝 0) :=
  ⟨tendsto_natSucc_atTop, by simp⟩

end Modes
end Transformer
