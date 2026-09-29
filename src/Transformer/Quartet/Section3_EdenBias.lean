/-
# The Corollary of §3.3 is false at every dimension

arXiv:2601.22813v2, §3.3: "For all `x ∈ ℝ^d` and scale `s ≠ 0`",
`E_{ω_RHT, ω_SR} RHT⁻¹(x̂, ω_RHT) = x` for the output `x̂` of `MS-EDEN`.

`Transformer.Quartet.Section3_EdenFalse` kills the quantifier over `s`: at `s = 21` the corrected
group scale saturates the E4M3 grid.  This module kills the quantifier over `d`, *inside* the
window of clipping factors the paper uses, `6 · 16/17 ≤ s ≤ (1/0.93) · 6 · 16/17` — the
non-clipping bound of §3.3 and the factor it "uses for the rest of the paper".  For every
dimension `d = 16 · 2^k` and every `s` in the window there is a vector `x` with

`E_{ω_RHT, ω_SR} RHT⁻¹(x̂, ω_RHT)_{0,0} = 101/102 ≠ 1 = x_{0,0}`.

The witness is `x = e₀ + (1/10) e₁` (`Section3_EdenPair`).  Its rotation takes only two values,
`(σ₀ ± σ₁/10)/√d` by the parity of the position, where `σ₀, σ₁ = ±1` are the signs the seed gives
the two nonzero entries.  So for every seed the rotated tensor has the same maximum `M = 11a`
in every group, `a = 1/(10 √d)`, every group scale is exactly `256`
(`Section3_EdenTwoValued`), the larger entries round to `±6` and the smaller ones, `9/11` of
them, to `±4`, and the EDEN correction is `S = s · 202 a² / (11 a · 102 a)` in every group.  The
`2^k` stochastic roundings of `256 S` are unbiased (`Section3_EdenCoins`); what is left is
arithmetic (`Section3_EdenMean`), and it gives `101/102`, exactly.

§3.2 quotes EDEN's guarantee for the `RHT` as a *limit* `d → ∞` ("in practice … converges fast
enough to be unbiased with RHT performed in groups as small as `d = 64`"), and Appendix A says the
kernels use `d = 128`; the Corollary drops the limit.  At finite `d` it is false: `d = 64` is
`k = 2`, `d = 128` is `k = 3`.  For `k = 3`, `msEden` is Algorithm 1 verbatim, one chunk of `128`;
for other `k` it is Algorithm 1 with the chunk equal to the whole vector, which the paper allows
("any multiple of the quantization group size 16 is valid").
-/

import Transformer.Quartet.Section3_EdenMean

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- **The mean over the coins of `MS-EDEN` on `edenPair`, at the coordinate `(0, 0)`, is
`101/102`**, for every seed of the signs, every `s` in the window `6 · 16/17 ≤ s ≤ (1/0.93) ·
6 · 16/17` and every dimension `16 · 2^k`.  The four sign patterns of the seed at the two
nonzero entries give the rotation `(±11a, ±9a)` or `(±9a, ±11a)`, with `a = 1/(10 √d)`. -/
theorem integral_rhtInv_msEden_edenPair {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s)
    (hs₁ : s ≤ 6 * (16 / 17) / 0.93) (ε : Fin (2 ^ k) → Fin 16 → Bool) :
    ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1),
      rhtInv k ε (msEden k s (edenPair k) ε u) 0 0 = 101 / 102 := by
  norm_num at hs₀ hs₁
  have hs5 : 5 < s := by linarith
  have hs61 : s ≤ 61 / 10 := by linarith
  have hy := rht_edenPair (k := k) ε
  obtain ⟨a, ha⟩ : ∃ a : ℝ, a = 1 / Real.sqrt (2 ^ (k + 4)) / 10 := ⟨_, rfl⟩
  have ha0 : 0 < a := by rw [ha]; positivity
  rcases h0 : ε 0 0 with _ | _ <;> rcases h1 : ε 0 1 with _ | _
  · -- signs `+`, `+`: `P = 11a`, `R = 9a`
    have hy' : rht k ε (edenPair k) = twoValued k (11 * a) (9 * a) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha] <;> ring
    exact integral_rhtInv_msEden_edenPair_case (qP := 6) (qR := 4) (σ := 1) hy'
      (by rw [h0]; simp) ha hs5 hs61 (abs_of_pos (by positivity)).le
      (by rw [abs_of_pos (by positivity)]; linarith) (Or.inl (abs_of_pos (by positivity)))
      (by rw [show 11 * a * s / (11 * a) = s by field_simp]; exact rtn_fp4_six hs5)
      (by rw [show 9 * a * s / (11 * a) = 9 / 11 * s by field_simp]
          exact rtn_fp4_four (by linarith) (by linarith))
      (by ring) (by ring) (by norm_num)
  · -- signs `+`, `-`: `P = 9a`, `R = 11a`
    have hy' : rht k ε (edenPair k) = twoValued k (9 * a) (11 * a) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha] <;> ring
    exact integral_rhtInv_msEden_edenPair_case (qP := 4) (qR := 6) (σ := 1) hy'
      (by rw [h0]; simp) ha hs5 hs61 (by rw [abs_of_pos (by positivity)]; linarith)
      (abs_of_pos (by positivity)).le (Or.inr (abs_of_pos (by positivity)))
      (by rw [show 9 * a * s / (11 * a) = 9 / 11 * s by field_simp]
          exact rtn_fp4_four (by linarith) (by linarith))
      (by rw [show 11 * a * s / (11 * a) = s by field_simp]; exact rtn_fp4_six hs5)
      (by ring) (by ring) (by norm_num)
  · -- signs `-`, `+`: `P = -9a`, `R = -11a`
    have hy' : rht k ε (edenPair k) = twoValued k (-(9 * a)) (-(11 * a)) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha] <;> ring
    exact integral_rhtInv_msEden_edenPair_case (qP := -4) (qR := -6) (σ := -1) hy'
      (by rw [h0]; simp) ha hs5 hs61
      (by rw [abs_neg, abs_of_pos (by positivity)]; linarith)
      (by rw [abs_neg, abs_of_pos (by positivity)])
      (Or.inr (by rw [abs_neg, abs_of_pos (by positivity)]))
      (by rw [show -(9 * a) * s / (11 * a) = -(9 / 11 * s) by field_simp]
          exact rtn_fp4_neg_four (by linarith) (by linarith))
      (by rw [show -(11 * a) * s / (11 * a) = -s by field_simp]; exact rtn_fp4_neg_six hs5)
      (by ring) (by ring) (by norm_num)
  · -- signs `-`, `-`: `P = -11a`, `R = -9a`
    have hy' : rht k ε (edenPair k) = twoValued k (-(11 * a)) (-(9 * a)) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha] <;> ring
    exact integral_rhtInv_msEden_edenPair_case (qP := -6) (qR := -4) (σ := -1) hy'
      (by rw [h0]; simp) ha hs5 hs61
      (by rw [abs_neg, abs_of_pos (by positivity)])
      (by rw [abs_neg, abs_of_pos (by positivity)]; linarith)
      (Or.inl (by rw [abs_neg, abs_of_pos (by positivity)]))
      (by rw [show -(11 * a) * s / (11 * a) = -s by field_simp]; exact rtn_fp4_neg_six hs5)
      (by rw [show -(9 * a) * s / (11 * a) = -(9 / 11 * s) by field_simp]
          exact rtn_fp4_neg_four (by linarith) (by linarith))
      (by ring) (by ring) (by norm_num)

/-- **`MS-EDEN` is biased on `edenPair`**: the mean over both seeds of `RHT⁻¹(x̂)` at `(0, 0)` is
`101/102`, not `1`, in every dimension `16 · 2^k` and for every `s` in the window of the
paper. -/
theorem mean_rhtInv_msEden_edenPair {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s)
    (hs₁ : s ≤ 6 * (16 / 17) / 0.93) :
    mean k (fun ε u => rhtInv k ε (msEden k s (edenPair k) ε u) 0 0) = 101 / 102 :=
  mean_eq_of_forall (integral_rhtInv_msEden_edenPair hs₀ hs₁)

/-- **The Corollary of §3.3 is false at every dimension, even in the window of clipping factors
the paper uses**: for every `k` and every `s` with `6 · 16/17 ≤ s ≤ (1/0.93) · 6 · 16/17`, the
statement "`E_{ω_RHT, ω_SR} RHT⁻¹(x̂, ω_RHT) = x` for all `x`" fails at `x = e₀ + (1/10) e₁`,
where the mean is `101/102`.

This refutes what was the sorried theorem `mean_rhtInv_msEden` — the paper's Corollary with `s`
restricted to that window, which is the strongest form of it that `not_mean_rhtInv_msEden`
leaves standing.  The paper's own reading of EDEN for the `RHT` is a limit `d → ∞` (§3.2).

Source: arXiv:2601.22813v2, §3.3, the Corollary after Algorithm 1. -/
theorem not_mean_rhtInv_msEden_window (k : ℕ) {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s)
    (hs₁ : s ≤ 6 * (16 / 17) / 0.93) :
    ¬ ∀ (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16),
      mean k (fun ε u => rhtInv k ε (msEden k s x ε u) i j) = x i j := by
  intro h
  have h0 := h (edenPair k) 0 0
  rw [mean_rhtInv_msEden_edenPair hs₀ hs₁] at h0
  simp [edenPair] at h0
  norm_num at h0

/-- The hypotheses are satisfiable, and the statement bites at the paper's own settings: the
rotation group `d = 128` (`k = 3`) of its kernels and the clipping factor `(1/0.93) · 6 · 16/17`
it uses "for the rest of the paper". -/
example : 6 * (16 / 17) ≤ (6 * (16 / 17) / 0.93 : ℝ) ∧
    ¬ ∀ (x : Fin (2 ^ 3) → Fin 16 → ℝ) (i : Fin (2 ^ 3)) (j : Fin 16),
      mean 3 (fun ε u => rhtInv 3 ε (msEden 3 (6 * (16 / 17) / 0.93) x ε u) i j) = x i j :=
  ⟨by norm_num, not_mean_rhtInv_msEden_window 3 (by norm_num) le_rfl⟩

end Quartet
end Transformer
