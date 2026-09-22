/-
# Metastability — the scale of the printed energy

arXiv:2410.06833v1 prints the interaction energy as

  `𝖤_β(x_1,…,x_n) = (1/(2 β e^β n²)) Σ_i Σ_j e^{β⟨x_i, x_j⟩}`

(`Metastability.Eβ`).  Every summand is at most `e^β`, so `0 ≤ 𝖤_β ≤ 1/(2β)`
(`Eβ_le`): the printed energy tends to `0` uniformly in the configuration as
`β → ∞`.  That is harmless for the gradient flow, which it only rescales, but
not for the statements of §6 that take the limit `β → ∞` of `𝖤_β` itself.
`conj: saddle-to-saddle` and `thm: staircase` ask the energy along the flow to
converge to a staircase `φ_∞` with values in `[0, 1]`, and the caption of the
staircase figure says the last step is `1`, "the maximal value of `𝖤_β`".
With the printed normalization every trajectory, under every
reparametrization, converges to `φ_∞ ≡ 0` — `printed_staircase_trivial` — so
the question is empty.

The normalization the section means is the one whose maximum is `1`:

  `2β 𝖤_β(x) = (1/n²) Σ_i Σ_j e^{β(⟨x_i, x_j⟩ - 1)} ∈ (0, 1]`,

which is what `HasStaircaseProfile` is stated with.
-/

import Transformer.Metastability.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- The printed energy is nonnegative.

Source: arXiv:2410.06833v1, §1, definition of `𝖤_β`. -/
theorem Eβ_nonneg {β : ℝ} (hβ : 0 < β) (X : SphereTuple d n) : 0 ≤ Eβ d n β X := by
  unfold Eβ
  positivity

/-- **The printed energy is at most `1/(2β)`.**  Each `e^{β⟨x_i, x_j⟩}` is at
most `e^β`, the `x_i` being unit vectors.

Source: arXiv:2410.06833v1, §1, definition of `𝖤_β`. -/
theorem Eβ_le {β : ℝ} (hβ : 0 < β) (X : SphereTuple d n) : Eβ d n β X ≤ 1 / (2 * β) := by
  have hterm : ∀ i j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) ≤ Real.exp β := by
    intro i j
    refine Real.exp_le_exp.mpr ?_
    have h1 := real_inner_le_norm (X i : EucSpace d) (X j : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp (X i).2, mem_sphere_zero_iff_norm.mp (X j).2] at h1
    nlinarith
  have hsum : ∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d))
        ≤ (n : ℝ) ^ 2 * Real.exp β := by
    calc _ ≤ ∑ _i : Idx n, ∑ _j : Idx n, Real.exp β :=
          Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hterm i j
      _ = (n : ℝ) ^ 2 * Real.exp β := by simp [Finset.card_univ]; ring
  unfold Eβ
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [Nat.cast_zero]
    norm_num
    positivity
  · have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
    calc _ ≤ 1 / (2 * β * Real.exp β * (n : ℝ) ^ 2) * ((n : ℝ) ^ 2 * Real.exp β) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = 1 / (2 * β) := by field_simp

/-- **With the printed energy, the staircase question is empty.**

For every family of trajectories `X β` and every family of time changes
`τ β` whatsoever, the printed energy along them converges, uniformly in time,
to `φ_∞ ≡ 0`: one jump time `T_1 = 1`, and `|𝖤_β - 0| ≤ 1/(2β)`.  This is the
conclusion of `conj: saddle-to-saddle` read with the printed `𝖤_β`, proved
without the dynamics; it is why `HasStaircaseProfile` is stated with `2β 𝖤_β`.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`, with `𝖤_β` of §1. -/
theorem printed_staircase_trivial (hn : 1 ≤ n)
    (X : ℝ → ℝ → SphereTuple d n) (τ : ℝ → ℝ → ℝ) :
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      1 ≤ k ∧ k ≤ n ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ t : ℝ, 0 ≤ t → |Eβ d n β (X β (τ β t)) - φ t| < ε := by
  refine ⟨1, fun i => i, fun _ => 0, le_rfl, hn, by simp, fun i _ => by simp,
    fun _ => ⟨le_rfl, zero_le_one⟩, fun ε hε => ⟨1 / ε, fun β hβ t _ => ?_⟩⟩
  have hβ0 : 0 < β := lt_trans (by positivity) hβ
  rw [sub_zero, abs_of_nonneg (Eβ_nonneg d n hβ0 _)]
  calc _ ≤ 1 / (2 * β) := Eβ_le d n hβ0 _
    _ < 1 / β := by rw [div_lt_div_iff_of_pos_left one_pos (by positivity) hβ0]; linarith
    _ < ε := by rw [div_lt_iff₀ hβ0]; rw [div_lt_iff₀ hε] at hβ; linarith

/-- The hypothesis of `printed_staircase_trivial` is satisfiable: `n = 1`. -/
example : 1 ≤ 1 := le_rfl

end Metastability
end Transformer
