/-
# A perturbed stack: the error after `N` steps

**Not a statement of any paper.**  The record of parameter-golf
(github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`)
runs its eleven layers in the order `0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, …, 10`
(`GPT.__init__`, `all_indices`, at the defaults `NUM_LAYERS = 11`,
`NUM_LOOPS = 2`, `LOOP_START = 3`, `LOOP_END = 5`), and ships its weight
matrices rounded to six bits (`gptq_mixed_quantize`, `MATRIX_BITS = 6`).  The
rounded model runs every block a little differently from the trained one, and
every block after the first on a state that the blocks before it have already
moved.  A depth-recurrent submission of the non-record track
(`records/track_non_record_16mb/2026-03-21_DepthRecurrence_MixedPrecisionQuant/README.md`,
"What Worked", §1) reports its first run at `2.07` bits per byte before the
rounding and `3.22` after, and the error amplified "roughly 900x through 3
recurrence cycles".

How far the rounding can move the output is a discrete Gronwall bound.  Let a
stack run `x_{k+1} = F_k(x_k)` and a perturbed copy `y_{k+1} = G_k(y_k)`, let
`L_k` bound how far `F_k` spreads the two states and `η_k` how far `G_k` is from
`F_k` at the perturbed one.  Then

  `‖y_N − x_N‖ ≤ ‖y_0 − x_0‖ Π_{j<N} L_j + Σ_{m<N} η_m Π_{m<j<N} L_j`

(`norm_sub_le_perturbBound`, from Mathlib's `discrete_gronwall_prod_general`):
the error made at each step is carried forward by the spreading of the steps
after it, and nothing else enters — in particular not whether two steps share
their weights.  A loop repeats one block and one rounding error, but the bound
sees only the constants along the two trajectories.  It cannot be improved in
terms of those constants: a linear stack perturbed in one fixed direction at
every step attains it with equality (`norm_sub_linearStack`).
-/

import Mathlib.Analysis.ODE.DiscreteGronwall
import Mathlib.Analysis.Normed.Module.Basic

open scoped BigOperators

namespace Transformer
namespace Perspective

/-- **The Gronwall bound of a perturbed stack.**  After `N` steps of spreading
`L_j` and errors `η_m`, from an initial distance `e₀`,

  `e₀ Π_{j<N} L_j + Σ_{m<N} η_m Π_{m<j<N} L_j`.

Source: none — posed here; the right-hand side of Mathlib's
`discrete_gronwall_prod_general`, started at `0`. -/
noncomputable def perturbBound (L η : ℕ → ℝ) (e₀ : ℝ) (N : ℕ) : ℝ :=
  e₀ * ∏ j ∈ Finset.range N, L j
    + ∑ m ∈ Finset.range N, η m * ∏ j ∈ Finset.Ico (m + 1) N, L j

@[simp] theorem perturbBound_zero (L η : ℕ → ℝ) (e₀ : ℝ) : perturbBound L η e₀ 0 = e₀ := by
  simp [perturbBound]

/-- **One more step spreads the bound and adds its own error.**

Source: none — posed here; the step of the induction behind
`discrete_gronwall_prod_general`. -/
theorem perturbBound_succ (L η : ℕ → ℝ) (e₀ : ℝ) (N : ℕ) :
    perturbBound L η e₀ (N + 1) = L N * perturbBound L η e₀ N + η N := by
  have hterm : ∀ m ∈ Finset.range N,
      η m * ∏ j ∈ Finset.Ico (m + 1) (N + 1), L j
        = L N * (η m * ∏ j ∈ Finset.Ico (m + 1) N, L j) := by
    intro m hm
    rw [Finset.prod_Ico_succ_top (Finset.mem_range.mp hm)]
    ring
  unfold perturbBound
  rw [Finset.prod_range_succ, Finset.sum_range_succ, Finset.sum_congr rfl hterm,
    Finset.Ico_self, Finset.prod_empty, mul_add, Finset.mul_sum]
  ring

/-- **Gronwall across a perturbed stack.**  If `x_{k+1} = F_k(x_k)` and
`y_{k+1} = G_k(y_k)`, where `F_k` spreads the two states by at most `L_k ≥ 0`
and `G_k` is within `η_k` of `F_k` at `y_k`, then after `N` steps

  `‖y_N − x_N‖ ≤ ‖y_0 − x_0‖ Π_{j<N} L_j + Σ_{m<N} η_m Π_{m<j<N} L_j`.

Nothing is assumed of the maps away from the two trajectories, and the steps
may repeat one block or all be different.

Source: none — posed here; the rounded weights of the record's looped layers
(`train_gpt.py`, `GPT.__init__` and `gptq_mixed_quantize`), and the error
compounding through the loop of the depth-recurrent submission (its
`README.md`, "What Worked", §1). -/
theorem norm_sub_le_perturbBound {E : Type*} [SeminormedAddCommGroup E]
    {F G : ℕ → E → E} {x y : ℕ → E} {L η : ℕ → ℝ}
    (hx : ∀ k, x (k + 1) = F k (x k)) (hy : ∀ k, y (k + 1) = G k (y k))
    (hL : ∀ k, 0 ≤ L k) (hF : ∀ k, ‖F k (y k) - F k (x k)‖ ≤ L k * ‖y k - x k‖)
    (hG : ∀ k, ‖G k (y k) - F k (y k)‖ ≤ η k) (N : ℕ) :
    ‖y N - x N‖ ≤ perturbBound L η ‖y 0 - x 0‖ N := by
  have hrec : ∀ k ≥ 0, ‖y (k + 1) - x (k + 1)‖ ≤ L k * ‖y k - x k‖ + η k := by
    intro k _
    rw [hx, hy]
    calc ‖G k (y k) - F k (x k)‖
        = ‖(F k (y k) - F k (x k)) + (G k (y k) - F k (y k))‖ := by congr 1; abel
      _ ≤ ‖F k (y k) - F k (x k)‖ + ‖G k (y k) - F k (y k)‖ := norm_add_le _ _
      _ ≤ L k * ‖y k - x k‖ + η k := add_le_add (hF k) (hG k)
  have h := discrete_gronwall_prod_general (u := fun k => ‖y k - x k‖) (c := L) (b := η)
    hrec (fun k _ => hL k) (Nat.zero_le N)
  simpa [perturbBound] using h

/-- The hypotheses of `norm_sub_le_perturbBound` are satisfiable, with an error
that grows: on `ℝ`, `F_k z = 2 z` from `x = 0` and `G_k z = 2 z + 1` from
`y_0 = 0`, so that `y_k = 2^k - 1`, with `L = 2` and `η = 1`. -/
example : (∀ _ : ℕ, (0 : ℝ) = 2 * 0) ∧
    (∀ k : ℕ, (2 : ℝ) ^ (k + 1) - 1 = 2 * (2 ^ k - 1) + 1) ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 2) ∧
    (∀ k : ℕ, ‖2 * ((2 : ℝ) ^ k - 1) - 2 * 0‖ ≤ 2 * ‖(2 : ℝ) ^ k - 1 - 0‖) ∧
    ∀ k : ℕ, ‖2 * ((2 : ℝ) ^ k - 1) + 1 - 2 * (2 ^ k - 1)‖ ≤ 1 :=
  ⟨fun _ => by ring, fun k => by ring, fun _ => by norm_num, fun k => by simp [norm_mul],
    fun k => by simp⟩

/-- **The bound is attained.**  A linear stack `x_{k+1} = L_k x_k`, perturbed at
every step by `η_k` in one fixed unit direction `u` and started `e₀` apart along
`u`, ends exactly at the Gronwall bound: the errors never cancel, so no bound in
terms of the spreadings and the errors alone is smaller.

Source: none — posed here; the sharpness of `norm_sub_le_perturbBound`, whose
hypotheses this stack meets with the same `L` and `η`. -/
theorem norm_sub_linearStack {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    {x y : ℕ → E} {L η : ℕ → ℝ} {u : E} {e₀ : ℝ} (hL : ∀ k, 0 ≤ L k)
    (hη : ∀ k, 0 ≤ η k) (he₀ : 0 ≤ e₀) (hu : ‖u‖ = 1) (hx : ∀ k, x (k + 1) = L k • x k)
    (hy : ∀ k, y (k + 1) = L k • y k + η k • u) (h0 : y 0 - x 0 = e₀ • u) (N : ℕ) :
    ‖y N - x N‖ = perturbBound L η e₀ N := by
  have hdiff : ∀ k, y k - x k = perturbBound L η e₀ k • u := by
    intro k
    induction k with
    | zero => rw [perturbBound_zero]; exact h0
    | succ k ih =>
        rw [hx, hy, perturbBound_succ, add_smul, mul_smul, ← ih, smul_sub]
        abel
  have hnn : 0 ≤ perturbBound L η e₀ N := by
    induction N with
    | zero => rw [perturbBound_zero]; exact he₀
    | succ N ih => rw [perturbBound_succ]; exact add_nonneg (mul_nonneg (hL N) ih) (hη N)
  rw [hdiff, norm_smul, hu, mul_one, Real.norm_of_nonneg hnn]

/-- The hypotheses of `norm_sub_linearStack` are satisfiable, with an initial
error and a growing one: on `ℝ`, `L = 2`, `η = 1`, `u = 1`, `e₀ = 1`, `x = 0` and
`y_k = 2^{k+1} - 1`. -/
example : (∀ _ : ℕ, (0 : ℝ) ≤ 2) ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 1) ∧ (0 : ℝ) ≤ 1 ∧ ‖(1 : ℝ)‖ = 1 ∧
    (∀ _ : ℕ, (0 : ℝ) = (2 : ℝ) • (0 : ℝ)) ∧
    (∀ k : ℕ, (2 : ℝ) ^ (k + 1 + 1) - 1 = (2 : ℝ) • ((2 : ℝ) ^ (k + 1) - 1) + (1 : ℝ) • (1 : ℝ)) ∧
    (2 : ℝ) ^ (0 + 1) - 1 - 0 = (1 : ℝ) • (1 : ℝ) := by
  refine ⟨fun _ => by norm_num, fun _ => by norm_num, zero_le_one, norm_one,
    fun _ => by simp, fun k => ?_, by norm_num⟩
  simp only [smul_eq_mul]
  ring

end Perspective
end Transformer
