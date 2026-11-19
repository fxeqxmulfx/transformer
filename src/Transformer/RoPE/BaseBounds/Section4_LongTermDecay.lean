module

public import Mathlib.Analysis.Real.Pi.Bounds
public import Transformer.RoPE.Scores

/-!
# §4.1 Long-term decay of the upper bound of the attention score

Men et al., *Base of RoPE Bounds Context Length*, arXiv:2405.14591v1, §4.1, Eq. (7):
`|A_ij| = |q_iᵀ R_m k_j| ≤ max_l |h_l − h_{l+1}| ∑_{n=1}^{d/2} |S_n|` with
`S_n = ∑_{l=0}^{n−1} e^{(j−i) θ_l √−1}`, "where `h_l = q_iᵀ[2l : 2l+1] k_j[2l : 2l+1]`".

With the printed real products `h_l` the bound is false (`real_decay_bound_false`): for one pair,
`q = (1, 0)` and `k = (0, 1)` give `h_0 = 0`, so with `h_{d/2} = 0` the right side is `0`, while
the score at `m = 1` is `sin 1`. The bound the source cites (Su et al., RoFormer §3.4.3) takes the
complex products `h_l = (q_{2l} + i q_{2l+1})(k_{2l} − i k_{2l+1})` and `h_{d/2} = 0`; summation
by parts proves it (`abs_score_le`). The bound is stated with any `M ≥ |h_l − h_{l+1}|`, the
maximum included. The source's next claim, that the bound "decays as the relative distance
increases", reads its Figure 3 and is not formalized.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE.BaseBounds

/-- RoFormer's complex pair product `h_l = q_[2l:2l+1] k*_[2l:2l+1]` of `N` pairs; `h_l = 0`
from `l = N` on. -/
noncomputable def pairProduct (N : ℕ) (q k : ℕ → ℝ × ℝ) (l : ℕ) : ℂ :=
  if l < N then ⟨(q l).1 * (k l).1 + (q l).2 * (k l).2, (q l).2 * (k l).1 - (q l).1 * (k l).2⟩
  else 0

/-- The source's real pair product `h_l = q[2l:2l+1]ᵀ k[2l:2l+1]`; `h_l = 0` from `l = N` on. -/
noncomputable def realProduct (N : ℕ) (q k : ℕ → ℝ × ℝ) (l : ℕ) : ℝ :=
  if l < N then (q l).1 * (k l).1 + (q l).2 * (k l).2 else 0

/-- The phase sums `S_n = ∑_{l<n} e^{i m θ_l}` of Eq. (7). -/
noncomputable def phaseSum (θ : ℕ → ℝ) (m : ℝ) (n : ℕ) : ℂ :=
  ∑ l ∈ Finset.range n, Complex.exp ((θ l * m : ℝ) * Complex.I)

/-- Summation by parts. -/
theorem sum_mul_eq_abel (h z : ℕ → ℂ) (N : ℕ) :
    ∑ l ∈ Finset.range N, h l * z l =
      ∑ l ∈ Finset.range N, (h l - h (l + 1)) * ∑ j ∈ Finset.range (l + 1), z j +
        h N * ∑ j ∈ Finset.range N, z j := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ (fun l => h l * z l), ih]
    simp only [Finset.sum_range_succ _ N]
    ring

/-- Summation by parts bounds the sum: with `h_N = 0` and `|h_l − h_{l+1}| ≤ M` for `l < N`,
`|∑_{l<N} h_l z_l| ≤ M ∑_{n=1}^{N} |z_0 + ⋯ + z_{n−1}|`. -/
theorem norm_sum_mul_le (h z : ℕ → ℂ) {N : ℕ} (hN : h N = 0) {M : ℝ}
    (hM : ∀ l < N, ‖h l - h (l + 1)‖ ≤ M) :
    ‖∑ l ∈ Finset.range N, h l * z l‖ ≤
      M * ∑ n ∈ Finset.range N, ‖∑ j ∈ Finset.range (n + 1), z j‖ := by
  rw [sum_mul_eq_abel, hN, zero_mul, add_zero, Finset.mul_sum]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun l hl => ?_)
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_right (hM l (Finset.mem_range.mp hl)) (norm_nonneg _)

example (z : ℕ → ℂ) :
    ‖∑ l ∈ Finset.range 2, (if l < 2 then (1 : ℂ) else 0) * z l‖ ≤
      1 * ∑ n ∈ Finset.range 2, ‖∑ j ∈ Finset.range (n + 1), z j‖ :=
  norm_sum_mul_le (fun l => if l < 2 then (1 : ℂ) else 0) z (by simp) fun l hl => by
    interval_cases l <;> simp

/-- A rotated pair's score is the real part of `h_l e^{i m θ_l}`. -/
theorem pairScore_eq_re (q k : ℝ × ℝ) (θ m : ℝ) :
    pairScore q k θ m = ((⟨q.1 * k.1 + q.2 * k.2, q.2 * k.1 - q.1 * k.2⟩ : ℂ) *
      Complex.exp ((θ * m : ℝ) * Complex.I)).re := by
  rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  simp only [pairScore, Complex.mul_re, Complex.add_re, Complex.add_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im]
  ring

/-- Eq. (7) with RoFormer's complex products and `h_{d/2} = 0`: for `N` rotated pairs,
`|∑_l q_lᵀ R_{m θ_l} k_l| ≤ M ∑_{n=1}^{N} |S_n|` whenever `|h_l − h_{l+1}| ≤ M` for `l < N`. -/
theorem abs_score_le (N : ℕ) (q k : ℕ → ℝ × ℝ) (θ : ℕ → ℝ) (m : ℝ) {M : ℝ}
    (hM : ∀ l < N, ‖pairProduct N q k l - pairProduct N q k (l + 1)‖ ≤ M) :
    |∑ l ∈ Finset.range N, pairScore (q l) (k l) (θ l) m| ≤
      M * ∑ n ∈ Finset.range N, ‖phaseSum θ m (n + 1)‖ := by
  have hsum : ∑ l ∈ Finset.range N, pairScore (q l) (k l) (θ l) m =
      (∑ l ∈ Finset.range N,
        pairProduct N q k l * Complex.exp ((θ l * m : ℝ) * Complex.I)).re := by
    rw [Complex.re_sum]
    refine Finset.sum_congr rfl fun l hl => ?_
    rw [pairScore_eq_re, pairProduct, ite_eq_left (Finset.mem_range.mp hl)]
  rw [hsum]
  exact (Complex.abs_re_le_norm _).trans
    (norm_sum_mul_le _ _ (by simp [pairProduct]) hM)

example (q k : ℕ → ℝ × ℝ) (θ : ℕ → ℝ) (m : ℝ) :
    |∑ l ∈ Finset.range 64, pairScore (q l) (k l) (θ l) m| ≤
      (∑ l ∈ Finset.range 64, ‖pairProduct 64 q k l - pairProduct 64 q k (l + 1)‖) *
        ∑ n ∈ Finset.range 64, ‖phaseSum θ m (n + 1)‖ :=
  abs_score_le 64 q k θ m fun _ hl =>
    Finset.single_le_sum (f := fun l => ‖pairProduct 64 q k l - pairProduct 64 q k (l + 1)‖)
      (fun _ _ => norm_nonneg _) (Finset.mem_range.mpr hl)

/-- Eq. (7) with the printed real products `h_l = q[2l:2l+1]ᵀ k[2l:2l+1]` is false. For one
pair, `q = (1, 0)` and `k = (0, 1)` give `h_0 = h_1 = 0`, so the bound with `M = 0` would force
the score `sin 1` to vanish. -/
theorem real_decay_bound_false :
    ¬ ∀ (N : ℕ) (q k : ℕ → ℝ × ℝ) (θ : ℕ → ℝ) (m M : ℝ),
      (∀ l < N, |realProduct N q k l - realProduct N q k (l + 1)| ≤ M) →
        |∑ l ∈ Finset.range N, pairScore (q l) (k l) (θ l) m| ≤
          M * ∑ n ∈ Finset.range N, ‖phaseSum θ m (n + 1)‖ := by
  intro h
  have hbound := h 1 (fun _ => (1, 0)) (fun _ => (0, 1)) (fun _ => 1) 1 0 fun l hl => by
    simp [realProduct, Nat.lt_one_iff.mp hl]
  have hsin := Real.sin_pos_of_pos_of_lt_pi one_pos (by linarith [Real.pi_gt_three])
  simp [pairScore] at hbound
  linarith

end Transformer.RoPE.BaseBounds
