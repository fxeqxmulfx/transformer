/-
# The softmax weights, and how far two of them can be apart

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Appendix D, `eq: lip.3`, needs one quantitative fact about softmax: moving the
scores by `ε` in the sup-norm moves the weights by `O(ε)` in *total variation*,
with a constant that does not depend on the number of scores.

The survey states it as "the softmax function with a parameter `β` is
`β`-Lipschitz (with respect to the Euclidean norm)" and then pays a factor `√n`
twice for passing through the Euclidean norm.  The `ℓ^∞ → ℓ^1` form proved
here,

  `Σ_j |σ(u)_j - σ(v)_j| ≤ 4 ‖u - v‖_∞`,

loses no factor of `n` at all.  The argument is elementary: the two weight
vectors are within a factor `e^{±2ε}` of each other entrywise, so their
positive parts differ by at most `(1 - e^{-2ε}) σ(u)_j`, which sums to
`1 - e^{-2ε} ≤ 2ε`; and a difference of two probability vectors has equal
positive and negative mass.

The constant `4` is not sharp — the sharp one is `1` — but it is what the
elementary argument gives, and it is enough for `eq: stability.4ortho`.
-/

import Transformer.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {m : ℕ}

/-- The softmax weights of a vector of scores: `σ(u)_j = e^{u_j} / Σ_k e^{u_k}`.

Source: arXiv:2312.10794v5, §2, the weights in `eq: SA`. -/
noncomputable def softmaxWeight (u : Idx m → ℝ) (j : Idx m) : ℝ :=
  Real.exp (u j) / ∑ k : Idx m, Real.exp (u k)

/-- The softmax partition function is positive as soon as there is a score. -/
theorem softmaxPartition_pos (hm : 0 < m) (u : Idx m → ℝ) :
    0 < ∑ k : Idx m, Real.exp (u k) := by
  have : Nonempty (Idx m) := ⟨⟨0, hm⟩⟩
  exact Finset.sum_pos (fun k _ => Real.exp_pos (u k)) Finset.univ_nonempty

/-- The hypothesis of `softmaxPartition_pos` is satisfiable. -/
example : 0 < 1 := one_pos

/-- The softmax weights are nonnegative. -/
theorem softmaxWeight_nonneg (u : Idx m → ℝ) (j : Idx m) : 0 ≤ softmaxWeight u j :=
  div_nonneg (Real.exp_pos _).le
    (Finset.sum_nonneg fun k _ => (Real.exp_pos (u k)).le)

/-- The softmax weights sum to `1`. -/
theorem sum_softmaxWeight (hm : 0 < m) (u : Idx m → ℝ) :
    ∑ j : Idx m, softmaxWeight u j = 1 := by
  simp only [softmaxWeight]
  rw [← Finset.sum_div, div_self (softmaxPartition_pos hm u).ne']

/-- The hypothesis of `sum_softmaxWeight` is satisfiable. -/
example : 0 < 1 := one_pos

/-- **The softmax is `4`-Lipschitz from `ℓ^∞` to `ℓ^1`.**

  `Σ_j |σ(u)_j - σ(v)_j| ≤ 4 ε`   whenever   `|u_j - v_j| ≤ ε` for every `j`.

This is the quantitative content of `eq: lip.3`, in the norms that do not cost
a power of `n`.

Source: arXiv:2312.10794v5, Appendix D, `eq: lip.3`. -/
theorem sum_abs_softmaxWeight_sub_le (hm : 0 < m) (u v : Idx m → ℝ) (ε : ℝ)
    (hε : 0 ≤ ε) (huv : ∀ j : Idx m, |u j - v j| ≤ ε) :
    ∑ j : Idx m, |softmaxWeight u j - softmaxWeight v j| ≤ 4 * ε := by
  have hZu : 0 < ∑ k : Idx m, Real.exp (u k) := softmaxPartition_pos hm u
  have hZv : 0 < ∑ k : Idx m, Real.exp (v k) := softmaxPartition_pos hm v
  -- the two partition functions are within `e^{±ε}` of each other
  have hZle : (∑ k : Idx m, Real.exp (v k))
      ≤ Real.exp ε * ∑ k : Idx m, Real.exp (u k) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [neg_le_of_abs_le (huv k)])
  -- hence the two weight vectors are within `e^{±2ε}` of each other, entrywise
  have hE : Real.exp (-(2 * ε)) * Real.exp ε = Real.exp (-ε) := by
    rw [← Real.exp_add]; ring_nf
  have key : ∀ j : Idx m,
      Real.exp (-(2 * ε)) * softmaxWeight u j ≤ softmaxWeight v j := by
    intro j
    have h1 : Real.exp (-ε) * Real.exp (u j) ≤ Real.exp (v j) := by
      rw [← Real.exp_add]
      exact Real.exp_le_exp.mpr (by linarith [le_of_abs_le (huv j)])
    rw [softmaxWeight, softmaxWeight, mul_div_assoc', div_le_div_iff₀ hZu hZv]
    calc Real.exp (-(2 * ε)) * Real.exp (u j) * ∑ k : Idx m, Real.exp (v k)
        ≤ Real.exp (-(2 * ε)) * Real.exp (u j)
            * (Real.exp ε * ∑ k : Idx m, Real.exp (u k)) := by
          exact mul_le_mul_of_nonneg_left hZle (by positivity)
      _ = (Real.exp (-(2 * ε)) * Real.exp ε)
            * (Real.exp (u j) * ∑ k : Idx m, Real.exp (u k)) := by ring
      _ = (Real.exp (-ε) * Real.exp (u j)) * ∑ k : Idx m, Real.exp (u k) := by
          rw [hE]; ring
      _ ≤ Real.exp (v j) * ∑ k : Idx m, Real.exp (u k) :=
          mul_le_mul_of_nonneg_right h1 hZu.le
  -- a difference of probability vectors carries equal positive and negative mass
  have hsum0 : ∑ j : Idx m, (softmaxWeight u j - softmaxWeight v j) = 0 := by
    rw [Finset.sum_sub_distrib, sum_softmaxWeight hm, sum_softmaxWeight hm, sub_self]
  have habs : ∀ j : Idx m, |softmaxWeight u j - softmaxWeight v j|
      = 2 * max (softmaxWeight u j - softmaxWeight v j) 0
        - (softmaxWeight u j - softmaxWeight v j) := by
    intro j
    rcases le_total 0 (softmaxWeight u j - softmaxWeight v j) with h | h
    · rw [abs_of_nonneg h, max_eq_left h]; ring
    · rw [abs_of_nonpos h, max_eq_right h]; ring
  have hexp1 : Real.exp (-(2 * ε)) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hmaxle : ∀ j : Idx m, max (softmaxWeight u j - softmaxWeight v j) 0
      ≤ (1 - Real.exp (-(2 * ε))) * softmaxWeight u j := by
    intro j
    refine max_le ?_ ?_
    · have := key j; nlinarith
    · have := softmaxWeight_nonneg u j; nlinarith
  calc ∑ j : Idx m, |softmaxWeight u j - softmaxWeight v j|
      = 2 * (∑ j : Idx m, max (softmaxWeight u j - softmaxWeight v j) 0)
          - ∑ j : Idx m, (softmaxWeight u j - softmaxWeight v j) := by
        simp only [habs]
        rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
    _ = 2 * ∑ j : Idx m, max (softmaxWeight u j - softmaxWeight v j) 0 := by
        rw [hsum0]; ring
    _ ≤ 2 * ∑ j : Idx m, (1 - Real.exp (-(2 * ε))) * softmaxWeight u j := by
        have := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) => hmaxle j)
        linarith
    _ = 2 * (1 - Real.exp (-(2 * ε))) := by
        rw [← Finset.mul_sum, sum_softmaxWeight hm]; ring
    _ ≤ 4 * ε := by
        have := Real.add_one_le_exp (-(2 * ε))
        linarith

/-- The hypotheses of `sum_abs_softmaxWeight_sub_le` are satisfiable: one score,
equal on both sides, with `ε = 0`. -/
example : (0 : ℕ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    ∀ j : Idx 1, |(fun _ : Idx 1 => (0 : ℝ)) j - (fun _ : Idx 1 => (0 : ℝ)) j| ≤ 0 :=
  ⟨one_pos, le_rfl, fun _ => by simp⟩

end Perspective
end Transformer
