/-
# Binomial arithmetic for Wendel's recurrence

The strict-sign count satisfies a Pascal recurrence. These identities show
that Wendel's closed expression satisfies the same recurrence and has the
same full-dimensional base value.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCountBase

namespace Transformer.Perspective

/-- Pascal's recurrence summed over the first `d + 1` coefficients.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem wendel_binomial_recurrence (n d : ℕ) :
    (∑ k ∈ Finset.range (d + 1), (n + 1).choose k) =
      (∑ k ∈ Finset.range (d + 1), n.choose k) +
        (∑ k ∈ Finset.range d, n.choose k) := by
  induction d with
  | zero => simp
  | succ d ih =>
    calc
      (∑ k ∈ Finset.range (d + 1 + 1), (n + 1).choose k) =
          (∑ k ∈ Finset.range (d + 1), (n + 1).choose k) +
            (n + 1).choose (d + 1) := Finset.sum_range_succ _ _
      _ = ((∑ k ∈ Finset.range (d + 1), n.choose k) +
            (∑ k ∈ Finset.range d, n.choose k)) +
            (n.choose (d + 1) + n.choose d) := by
          rw [ih, Nat.choose_succ_succ]
          simp only [Nat.succ_eq_add_one]
          omega
      _ = (∑ k ∈ Finset.range (d + 1 + 1), n.choose k) +
            (∑ k ∈ Finset.range (d + 1), n.choose k) := by
          have hs1 : (∑ k ∈ Finset.range (d + 1 + 1), n.choose k) =
              (∑ k ∈ Finset.range (d + 1), n.choose k) + n.choose (d + 1) :=
            Finset.sum_range_succ _ _
          have hs2 : (∑ k ∈ Finset.range (d + 1), n.choose k) =
              (∑ k ∈ Finset.range d, n.choose k) + n.choose d :=
            Finset.sum_range_succ _ _
          rw [hs1, hs2]
          omega

/-- The sum in Wendel's expression is `2^(d-1)` at `n = d ≥ 1`.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem wendel_binomial_fullDim (d : ℕ) (hd : 1 ≤ d) :
    2 * (∑ k ∈ Finset.range d, (d - 1).choose k) = 2 ^ d := by
  have h := Nat.sum_range_choose (d - 1)
  have hsub : d - 1 + 1 = d := by omega
  rw [hsub] at h
  rw [h]
  calc
    2 * 2 ^ (d - 1) = 2 ^ (d - 1 + 1) := by rw [pow_succ]; omega
    _ = 2 ^ d := by rw [hsub]

end Transformer.Perspective
