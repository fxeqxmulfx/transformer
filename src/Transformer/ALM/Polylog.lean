/-
# The barrier without the Sparsification Lemma

`Transformer.ALM.SETH` derives the Orthogonal Vectors conjecture from SETH in
the sparsified form, and `Transformer.ALM.Sparsification` shows what that form
costs: a second assumption, the Sparsification Lemma of Impagliazzo, Paturi
and Zane, which is a real theorem but is not proved in this repository.

This file removes that second assumption instead of carrying it.  The price is
paid in the *dimension*: the vectors of the Orthogonal Vectors instance are
indexed by the clauses of the formula, so with `m ≤ n^K` clauses the dimension
is `(log N)^K` rather than `C · log N`.  Nothing else changes, because the
dimension never enters the cost accounting — it is only reported.

`OVHardPoly` is the conjecture at polylogarithmic dimension,
`OVHardPoly_of_OVHard` records that it is the weaker of the two, and
`OVHardPoly_of_SETHGeneral` proves it from SETH alone, with no sparsification
anywhere in the chain.  `query_ge_of_build_small_of_SETHGeneral` is the
barrier that results.

For the lookup machine the weakening is immaterial: an index that answered
exact queries in `n^{1-ε}` at dimension `polylog n` would be just as much of a
breakthrough as one that did it at dimension `Θ(log n)`.  The sharper
`Θ(log n)` form remains available in `Transformer.ALM.SETH`, at the price of
the extra hypothesis.

* V. Vassilevska Williams, *On some fine-grained questions in algorithms and
  complexity*, ICM 2018, §3 — the low-dimensional form of the conjecture, and
  the role sparsification plays in reaching it.
-/

import Transformer.ALM.Sparsification

namespace Transformer
namespace ALM

namespace CostModel

variable (M : CostModel)

/-- **The Orthogonal Vectors conjecture at polylogarithmic dimension**: for
every `ε > 0` there are `c` and `K` such that no correct algorithm stays below
`n^{2-ε}` on instances of *some* dimension `d ≤ c · (log₂ n)^K`.

Two things are weaker here than in `OVHard`.  The dimension is polylogarithmic
rather than logarithmic, and the hard dimension is existential rather than
prescribed — which is what dropping the clause-density restriction leaves. -/
def OVHardPoly : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ c K : ℕ, ∀ a : M.Alg, M.Solves a → ∀ N : ℕ,
    ∃ n d : ℕ, N ≤ n ∧ d ≤ c * Nat.log 2 n ^ K ∧ (n : ℝ) ^ (2 - ε) ≤ M.cost a n d

end CostModel

/-- **It is the weaker conjecture.**  Dimension `c · log n` is a dimension
`d ≤ c · (log n)^1`. -/
theorem CostModel.OVHardPoly_of_OVHard {M : CostModel} (hM : M.OVHard) :
    M.OVHardPoly := by
  intro ε hε
  obtain ⟨c, hc⟩ := hM ε hε
  refine ⟨c, 1, fun a ha N => ?_⟩
  obtain ⟨n, hn, hcost⟩ := hc a ha N
  exact ⟨n, c * Nat.log 2 n, hn, by rw [pow_one], hcost⟩

/-! ### From SETH, with nothing else assumed -/

/-- **SETH implies the Orthogonal Vectors conjecture at polylogarithmic
dimension.**  The same split-and-enumerate reduction as `OVHard_of_SETH`, run
on the general form of the hypothesis: the `2^n` assignments of each half
become the vectors, and the `m ≤ n^K` clauses become the coordinates, so the
dimension is `(log N)^K` where `N = 2^n` is the number of vectors.

The cost accounting is identical — `two_pow_rpow_le_of_bill` is shared with
`OVHard_of_SETH` — because the dimension is not charged for.  No
sparsification is used, and none is needed. -/
theorem OVHardPoly_of_SETHGeneral {S : SATModel} {M : CostModel}
    (hS : S.SETHGeneral) (R : Reduces S M) : M.OVHardPoly := by
  intro ε hε
  obtain ⟨K, hK⟩ := hS (ε / 4) (by linarith)
  refine ⟨1, K, fun a ha N₀ => ?_⟩
  obtain ⟨n, m, hn, hm, hcost⟩ := hK (R.alg a) (R.solves a ha) (max N₀ ⌈2 / ε⌉₊)
  refine ⟨2 ^ n, m,
    le_trans (le_max_left _ _) (le_trans hn (Nat.le_of_lt Nat.lt_two_pow_self)), ?_, ?_⟩
  · rw [Nat.log_pow (by norm_num) n, one_mul]
    exact hm
  · have hnbig : 2 / ε ≤ (n : ℝ) := by
      have h1 : (⌈2 / ε⌉₊ : ℝ) ≤ (n : ℝ) := by
        exact_mod_cast le_trans (le_max_right N₀ _) hn
      exact le_trans (Nat.le_ceil _) h1
    have hin := R.cost_ge_input a (2 ^ n) m
    have hle := R.cost_le a n m
    refine two_pow_rpow_le_of_bill hε hnbig (le_trans hcost hle) ?_
    push_cast at hin ⊢
    linarith

/-! ### The barrier it yields -/

/-- **Conditional lower bound at polylogarithmic dimension.**  The polylog
counterpart of `query_lower_bound`: build plus `n` queries costs `n^{2-ε}` at
some dimension below `c · (log n)^K`. -/
theorem query_lower_bound_poly {M : CostModel} (hM : M.OVHardPoly) {I : NNIndex}
    (hI : M.Implements I) {ε : ℝ} (hε : 0 < ε) :
    ∃ c K : ℕ, ∀ N : ℕ, ∃ n d : ℕ, N ≤ n ∧ d ≤ c * Nat.log 2 n ^ K ∧
      (n : ℝ) ^ (2 - ε) ≤ I.build n (d + d) + n * I.query n (d + d) := by
  obtain ⟨c, K, hc⟩ := hM ε hε
  refine ⟨c, K, fun N => ?_⟩
  have hsolves : M.Solves hI.alg := fun A B =>
    (hI.decides_reduce A B).trans (I.reduce_iff A B)
  obtain ⟨n, d, hn, hd, hcost⟩ := hc hI.alg hsolves N
  exact ⟨n, d, hn, hd, hcost.trans (hI.cost_le n d)⟩

/-- **The form the bound takes for a cheap index**, at polylogarithmic
dimension. -/
theorem query_ge_of_build_small_poly {M : CostModel} (hM : M.OVHardPoly)
    {I : NNIndex} (hI : M.Implements I) {ε : ℝ} (hε : 0 < ε)
    (hbuild : ∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - ε) / 2) :
    ∃ c K : ℕ, ∀ N : ℕ, ∃ n d : ℕ, N ≤ n ∧ d ≤ c * Nat.log 2 n ^ K ∧
      (n : ℝ) ^ (1 - ε) / 2 ≤ I.query n (d + d) := by
  obtain ⟨c, K, hc⟩ := query_lower_bound_poly hM hI hε
  refine ⟨c, K, fun N => ?_⟩
  obtain ⟨n, d, hn, hd, hmain⟩ := hc (max N 1)
  have hN : N ≤ n := le_trans (le_max_left _ _) hn
  have hn1 : 1 ≤ n := le_trans (le_max_right _ _) hn
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn1
  refine ⟨n, d, hN, hd, ?_⟩
  have hsplit : (n : ℝ) ^ (2 - ε) = (n : ℝ) * (n : ℝ) ^ (1 - ε) := by
    rw [show (2 : ℝ) - ε = 1 + (1 - ε) by ring, Real.rpow_add hn0, Real.rpow_one]
  have hb := hbuild n (d + d)
  have hmul : (n : ℝ) * ((n : ℝ) ^ (1 - ε) / 2) ≤ (n : ℝ) * I.query n (d + d) := by
    rw [show (n : ℝ) * ((n : ℝ) ^ (1 - ε) / 2) = (n : ℝ) * (n : ℝ) ^ (1 - ε) / 2 by ring,
      ← hsplit]
    linarith
  exact le_of_mul_le_mul_left hmul hn0

/-- **The barrier with SETH as the only assumption.**  An exact lookup index
whose preprocessing stays within half the budget spends `n^{1-ε}/2` on some
query at dimension `polylog n`, for arbitrarily large `n` — unless SETH fails.

Compared with `query_ge_of_build_small_of_SETH` this gives up the constant in
the dimension and gains the Sparsification Lemma: the chain from SETH to here
is complete, with no unproved combinatorial input. -/
theorem query_ge_of_build_small_of_SETHGeneral {S : SATModel} {M : CostModel}
    (hS : S.SETHGeneral) (R : Reduces S M) {I : NNIndex} (hI : M.Implements I)
    {ε : ℝ} (hε : 0 < ε)
    (hbuild : ∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - ε) / 2) :
    ∃ c K : ℕ, ∀ N : ℕ, ∃ n d : ℕ, N ≤ n ∧ d ≤ c * Nat.log 2 n ^ K ∧
      (n : ℝ) ^ (1 - ε) / 2 ≤ I.query n (d + d) :=
  query_ge_of_build_small_poly (OVHardPoly_of_SETHGeneral hS R) hI hε hbuild

/-! ### The hypotheses are satisfiable together -/

/-- The naive satisfiability model satisfies the general hypothesis too, its
sparse instances being a special case of its general ones. -/
lemma naiveSATModel_SETHGeneral : naiveSATModel.SETHGeneral :=
  SATModel.SETHGeneral_of_SETH naiveSATModel_SETH

/-- So the polylogarithmic conjecture holds in `naiveModel` — not as evidence,
but as a witness that `OVHardPoly_of_SETHGeneral` has a model in which every
hypothesis holds at once, `naiveModel_implements` supplying the last one. -/
example : naiveModel.OVHardPoly :=
  OVHardPoly_of_SETHGeneral naiveSATModel_SETHGeneral naiveReduces

end ALM
end Transformer
