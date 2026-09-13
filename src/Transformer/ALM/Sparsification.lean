/-
# Where SETH is really assumed

`Transformer.ALM.SETH` states the hypothesis in the *sparsified* form — clause
count linear in the number of variables — because that is the form Williams'
reduction consumes.  That conflates two separate assumptions: the Strong
Exponential Time Hypothesis itself, which says nothing about clause density,
and the Sparsification Lemma, which is what licenses passing to the sparse
form.

This file separates them.  `SATModel.SETHGeneral` is the hypothesis with no
density restriction, `Sparsification` is the lemma stated as an interface, and
`SETH_of_general` derives the sparse form from the two.  After this, the only
assumption of the chain that is about *hardness* is SETH as ordinarily stated.

* Impagliazzo, Paturi, *On the complexity of k-SAT*, JCSS 62 (2001) — SETH.
* Impagliazzo, Paturi, Zane, *Which problems have strongly exponential
  complexity?*, JCSS 63 (2001), Lemma 1 — the Sparsification Lemma: a `k`-CNF
  on `n` variables is a union of `2^{εn}` `k`-CNFs of at most `c(k, ε)·n`
  clauses each, computable in time `2^{εn}·poly`.

The Sparsification Lemma is still assumed rather than proved — it is a
combinatorial theorem about a clause-splitting recursion, and formalizing it
is a separate project — but it is now assumed *as itself*, with its cost
accounting visible, instead of being folded silently into the statement of
SETH.

`Transformer.ALM.SparseModel` carries the model witnessing that the two
hypotheses are satisfiable together.
-/

import Transformer.ALM.SETH

open Filter

namespace Transformer
namespace ALM

/-! ### A polynomial is eventually below any exponential

The one analytic ingredient.  General SETH bounds the clause count of its hard
instances by a polynomial; the sparsification overhead has to swallow that,
and it can, because the slack left over is exponential. -/

/-- For every degree `K` and every rate `c > 0` there is a threshold past which
`n^K + 1 ≤ 2^{c·n}`. -/
theorem exists_pow_add_one_le_rpow (K : ℕ) {c : ℝ} (hc : 0 < c) :
    ∃ N : ℕ, ∀ n : ℕ, N ≤ n → (n : ℝ) ^ K + 1 ≤ (2 : ℝ) ^ (c * (n : ℝ)) := by
  set r : ℝ := (2 : ℝ) ^ c with hrdef
  have hr : 1 < r := by
    rw [hrdef]
    exact Real.one_lt_rpow_iff_of_pos (by norm_num) |>.mpr (Or.inl ⟨by norm_num, hc⟩)
  have hpow : ∀ n : ℕ, r ^ n = (2 : ℝ) ^ (c * (n : ℝ)) := by
    intro n
    rw [hrdef, ← Real.rpow_natCast ((2 : ℝ) ^ c) n, ← Real.rpow_mul (by norm_num)]
  have h1 : ∀ᶠ n : ℕ in atTop, ‖((n : ℝ)) ^ K‖ ≤ (1 / 2) * ‖r ^ n‖ :=
    (isLittleO_pow_const_const_pow_of_one_lt (R := ℝ) K hr).def (by norm_num)
  have h2 : ∀ᶠ n : ℕ in atTop, (2 : ℝ) ≤ r ^ n :=
    (tendsto_pow_atTop_atTop_of_one_lt hr).eventually_ge_atTop 2
  obtain ⟨N, hN⟩ := eventually_atTop.mp (h1.and h2)
  refine ⟨N, fun n hn => ?_⟩
  obtain ⟨ha, hb⟩ := hN n hn
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (by positivity),
    abs_of_nonneg (by positivity)] at ha
  rw [← hpow]
  linarith

/-! ### The hypothesis without a density restriction -/

namespace SATModel

variable (S : SATModel)

/-- **SETH as ordinarily stated**: for every `δ > 0`, no correct algorithm
stays below `2^{(1-δ)·(2n)}` on formulas of `2n` variables.  The clause count
is not restricted to be linear, only polynomial — which is automatic for the
`k`-CNF the hypothesis is about, where `m ≤ (2n)^k`.

`SETH` of `Transformer.ALM.SETH` is the same statement with `m` pinned to
`C·n`; that extra strength is exactly what the Sparsification Lemma
supplies. -/
def SETHGeneral : Prop :=
  ∀ δ : ℝ, 0 < δ → ∃ K : ℕ, ∀ a : S.Alg, S.Solves a → ∀ N : ℕ,
    ∃ n m : ℕ, N ≤ n ∧ m ≤ n ^ K ∧ (2 : ℝ) ^ (2 * (n : ℝ) * (1 - δ)) ≤ S.cost a n m

end SATModel

/-- **The sparse hypothesis is the stronger one.**  Pinning the clause count
to `C·n` is a restriction on the instances the hypothesis speaks about, so it
asserts more; dropping it can only weaken the statement.  `C·n ≤ n^{C+2}` once
`n` is past `C`, which is all the bookkeeping the direction needs. -/
theorem SATModel.SETHGeneral_of_SETH {S : SATModel} (hS : S.SETH) : S.SETHGeneral := by
  intro δ hδ
  obtain ⟨C, hC⟩ := hS δ hδ
  refine ⟨C + 2, fun a ha N => ?_⟩
  obtain ⟨n, hn, hcost⟩ := hC a ha (max N (C + 1))
  have hbig : C + 1 ≤ n := le_trans (le_max_right N (C + 1)) hn
  refine ⟨n, C * n, le_trans (le_max_left _ _) hn, ?_, hcost⟩
  calc C * n ≤ n * n := Nat.mul_le_mul_right n (by omega)
    _ = n ^ 2 := by ring
    _ ≤ n ^ (C + 2) := Nat.pow_le_pow_right (by omega) (by omega)

/-- **The Sparsification Lemma**, as an interface.  From an algorithm `a` good
on sparse formulas it builds, for each `ε > 0`, an algorithm good on all of
them: split the formula into `2^{εn}` pieces of at most `dens ε · n` clauses
and run `a` on each.  The cost is `2^{εn}` times the cost of one piece, plus
the polynomial cost of producing the pieces, for which `n·m` stands. -/
structure Sparsification (S : SATModel) where
  /-- The clause density `c(k, ε)` the lemma produces. -/
  dens : ℝ → ℕ
  /-- The algorithm for general formulas built from one for sparse formulas. -/
  alg : ℝ → S.Alg → S.Alg
  /-- Splitting preserves satisfiability, so the built algorithm is correct. -/
  solves : ∀ (ε : ℝ) (a : S.Alg), S.Solves a → S.Solves (alg ε a)
  /-- `2^{εn}` pieces, each of `dens ε · n` clauses. -/
  cost_le : ∀ ε : ℝ, 0 < ε → ∀ (a : S.Alg) (n m : ℕ),
    S.cost (alg ε a) n m
      ≤ (2 : ℝ) ^ (ε * (n : ℝ)) * ((n : ℝ) * m + S.cost a n (dens ε * n))

/-- **The sparse hypothesis follows from the general one.**  An algorithm fast
on formulas of linear density would, run on every piece of a sparsification at
rate `ε = δ/4`, be fast on all formulas: the `2^{εn}` pieces cost a factor
`2^{δn/4}`, and the polynomial cost of producing them is below the exponential
slack that remains.  So SETH in the form `Transformer.ALM.SETH` consumes is
not an extra assumption beyond SETH plus sparsification. -/
theorem SATModel.SETH_of_general {S : SATModel} (hS : S.SETHGeneral)
    (P : Sparsification S) : S.SETH := by
  intro δ hδ
  -- the conclusion only weakens as `δ` grows, so it is enough to prove it at `δ ≤ 1`
  set d : ℝ := min δ 1 with hddef
  have hd0 : 0 < d := lt_min hδ one_pos
  have hd1 : d ≤ 1 := min_le_right _ _
  have hdδ : d ≤ δ := min_le_left _ _
  obtain ⟨K, hK⟩ := hS (d / 2) (by positivity)
  obtain ⟨N₁, hN₁⟩ := exists_pow_add_one_le_rpow (K + 1) (show (0 : ℝ) < 3 * d / 4 by positivity)
  refine ⟨P.dens (d / 4), fun a ha N₀ => ?_⟩
  obtain ⟨n, m, hn, hm, hcost⟩ := hK (P.alg (d / 4) a) (P.solves _ a ha) (max N₀ N₁)
  refine ⟨n, le_trans (le_max_left _ _) hn, ?_⟩
  set x : ℝ := (n : ℝ) with hxdef
  have hx : (0 : ℝ) ≤ x := Nat.cast_nonneg n
  -- the sparsification's own bill
  have hA : (2 : ℝ) ^ (2 * x * (1 - d / 2))
      ≤ (2 : ℝ) ^ (d / 4 * x) * (x * m + S.cost a n (P.dens (d / 4) * n)) :=
    le_trans hcost (P.cost_le (d / 4) (by positivity) a n m)
  have hsplit : (2 : ℝ) ^ (d / 4 * x)
        * ((2 : ℝ) ^ (2 * x * (1 - d)) * (2 : ℝ) ^ (3 * d / 4 * x))
      = (2 : ℝ) ^ (2 * x * (1 - d / 2)) := by
    rw [← Real.rpow_add (by norm_num : (0:ℝ) < 2), ← Real.rpow_add (by norm_num : (0:ℝ) < 2)]
    congr 1
    ring
  have hcancel : (2 : ℝ) ^ (2 * x * (1 - d)) * (2 : ℝ) ^ (3 * d / 4 * x)
      ≤ x * m + S.cost a n (P.dens (d / 4) * n) :=
    le_of_mul_le_mul_left (by rw [hsplit]; exact hA) (Real.rpow_pos_of_pos (by norm_num) _)
  -- the construction cost is polynomial, the slack exponential
  have hnm : x * (m : ℝ) ≤ x ^ (K + 1) := by
    have hmx : (m : ℝ) ≤ x ^ K := by rw [hxdef]; exact_mod_cast hm
    calc x * (m : ℝ) ≤ x * x ^ K := mul_le_mul_of_nonneg_left hmx hx
      _ = x ^ (K + 1) := by ring
  have hQ : x * m + 1 ≤ (2 : ℝ) ^ (3 * d / 4 * x) := by
    have := hN₁ n (le_trans (le_max_right N₀ N₁) hn)
    rw [show 3 * d / 4 * x = 3 * d / 4 * (n : ℝ) by rw [hxdef]]
    linarith
  have hP1 : (1 : ℝ) ≤ (2 : ℝ) ^ (2 * x * (1 - d)) := by
    have := Real.rpow_le_rpow_of_exponent_le (by norm_num : (1:ℝ) ≤ 2)
      (show (0 : ℝ) ≤ 2 * x * (1 - d) by nlinarith)
    simpa using this
  have hxm : (0 : ℝ) ≤ x * m := by positivity
  have hstep : (2 : ℝ) ^ (2 * x * (1 - d)) * (x * m + 1)
      ≤ (2 : ℝ) ^ (2 * x * (1 - d)) * (2 : ℝ) ^ (3 * d / 4 * x) :=
    mul_le_mul_of_nonneg_left hQ (by positivity)
  have hmul : x * m ≤ (2 : ℝ) ^ (2 * x * (1 - d)) * (x * m) := by
    have := mul_nonneg (by linarith : (0 : ℝ) ≤ (2 : ℝ) ^ (2 * x * (1 - d)) - 1) hxm
    linarith
  have hY : (2 : ℝ) ^ (2 * x * (1 - d)) ≤ S.cost a n (P.dens (d / 4) * n) := by linarith
  exact le_trans (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by nlinarith)) hY

end ALM
end Transformer
