/-
# Moving the hypothesis back to SETH

`Transformer.ALM.Hardness` assumes `OVHard`, the Orthogonal Vectors
conjecture.  `Transformer.ALM.SAT` proves the combinatorial half of the
reduction that produces it.  This file supplies the arithmetic half, so that
`OVHard` becomes a **theorem** and the assumption moves one level back, to the
Strong Exponential Time Hypothesis:

* Impagliazzo, Paturi, *On the complexity of k-SAT*, JCSS 62 (2001) — SETH,
  and the Sparsification Lemma with Zane.
* R. Williams, *A new algorithm for optimal 2-constraint satisfaction and its
  implications*, Theoret. Comput. Sci. 348 (2005), §4.

The hypothesis cannot be removed, only moved.  SETH is open, and it implies
`P ≠ NP`; no formalization can discharge it.  What this file buys is that the
assumption is now a single, standard, widely studied one instead of a derived
statement about a geometric problem.

`SETH` is stated in the sparsified form — clause count linear in the number of
variables — which is what the reduction consumes.  The Sparsification Lemma
that licenses that form is **not** proved here; it is part of the cited
hypothesis.

One mild convention appears as the field `cost_ge_input`: an algorithm costs
at least the size of its input.  It is what lets the construction cost be
absorbed without an asymptotic argument.
-/

import Transformer.ALM.SAT
import Transformer.ALM.Hardness

namespace Transformer
namespace ALM

/-- The counterpart of `CostModel` for satisfiability: a type of algorithms,
what each decides, and what each costs on a formula with `2n` variables in two
halves of `n` and `m` clauses. -/
structure SATModel where
  /-- The algorithms of the model. -/
  Alg : Type
  /-- What an algorithm answers on a formula. -/
  decides : Alg → ∀ {n m : ℕ}, CNF n m → Prop
  /-- What it costs on two halves of `n` variables and `m` clauses. -/
  cost : Alg → ℕ → ℕ → ℝ

namespace SATModel

variable (S : SATModel)

/-- An algorithm solves satisfiability when its answer is always right. -/
def Solves (a : S.Alg) : Prop :=
  ∀ {n m : ℕ} (φ : CNF n m), S.decides a φ ↔ Satisfiable φ

/-- **The Strong Exponential Time Hypothesis**, in the sparsified form the
reduction consumes: for every `δ > 0` there is a clause-density constant `C`
such that no correct algorithm stays below `2^{(1-δ)·(2n)}` on formulas of
`2n` variables and `C·n` clauses — it exceeds that for arbitrarily large `n`.

The `2n` is the total number of variables, split into two halves of `n`. -/
def SETH : Prop :=
  ∀ δ : ℝ, 0 < δ → ∃ C : ℕ, ∀ a : S.Alg, S.Solves a → ∀ N : ℕ,
    ∃ n, N ≤ n ∧ (2 : ℝ) ^ (2 * (n : ℝ) * (1 - δ)) ≤ S.cost a n (C * n)

end SATModel

/-- The satisfiability model runs Williams' reduction on top of the lookup
model: it enumerates the assignments of each half, builds the two vector
families of `Transformer.ALM.SAT`, and calls the Orthogonal Vectors
algorithm once. -/
structure Reduces (S : SATModel) (M : CostModel) where
  /-- The satisfiability algorithm built from an Orthogonal Vectors one. -/
  alg : M.Alg → S.Alg
  /-- It answers what the constructed instance answers. -/
  decides_reduce : ∀ (a : M.Alg) {n m : ℕ} (φ : CNF n m),
    S.decides (alg a) φ ↔
      M.decides a (fun i => aVec φ (assign n i)) (fun j => bVec φ (assign n j))
  /-- It costs the construction of `2^n` vectors plus one call. -/
  cost_le : ∀ (a : M.Alg) (n m : ℕ),
    S.cost (alg a) n m ≤ (2 : ℝ) ^ n * m + M.cost a (2 ^ n) m
  /-- An algorithm costs at least the size of its input. -/
  cost_ge_input : ∀ (a : M.Alg) (N d : ℕ), (N : ℝ) * d ≤ M.cost a N d

/-- **The reduction is correct**, as a consequence of `satisfiable_iff_ov`
rather than an assumption: if the lookup algorithm answers Orthogonal Vectors
correctly, the algorithm built on it answers satisfiability correctly. -/
theorem Reduces.solves {S : SATModel} {M : CostModel} (R : Reduces S M)
    (a : M.Alg) (ha : M.Solves a) : S.Solves (R.alg a) := by
  intro n m φ
  have : Nonempty (Fin (2 ^ n)) := ⟨⟨0, pow_pos (by norm_num) n⟩⟩
  rw [R.decides_reduce, ha, ← satisfiable_iff_ov]

/-! ### The conjecture becomes a theorem -/

/-- **SETH implies the Orthogonal Vectors conjecture.**  An Orthogonal Vectors
algorithm running in `N^{2-ε}` on `N = 2^n` vectors of dimension `C·n` would
decide satisfiability of a `2n`-variable formula in about `2^{2n(1-ε/2)}`,
which SETH forbids.

The `ε/4` is slack: a quarter is spent beating the exponent SETH names, and
the rest absorbs the cost of writing the `2^n` vectors down. -/
theorem OVHard_of_SETH {S : SATModel} {M : CostModel} (hS : S.SETH)
    (R : Reduces S M) : M.OVHard := by
  intro ε hε
  obtain ⟨C, hC⟩ := hS (ε / 4) (by linarith)
  refine ⟨C, fun a ha N₀ => ?_⟩
  obtain ⟨n, hn, hcost⟩ := hC (R.alg a) (R.solves a ha) (max N₀ ⌈2 / ε⌉₊)
  refine ⟨2 ^ n, le_trans (le_max_left _ _) (le_trans hn (Nat.le_of_lt (Nat.lt_two_pow_self))), ?_⟩
  rw [Nat.log_pow (by norm_num) n]
  -- `n` is large enough that half the slack already beats the construction
  have hnbig : 2 / ε ≤ (n : ℝ) := by
    have h1 : (⌈2 / ε⌉₊ : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast le_trans (le_max_right N₀ _) hn
    exact le_trans (Nat.le_ceil _) h1
  have hslack : 1 ≤ (n : ℝ) * ε / 2 := by
    rw [div_le_iff₀ hε] at hnbig
    linarith
  -- the reduction's cost, with the construction absorbed into the call
  have hin := R.cost_ge_input a (2 ^ n) (C * n)
  have hle := R.cost_le a n (C * n)
  have habs : (2 : ℝ) ^ (2 * (n : ℝ) * (1 - ε / 4)) ≤ 2 * M.cost a (2 ^ n) (C * n) := by
    push_cast at hin hle
    linarith
  -- and the exponent SETH names beats the one the conclusion asks for
  have hexp : (n : ℝ) * (2 - ε) ≤ 2 * (n : ℝ) * (1 - ε / 4) - 1 := by nlinarith
  have hmono : (2 : ℝ) ^ ((n : ℝ) * (2 - ε))
      ≤ (2 : ℝ) ^ (2 * (n : ℝ) * (1 - ε / 4) - 1) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) hexp
  have hhalf : (2 : ℝ) ^ (2 * (n : ℝ) * (1 - ε / 4) - 1)
      = (2 : ℝ) ^ (2 * (n : ℝ) * (1 - ε / 4)) / 2 := by
    rw [Real.rpow_sub (by norm_num), Real.rpow_one]
  have hpow : ((2 ^ n : ℕ) : ℝ) ^ (2 - ε) = (2 : ℝ) ^ ((n : ℝ) * (2 - ε)) := by
    push_cast
    rw [← Real.rpow_natCast 2 n, ← Real.rpow_mul (by norm_num)]
  rw [hpow]
  rw [hhalf] at hmono
  linarith

/-- **The barrier, with SETH as the only conjecture.**  Composing with
`query_ge_of_build_small` of `Transformer.ALM.Hardness`: an exact lookup index
whose preprocessing stays within half the budget must spend `n^{1-ε}/2` per
query at dimension `Θ(log n)`, for arbitrarily large `n` — unless SETH fails.

This is as far as the chain can be pushed.  `hS` cannot be discharged: SETH is
open, and implies `P ≠ NP`. -/
theorem query_ge_of_build_small_of_SETH {S : SATModel} {M : CostModel}
    (hS : S.SETH) (R : Reduces S M) {I : NNIndex} (hI : M.Implements I)
    {ε : ℝ} (hε : 0 < ε)
    (hbuild : ∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - ε) / 2) :
    ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
      (n : ℝ) ^ (1 - ε) / 2 ≤
        I.query n (c * Nat.log 2 n + c * Nat.log 2 n) :=
  query_ge_of_build_small (OVHard_of_SETH hS R) hI hε hbuild

/-! ### The hypotheses are satisfiable together -/

/-- The satisfiability model obtained by running the reduction on top of
`naiveModel`: the quadratic scan over all pairs of assignments. -/
noncomputable def naiveSATModel : SATModel where
  Alg := Unit
  decides := fun a => fun φ =>
    naiveModel.decides a (fun i => aVec φ (assign _ i)) (fun j => bVec φ (assign _ j))
  cost := fun _ n m => (2 : ℝ) ^ n * m + 2 * ((2 ^ n : ℕ) : ℝ) ^ 2 * m

/-- SETH holds in `naiveSATModel` — trivially, its only algorithm being the
exponential one.  This is not evidence for SETH; it witnesses that the
hypothesis of `OVHard_of_SETH` is satisfiable. -/
example : naiveSATModel.SETH := by
  intro δ hδ
  refine ⟨1, fun _ _ N => ⟨max N 1, le_max_left _ _, ?_⟩⟩
  set n := max N 1 with hn
  have hn1 : 1 ≤ n := le_max_right _ _
  have hnR : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn1
  have h2n : (0 : ℝ) < (2 : ℝ) ^ n := by positivity
  have hstep : (2 : ℝ) ^ (2 * (n : ℝ) * (1 - δ)) ≤ (2 : ℝ) ^ (2 * (n : ℝ)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by nlinarith)
  have hsq : (2 : ℝ) ^ (2 * (n : ℝ)) = ((2 : ℝ) ^ n) ^ 2 := by
    rw [show (2 : ℝ) * (n : ℝ) = (n : ℝ) * 2 by ring, Real.rpow_mul (by norm_num),
      Real.rpow_natCast, show ((2 : ℝ) : ℝ) = ((2 : ℕ) : ℝ) by norm_num,
      Real.rpow_natCast]
  show (2 : ℝ) ^ (2 * (n : ℝ) * (1 - δ)) ≤ (2 : ℝ) ^ n * _ + 2 * ((2 ^ n : ℕ) : ℝ) ^ 2 * _
  push_cast
  have hmR : (1 : ℝ) ≤ ((1 * n : ℕ) : ℝ) := by rw [one_mul]; exact_mod_cast hn1
  push_cast at hmR
  nlinarith [hstep, hsq, h2n, sq_nonneg ((2 : ℝ) ^ n)]

/-- And the reduction is available between the two naive models, so all the
hypotheses of `OVHard_of_SETH` hold at once. -/
example : Reduces naiveSATModel naiveModel where
  alg := id
  decides_reduce := fun _ => fun _ => Iff.rfl
  cost_le := fun _ n m => by
    show (2 : ℝ) ^ n * m + 2 * ((2 ^ n : ℕ) : ℝ) ^ 2 * m
      ≤ (2 : ℝ) ^ n * m + 2 * ((2 ^ n : ℕ) : ℝ) ^ 2 * m
    exact le_refl _
  cost_ge_input := fun _ N d => by
    show (N : ℝ) * d ≤ 2 * (N : ℝ) ^ 2 * d
    rcases Nat.eq_zero_or_pos N with rfl | hN
    · simp
    · have h1 : (1 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
      have hd : (0 : ℝ) ≤ d := Nat.cast_nonneg d
      have hNd : (0 : ℝ) ≤ (N : ℝ) * d := mul_nonneg (by linarith) hd
      have hkey : (0 : ℝ) ≤ (N : ℝ) * d * ((N : ℝ) - 1) := mul_nonneg hNd (by linarith)
      nlinarith [hNd, hkey]

end ALM
end Transformer
