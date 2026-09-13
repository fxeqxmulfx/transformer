/-
# Why exact lookup is believed hard in high dimension

The claim that no exact nearest-neighbour index can have near-linear space and
polylogarithmic query time in dimension `d = Θ(log n)` is **not** a theorem.
It is a consequence of the Orthogonal Vectors conjecture, which in turn
follows from the Strong Exponential Time Hypothesis:

* Impagliazzo, Paturi, *On the complexity of k-SAT*, JCSS 62 (2001) — SETH.
* R. Williams, *A new algorithm for optimal 2-constraint satisfaction and its
  implications*, Theoret. Comput. Sci. 348 (2005) — SETH implies Orthogonal
  Vectors needs `n^{2-o(1)}` time.
* V. Vassilevska Williams, *On some fine-grained questions in algorithms and
  complexity*, ICM 2018, §3.

Formalizing SETH itself would need a time-bounded model of computation, which
Mathlib does not provide and which this repository has no business building.
What *can* be stated honestly is the implication, with the conjecture carried
as a hypothesis rather than an axiom, exactly as the conventions of this
repository require.

`CostModel` is the minimum needed to phrase the hypothesis: a type of
algorithms, what each one decides, and what each one costs.  `OVHard` is the
conjecture in that model.  `Implements` says the model actually contains the
reduction of `Transformer.ALM.LookupIndex`.  From those three,
`query_lower_bound` is a theorem — and the final `example` exhibits a model
where all three hold at once, so the implication is not vacuous.

The dimension in the conjecture is `c · log n`, not a constant.  That is the
honest statement, and it is why the barrier says nothing about the planar
hull the machine actually uses: for fixed `d` the hypothesis is silent.
-/

import Transformer.ALM.LookupIndex

namespace Transformer
namespace ALM

/-- Just enough of a model of computation to state the hypothesis: a type of
algorithms, the predicate each one decides on an Orthogonal Vectors instance,
and the cost each one incurs on `n` vectors in dimension `d`. -/
structure CostModel where
  /-- The algorithms of the model. -/
  Alg : Type
  /-- What an algorithm answers on an instance. -/
  decides : Alg → ∀ {d n : ℕ}, (Fin n → BVec d) → (Fin n → BVec d) → Prop
  /-- What it costs on `n` vectors in dimension `d`. -/
  cost : Alg → ℕ → ℕ → ℝ

namespace CostModel

variable (M : CostModel)

/-- An algorithm solves Orthogonal Vectors when its answer is the right one on
every nonempty instance. -/
def Solves (a : M.Alg) : Prop :=
  ∀ {d n : ℕ} [Nonempty (Fin n)] (A B : Fin n → BVec d),
    M.decides a A B ↔ ∃ i j, Orth (A i) (B j)

/-- **The Orthogonal Vectors conjecture**, as a hypothesis about the model:
for every `ε > 0` there is a dimension constant `c` such that no correct
algorithm stays below `n^{2-ε}` on instances of dimension `c · log₂ n` — it
exceeds it for arbitrarily large `n`.

Note the quantifier order: the dimension grows with `n`.  The conjecture makes
no claim at any fixed dimension. -/
def OVHard : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ c : ℕ, ∀ a : M.Alg, M.Solves a → ∀ N : ℕ,
    ∃ n, N ≤ n ∧ (n : ℝ) ^ (2 - ε) ≤ M.cost a n (c * Nat.log 2 n)

/-- The model contains the reduction: some algorithm of `M` answers exactly
what querying `I` once per left-hand vector answers, and costs no more than
that procedure's accounted cost — one build plus `n` queries in the doubled
dimension. -/
structure Implements (I : NNIndex) where
  /-- The algorithm carrying out the reduction. -/
  alg : M.Alg
  /-- It answers what the reduction answers. -/
  decides_reduce : ∀ {d n : ℕ} [Nonempty (Fin n)] (A B : Fin n → BVec d),
    M.decides alg A B ↔ I.reduce A B
  /-- And it costs no more than one build plus `n` queries. -/
  cost_le : ∀ n d : ℕ,
    M.cost alg n d ≤ I.build n (d + d) + n * I.query n (d + d)

end CostModel

/-! ### The conditional lower bound -/

/-- **Conditional lower bound on exact lookup.**  In a model where Orthogonal
Vectors is hard and the reduction is available, the total cost of an exact
index — one build plus `n` queries at dimension `Θ(log n)` — is at least
`n^{2-ε}` for arbitrarily large `n`.

The conclusion is about the *sum*: the index may spend the budget on
preprocessing or on queries, but not escape it. -/
theorem query_lower_bound {M : CostModel} (hM : M.OVHard) {I : NNIndex}
    (hI : M.Implements I) {ε : ℝ} (hε : 0 < ε) :
    ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
      (n : ℝ) ^ (2 - ε) ≤
        I.build n (c * Nat.log 2 n + c * Nat.log 2 n)
          + n * I.query n (c * Nat.log 2 n + c * Nat.log 2 n) := by
  obtain ⟨c, hc⟩ := hM ε hε
  refine ⟨c, fun N => ?_⟩
  have hsolves : M.Solves hI.alg := fun A B =>
    (hI.decides_reduce A B).trans (I.reduce_iff A B)
  obtain ⟨n, hn, hcost⟩ := hc hI.alg hsolves N
  exact ⟨n, hn, hcost.trans (hI.cost_le n _)⟩

/-- **The form the bound takes for a cheap index.**  If preprocessing is at
most half the budget, the query cost alone is at least `n^{1-ε}/2` for
arbitrarily large `n`: no polylogarithmic query survives at dimension
`Θ(log n)`. -/
theorem query_ge_of_build_small {M : CostModel} (hM : M.OVHard) {I : NNIndex}
    (hI : M.Implements I) {ε : ℝ} (hε : 0 < ε)
    (hbuild : ∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - ε) / 2) :
    ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
      (n : ℝ) ^ (1 - ε) / 2 ≤
        I.query n (c * Nat.log 2 n + c * Nat.log 2 n) := by
  obtain ⟨c, hc⟩ := query_lower_bound hM hI hε
  refine ⟨c, fun N => ?_⟩
  obtain ⟨n, hn, hmain⟩ := hc (max N 1)
  have hN : N ≤ n := le_trans (le_max_left _ _) hn
  have hn1 : 1 ≤ n := le_trans (le_max_right _ _) hn
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn1
  refine ⟨n, hN, ?_⟩
  have hsplit : (n : ℝ) ^ (2 - ε) = (n : ℝ) * (n : ℝ) ^ (1 - ε) := by
    rw [show (2 : ℝ) - ε = 1 + (1 - ε) by ring, Real.rpow_add hn0, Real.rpow_one]
  have hb := hbuild n (c * Nat.log 2 n + c * Nat.log 2 n)
  have hmul : (n : ℝ) * ((n : ℝ) ^ (1 - ε) / 2)
      ≤ (n : ℝ) * I.query n (c * Nat.log 2 n + c * Nat.log 2 n) := by
    rw [show (n : ℝ) * ((n : ℝ) ^ (1 - ε) / 2) = (n : ℝ) * (n : ℝ) ^ (1 - ε) / 2 by ring,
      ← hsplit]
    linarith
  exact le_of_mul_le_mul_left hmul hn0

/-! ### The hypotheses are satisfiable together -/

/-- A model with one algorithm, the quadratic scan over all pairs, charged
`2n²d`. -/
noncomputable def naiveModel : CostModel where
  Alg := Unit
  decides := fun _ => fun A B => ∃ i j, Orth (A i) (B j)
  cost := fun _ n d => 2 * (n : ℝ) ^ 2 * (d : ℝ)

/-- Orthogonal Vectors is "hard" in `naiveModel` — trivially, since its only
algorithm is the quadratic one.  This is not evidence for the conjecture; it
witnesses that `OVHard` is satisfiable, so `query_lower_bound` is not vacuous
on that side. -/
lemma naiveModel_OVHard : naiveModel.OVHard := by
  intro ε hε
  refine ⟨1, fun _ _ N => ⟨max N 2, le_max_left _ _, ?_⟩⟩
  have hn2 : 2 ≤ max N 2 := le_max_right _ _
  have hlog : 1 ≤ Nat.log 2 (max N 2) := Nat.log_pos (by norm_num) hn2
  have hlogR : (1 : ℝ) ≤ ((1 * Nat.log 2 (max N 2) : ℕ) : ℝ) := by
    rw [one_mul]; exact_mod_cast hlog
  have hnR : (1 : ℝ) ≤ ((max N 2 : ℕ) : ℝ) := by
    have : 1 ≤ max N 2 := le_trans (by norm_num) hn2
    exact_mod_cast this
  have hpow : ((max N 2 : ℕ) : ℝ) ^ (2 - ε) ≤ ((max N 2 : ℕ) : ℝ) ^ 2 := by
    calc ((max N 2 : ℕ) : ℝ) ^ (2 - ε)
        ≤ ((max N 2 : ℕ) : ℝ) ^ (2 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le hnR (by linarith)
      _ = ((max N 2 : ℕ) : ℝ) ^ 2 := by
          rw [show (2 : ℝ) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have hsq : (0 : ℝ) ≤ ((max N 2 : ℕ) : ℝ) ^ 2 := sq_nonneg _
  show ((max N 2 : ℕ) : ℝ) ^ (2 - ε) ≤ 2 * ((max N 2 : ℕ) : ℝ) ^ 2 * _
  nlinarith [hpow, hsq, hlogR]

/-- And `naiveModel` implements the reduction through `bruteForce`, at exactly
the accounted cost.  So all three hypotheses of `query_lower_bound` hold
simultaneously — the theorem has content. -/
noncomputable def naiveModel_implements : naiveModel.Implements bruteForce where
  alg := ()
  decides_reduce := fun A B => (bruteForce.reduce_iff A B).symm
  cost_le := fun n d => by
    show 2 * (n : ℝ) ^ 2 * (d : ℝ) ≤ 0 + (n : ℝ) * ((n : ℝ) * (((d + d : ℕ)) : ℝ))
    push_cast
    ring_nf
    exact le_refl _

end ALM
end Transformer
