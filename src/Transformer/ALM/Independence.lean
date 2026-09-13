/-
# Why the hypothesis cannot be dropped

`Transformer.ALM.SETH` and `Transformer.ALM.Polylog` reduce the barrier to a
single conjecture, and the docstrings there say the conjecture cannot be
removed, only moved.  That is prose.  This file makes it a theorem.

Every statement below is about the *interfaces* `SATModel` and `CostModel`:
their fields say what an algorithm decides and what it costs, and say nothing
about the two being related.  So there is a model in which nothing costs
anything — an oracle that answers instantly — and in it SETH is false, the
Orthogonal Vectors conjecture is false, and the query lower bound fails.
Alongside the naive models of `Transformer.ALM.SETH`, where all three hold,
that settles the matter: none of them is derivable, and none is refutable,
from the interface alone.

`query_ge_needs_OVHard` is the sharp form.  It exhibits a model and an index
satisfying *every* hypothesis of `query_ge_of_build_small` except `hM`, with
the conclusion false — so `hM` is not decoration that a cleverer proof could
discharge, it is doing the work.

None of this is evidence about SETH.  SETH is a statement about Turing
machines, and it implies `P ≠ NP` (Impagliazzo, Paturi, *On the complexity of
k-SAT*, JCSS 62 (2001)); a `SATModel` is far too coarse to settle it either
way, which is exactly why it is carried as a hypothesis.  What the file rules
out is the hope that the formalization contains a hidden proof, or that the
hypothesis could be weakened away by rearranging the chain.

* R. Williams, *A new algorithm for optimal 2-constraint satisfaction and its
  implications*, Theoret. Comput. Sci. 348 (2005), §4 — the chain whose first
  link is shown here to be load-bearing.
-/

import Transformer.ALM.Polylog

namespace Transformer
namespace ALM

/-! ### A model in which everything is free

Charging nothing is not a model of *time*, and it is not offered as one.  It
is a model of the interface: it inhabits every field of `SATModel` and of
`CostModel`, and answers correctly.  That is all a counterexample needs. -/

/-- Satisfiability answered by oracle, at no cost. -/
def freeSATModel : SATModel where
  Alg := Unit
  decides := fun _ => fun φ => Satisfiable φ
  cost := fun _ _ _ => 0

/-- Its one algorithm is correct — that is what makes it a counterexample
rather than a model excluded by `Solves`. -/
lemma freeSATModel_solves : freeSATModel.Solves () := fun _ => Iff.rfl

/-- Orthogonal Vectors answered by oracle, at no cost. -/
def freeModel : CostModel where
  Alg := Unit
  decides := fun _ => fun A B => ∃ i j, Orth (A i) (B j)
  cost := fun _ _ _ => 0

/-- And it too is correct. -/
lemma freeModel_solves : freeModel.Solves () := fun _ _ => Iff.rfl

/-! ### The conjectures fail there -/

/-- **SETH is false in `freeSATModel`.**  At `δ = 1/2` its bound is `2^n`,
and the model's cost is zero. -/
theorem freeSATModel_not_SETH : ¬ freeSATModel.SETH := by
  intro h
  obtain ⟨C, hC⟩ := h (1 / 2) (by norm_num)
  obtain ⟨n, _, hcost⟩ := hC () freeSATModel_solves 0
  have hpos : (0 : ℝ) < (2 : ℝ) ^ (2 * (n : ℝ) * (1 - 1 / 2)) :=
    Real.rpow_pos_of_pos (by norm_num) _
  rw [show freeSATModel.cost () n (C * n) = 0 from rfl] at hcost
  linarith

/-- The same for the form without a density restriction: dropping a
restriction cannot rescue a hypothesis that already fails. -/
theorem freeSATModel_not_SETHGeneral : ¬ freeSATModel.SETHGeneral := by
  intro h
  obtain ⟨K, hK⟩ := h 1 one_pos
  obtain ⟨n, m, _, _, hcost⟩ := hK () freeSATModel_solves 0
  exact absurd hcost (by simp [freeSATModel])

/-- **Orthogonal Vectors is easy in `freeModel`.**  Taking `ε = 1`, the
conjecture would demand `n ≤ 0` for some `n ≥ 1`. -/
theorem freeModel_not_OVHard : ¬ freeModel.OVHard := by
  intro h
  obtain ⟨c, hc⟩ := h 1 one_pos
  obtain ⟨n, hn, hcost⟩ := hc () freeModel_solves 1
  have hn0 : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hpos : (0 : ℝ) < (n : ℝ) ^ (2 - 1 : ℝ) := Real.rpow_pos_of_pos hn0 _
  exact absurd hcost (by simpa [freeModel] using not_le.mpr hpos)

/-- And so is its polylogarithmic weakening, by `OVHardPoly_of_OVHard`
contraposed. -/
theorem freeModel_not_OVHardPoly : ¬ freeModel.OVHardPoly := fun h => by
  obtain ⟨c, K, hc⟩ := h 1 one_pos
  obtain ⟨n, d, hn, _, hcost⟩ := hc () freeModel_solves 1
  have hn0 : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hpos : (0 : ℝ) < (n : ℝ) ^ (2 - 1 : ℝ) := Real.rpow_pos_of_pos hn0 _
  exact absurd hcost (by simpa [freeModel] using not_le.mpr hpos)

/-! ### Independence -/

/-- **SETH is neither provable nor refutable from the interface.**  One model
satisfies it, another falsifies it; no argument that quantifies over
`SATModel` can settle it.

The witness on the positive side is `naiveSATModel`, whose only algorithm is
the exponential one — again no evidence about SETH, only about the shape of
the interface. -/
theorem SETH_independent :
    (∃ S : SATModel, S.SETH) ∧ ∃ S : SATModel, ¬ S.SETH :=
  ⟨⟨naiveSATModel, naiveSATModel_SETH⟩, ⟨freeSATModel, freeSATModel_not_SETH⟩⟩

/-- The same for the Orthogonal Vectors conjecture, one level down the
chain. -/
theorem OVHard_independent :
    (∃ M : CostModel, M.OVHard) ∧ ∃ M : CostModel, ¬ M.OVHard :=
  ⟨⟨naiveModel, naiveModel_OVHard⟩, ⟨freeModel, freeModel_not_OVHard⟩⟩

/-! ### The hypothesis is load-bearing -/

/-- The exhaustive index with its costs zeroed out.  It still returns a true
argmax — `ans` and `ans_isGreatest` are untouched — so it is a legitimate
`NNIndex`; only its declared prices are fictional. -/
noncomputable def freeIndex : NNIndex :=
  { bruteForce with build := fun _ _ => 0, query := fun _ _ => 0 }

/-- `freeModel` implements it, at the accounted cost of zero. -/
noncomputable def freeModel_implements : freeModel.Implements freeIndex where
  alg := ()
  decides_reduce := fun A B => (freeIndex.reduce_iff A B).symm
  cost_le := fun n d => by simp [freeModel, freeIndex]

/-- **`hM` is doing the work.**  Here is a model and an exact index for which
every other hypothesis of `query_ge_of_build_small` holds — the reduction is
implemented, preprocessing is well within half the budget — and the
conclusion is false at every dimension, because the queries are free.

So the Orthogonal Vectors hypothesis cannot be weakened away or derived from
the rest: remove it and the theorem is false.  By `OVHard_of_SETH`, the same
verdict passes back to SETH. -/
theorem query_ge_needs_OVHard :
    ∃ (M : CostModel) (I : NNIndex), Nonempty (M.Implements I) ∧
      (∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - (1 / 2 : ℝ)) / 2) ∧
      ¬ ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
        (n : ℝ) ^ (1 - (1 / 2 : ℝ)) / 2
          ≤ I.query n (c * Nat.log 2 n + c * Nat.log 2 n) := by
  refine ⟨freeModel, freeIndex, ⟨freeModel_implements⟩, fun n d => ?_, ?_⟩
  · exact le_of_eq_of_le rfl
      (div_nonneg (Real.rpow_nonneg (Nat.cast_nonneg n) _) (by norm_num))
  · rintro ⟨c, hc⟩
    obtain ⟨n, hn, hq⟩ := hc 1
    have hn0 : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
    have hpos : (0 : ℝ) < (n : ℝ) ^ (1 - (1 / 2 : ℝ)) := Real.rpow_pos_of_pos hn0 _
    exact absurd hq (by simpa [freeIndex] using not_le.mpr (by linarith))

end ALM
end Transformer
