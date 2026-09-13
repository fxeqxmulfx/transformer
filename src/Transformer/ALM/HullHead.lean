/-
# The tie the head averages is the tie the hull found

`Transformer.ALM.SoftmaxTie` proves that a softmax head returns the centroid of
a level set of its scores, and instantiates that at two keys carrying the same
score.  Which two, it does not say: `hb : s b = σ` and `hc : s c = σ` are
handed to it.  `Transformer.ALM.HullCost` proves the other half — the walk of
`HullHalf::query` collects the winner and at most one neighbour, and `resolve`
averages exactly those two.  So one file knew *that* a tie is averaged and the
other knew *where* the ties are, and nothing said they are the same tie.

`score_eq_of_mem_argmaxSet` is the missing step, and it is short: two indices
of `argmaxSet` are tied *as lines* (`argmaxSet_tie`), and in dimension one a
line value at `q` is a score (`score_eq_lineEval`), so any key the walk
collects scores exactly what the winner scores.  `hullTie_head_resolves` then
needs no hypothesis about scores at all: the geometry the search landed in
supplies it, and the head's output sits within `ε·C` of the aggregate the walk
built — the machine's `TieBreak::AVERAGE` answer
(`transformer_vm/attention/hull2d_cht.h`, lines 70-83, 276-306).
-/

import Transformer.ALM.SoftmaxTie
import Transformer.ALM.HullCost

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **What the walk collects, the head cannot tell apart.**  A key whose scalar
value is one of the maximizing positions of the sorted array scores exactly
what the key at `hullProbe` scores: `argmaxSet_tie` at the level of lines,
`score_eq_lineEval` to read it as a score. -/
theorem score_eq_of_mem_argmaxSet [Nonempty (Fin n)] (Kv : Fin n → EucSpace 1)
    (qv : EucSpace 1) (i : Fin n) (c : ℕ)
    (hc : c ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (h : Kv i 0 = sortedKey (fun j => Kv j 0) c) :
    score qv (Kv i) = score qv (Kv (hullIdx (fun j => Kv j 0) (qv 0))) := by
  rw [score_eq_lineEval, score_eq_lineEval, h,
    show Kv (hullIdx (fun j => Kv j 0) (qv 0)) 0
      = sortedKey (fun j => Kv j 0) (hullProbe (fun j => Kv j 0) (qv 0)) from
      hullIdx_spec (fun j => Kv j 0) (qv 0)]
  exact argmaxSet_tie (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1) hc
    (hullProbe_mem_argmaxSet (fun j => Kv j 0) (qv 0))

/-- The hypotheses are satisfiable, and by the position the search itself
returns: `hullProbe` is a maximizer, and the key `hullIdx` names carries its
value. -/
example [Nonempty (Fin n)] (Kv : Fin n → EucSpace 1) (qv : EucSpace 1) :
    hullProbe (fun j => Kv j 0) (qv 0)
        ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1) ∧
      Kv (hullIdx (fun j => Kv j 0) (qv 0)) 0
        = sortedKey (fun j => Kv j 0) (hullProbe (fun j => Kv j 0) (qv 0)) :=
  ⟨hullProbe_mem_argmaxSet (fun j => Kv j 0) (qv 0), hullIdx_spec (fun j => Kv j 0) (qv 0)⟩

/-- **The head returns what the walk resolved, at any two lines the walk can
collect.**  Both keys are named by positions the tie set contains, and nothing
says either is the one the exact search returns: two keys collected by *any*
walk over `argmaxSet` score alike, because each of them scores what `hullIdx`
scores.  `hullTie_head_resolves` is the case where one of them is `hullIdx`;
`Transformer.ALM.FloatHeadTie` is the case where one of them is the index the
floating-point search returned, which at a tie need not be the same. -/
theorem argmaxTie_head_resolves [Nonempty (Fin n)] (Kv : Fin n → EucSpace 1) (qv : EucSpace 1)
    (β ε C : ℝ) (V : Fin n → ℝ × ℝ) (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ)
    (i₁ i₂ : Fin n) (b c : ℕ)
    (hb : b ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (hc : c ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (h₁ : Kv i₁ 0 = sortedKey (fun j => Kv j 0) b)
    (h₂ : Kv i₂ 0 = sortedKey (fun j => Kv j 0) c)
    (hne : i₁ ≠ i₂) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : M p = Meta.empty.add (V i₁) sp) (hMr : M r = Meta.empty.add (V i₂) sr)
    (hmass : 1 - ε
      ≤ Real.exp (β * score qv (Kv i₁)) / ∑ k, Real.exp (β * score qv (Kv k))
          + Real.exp (β * score qv (Kv i₂)) / ∑ k, Real.exp (β * score qv (Kv k)))
    (hC : ∀ j, ‖V j - (((V i₁).1 + (V i₂).1) / 2, ((V i₁).2 + (V i₂).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score qv (Kv j)) / ∑ k, Real.exp (β * score qv (Kv k))) • V j)
        - (scanCombined M p r).resolveAverage‖ ≤ ε * C :=
  softmax_head_resolves_average β (fun j => score qv (Kv j)) V i₁ i₂ hne
    (score qv (Kv i₁)) rfl
    ((score_eq_of_mem_argmaxSet Kv qv i₂ c hc h₂).trans
      (score_eq_of_mem_argmaxSet Kv qv i₁ b hb h₁).symm)
    M p r sp sr hsp hsr hMp hMr ε C hmass hC

/-- **The head returns what the walk resolved, on the hull's own tie.**  No
hypothesis names a score: the second key is given by a position the search
collected, and `score_eq_of_mem_argmaxSet` turns that into the level-set
hypothesis of `softmax_head_resolves_average`.  So where the machine has to
choose, the running head is within `ε·C` of the choice it makes, `ε` being the
mass outside the two tied keys. -/
theorem hullTie_head_resolves [Nonempty (Fin n)] (Kv : Fin n → EucSpace 1) (qv : EucSpace 1)
    (β ε C : ℝ) (V : Fin n → ℝ × ℝ) (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ)
    (i₂ : Fin n) (c : ℕ)
    (hc : c ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (h₂ : Kv i₂ 0 = sortedKey (fun j => Kv j 0) c)
    (hne : hullIdx (fun j => Kv j 0) (qv 0) ≠ i₂)
    (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : M p = Meta.empty.add (V (hullIdx (fun j => Kv j 0) (qv 0))) sp)
    (hMr : M r = Meta.empty.add (V i₂) sr)
    (hmass : 1 - ε
      ≤ Real.exp (β * score qv (Kv (hullIdx (fun j => Kv j 0) (qv 0))))
            / ∑ k, Real.exp (β * score qv (Kv k))
          + Real.exp (β * score qv (Kv i₂)) / ∑ k, Real.exp (β * score qv (Kv k)))
    (hC : ∀ j, ‖V j - (((V (hullIdx (fun j => Kv j 0) (qv 0))).1 + (V i₂).1) / 2,
        ((V (hullIdx (fun j => Kv j 0) (qv 0))).2 + (V i₂).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score qv (Kv j)) / ∑ k, Real.exp (β * score qv (Kv k))) • V j)
        - (scanCombined M p r).resolveAverage‖ ≤ ε * C :=
  argmaxTie_head_resolves Kv qv β ε C V M p r sp sr
    (hullIdx (fun j => Kv j 0) (qv 0)) i₂ (hullProbe (fun j => Kv j 0) (qv 0)) c
    (hullProbe_mem_argmaxSet (fun j => Kv j 0) (qv 0)) hc
    (hullIdx_spec (fun j => Kv j 0) (qv 0)) h₂ hne hsp hsr hMp hMr hmass hC

/-- The hypotheses of both are satisfiable together, and the tie is a real
one: two lines carrying the same key tie at every query, so whichever of the
two `hullIdx` names, the other is collected beside it, and the head's output
is the aggregate the walk built. -/
example :
    ‖(∑ _j : Fin 2, (Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
            / ∑ _k : Fin 2, Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)))
          • ((0 : ℝ), (0 : ℝ)))
        - (scanCombined (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0).resolveAverage‖
      ≤ 1 * 0 := by
  refine hullTie_head_resolves (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
    (WithLp.toLp 2 ![(0 : ℝ)]) 1 1 0 (fun _ => ((0 : ℝ), (0 : ℝ)))
    (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0 0 0
    (Fin.rev (hullIdx (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0)))
    (hullProbe (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    (hullProbe_mem_argmaxSet _ _)
    (hullIdx_spec (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    ((by decide : ∀ i : Fin 2, i ≠ Fin.rev i) _) le_rfl le_rfl rfl rfl ?_ ?_
  · have h0 := softmax_weight_nonneg (n := 2) 1
      (fun _ => score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) 0
    simp only [sub_self] at *
    linarith
  · intro j
    norm_num

end ALM
end Transformer
