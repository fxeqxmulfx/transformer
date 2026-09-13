/-
# The softmax head returns the index's answer

`Transformer.ALM.Softmax` and its refinements bound the softmax weight of a
winning score, and `Transformer.ALM.VectorInt` removes the factor `n` from
that bound for integer keys.  None of it mentions an index: the winner is a
hypothesis, `i₀`, handed to the theorem.  On the other side
`Transformer.ALM.HullIndex` builds the index and proves it exact.  The two
met only through the definition of `score`, so "the real head, which uses
softmax, returns what the hull returns" was an expectation, not a statement.

`NNIndex.ans_eq_of_query_mem` closes it in one step, and for every index at
once: when the query is one of the stored keys, exactness alone forces the
index to return that key.  Nothing about the hull is used — exactness is the
whole hypothesis — so it applies to `hullIndex`, `bruteForce` and `scanIndex`
alike.

What follows is then a statement about the running head.  In every dimension
`softmax_at_index_ge` gives the published bound `1 - (n-1)e^{-β}` at the key
the index names, the gap being the unit lattice gap of
`one_le_dist_sq_of_int`; in dimension one — the case `hull2d_cht.h` compiles
to — `softmax_at_hullIndex_int` gives the length-free bound of
`softmax_winner_int_sharp_one`, which does not degrade as keys accumulate.

Source of the head: `transformer_vm/attention/hull2d_cht.h`, lines 203-215.
-/

import Transformer.ALM.HullIndex
import Transformer.ALM.VectorInt

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-! ### An exact index returns a stored key -/

/-- **Exactness alone pins the answer.**  If the query is one of the keys and
the keys are distinct, every exact index returns that key's own position: the
argmax is unique by `score_lt_of_ne`, so there is nothing to choose. -/
theorem NNIndex.ans_eq_of_query_mem (I : NNIndex) [Nonempty (Fin n)]
    (K : Fin n → EucSpace m) (hinj : Function.Injective K) (i₀ : Fin n) :
    I.ans K (K i₀) = i₀ := by
  by_contra hne
  have hbest := I.ans_isGreatest K (K i₀) i₀
  have hkey : K (I.ans K (K i₀)) ≠ K i₀ := fun h => hne (hinj h)
  have hlt := score_lt_of_ne hkey
  linarith

/-- The hypotheses are satisfiable: two distinct keys in dimension one. -/
example : Function.Injective (fun j : Fin 2 => embInt (fun _ : Fin 1 => (j : ℤ))) := by
  intro a b h
  have := congrFun (congrArg (fun x : EucSpace 1 => (x : Fin 1 → ℝ)) h) 0
  simp only [embInt_apply] at this
  exact Fin.ext (by exact_mod_cast this)

/-- The hull's index is one of them. -/
theorem hullIndex_ans_eq [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (hinj : Function.Injective K) (i₀ : Fin n) :
    hullIndex.ans K (K i₀) = i₀ :=
  hullIndex.ans_eq_of_query_mem K hinj i₀

/-! ### Integer keys are separated, so the softmax concentrates there -/

/-- The unit gap of the lattice, as `Transformer.ALM.Softmax` wants it: every
other key scores at least `1` below the query's own key. -/
theorem score_gap_one_of_int [Nonempty (Fin n)] (K : Fin n → (Fin m → ℤ))
    (hinj : Function.Injective K) (i₀ : Fin n) :
    ∀ j, j ≠ i₀ → score (embInt (K i₀)) (embInt (K j)) + 1
      ≤ score (embInt (K i₀)) (embInt (K i₀)) := by
  intro j hj
  have hne : embInt (K j) ≠ embInt (K i₀) := by
    intro h
    refine hj (hinj ?_)
    funext i
    have := congrFun (congrArg (fun x : EucSpace m => (x : Fin m → ℝ)) h) i
    simp only [embInt_apply] at this
    exact_mod_cast this
  have hdist := one_le_dist_sq_of_int (embInt (K j)) (embInt (K i₀))
    (fun i => ⟨K j i, rfl⟩) (fun i => ⟨K i₀ i, rfl⟩) hne
  have hgap := score_gap (embInt (K i₀)) (embInt (K j))
  linarith

/-- **The published bound, at the key the index names.**  The head's softmax
weight on the index's own answer is at least `1 - (n-1)e^{-β}`: the machine's
argmax and the head's output agree up to an error exponentially small in the
inverse temperature. -/
theorem softmax_at_index_ge (I : NNIndex) [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Fin n → (Fin m → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 - ((n : ℝ) - 1) * Real.exp (-(β * 1))
      ≤ Real.exp (β * score (embInt (K i₀))
            (embInt (K (I.ans (fun j => embInt (K j)) (embInt (K i₀))))))
          / ∑ j, Real.exp (β * score (embInt (K i₀)) (embInt (K j))) := by
  have hinj' : Function.Injective (fun j => embInt (K j)) := by
    intro a b h
    refine hinj ?_
    funext i
    have := congrFun (congrArg (fun x : EucSpace m => (x : Fin m → ℝ)) h) i
    simp only [embInt_apply] at this
    exact_mod_cast this
  rw [I.ans_eq_of_query_mem (fun j => embInt (K j)) hinj' i₀]
  exact softmax_winner_ge β hβ (fun j => score (embInt (K i₀)) (embInt (K j))) i₀ 1
    (score_gap_one_of_int K hinj i₀)

/-- **And the length-free bound, in the dimension the machine runs in.**  For
scalar integer keys the weight at `hullIndex`'s answer is bounded below by a
quantity that does not mention `n` at all, so the head keeps agreeing with the
index as keys accumulate. -/
theorem softmax_at_hullIndex_int [Nonempty (Fin n)] (β : ℝ) (hβ : 0 < β)
    (K : Fin n → (Fin 1 → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3)))
      ≤ Real.exp (β * sScore (K i₀ 0)
            (K (hullIndex.ans (fun j => embInt (K j)) (embInt (K i₀))) 0))
          / ∑ j, Real.exp (β * sScore (K i₀ 0) (K j 0)) := by
  have hinj' : Function.Injective (fun j => embInt (K j)) := by
    intro a b h
    refine hinj ?_
    funext i
    have := congrFun (congrArg (fun x : EucSpace 1 => (x : Fin 1 → ℝ)) h) i
    simp only [embInt_apply] at this
    exact_mod_cast this
  rw [hullIndex.ans_eq_of_query_mem (fun j => embInt (K j)) hinj' i₀]
  exact softmax_winner_int_sharp_one β hβ K hinj i₀

/-- The hypotheses are satisfiable: three distinct scalar integer keys at a
positive inverse temperature. -/
example : (0 : ℝ) < 1 ∧ Function.Injective (fun j : Fin 3 => fun _ : Fin 1 => (j : ℤ)) := by
  refine ⟨by norm_num, fun a b h => Fin.ext ?_⟩
  have h0 : ((a : ℕ) : ℤ) = ((b : ℕ) : ℤ) := congrFun h 0
  exact_mod_cast h0

end ALM
end Transformer
