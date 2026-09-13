/-
# The head's output at the tie the running code merged

`Transformer.ALM.HullHead` bounds the softmax head against the aggregate
`resolve` builds from two tied keys, for any two keys the tie set contains
(`argmaxTie_head_resolves`).  `Transformer.ALM.FloatWalk` says which keys the
implementation's two `while` loops actually put into that aggregate: the
members of `fpTieSet`, the lines whose recomputed score compares `==` to
`best_score` in the arithmetic the machine runs.

The two never met.  The head-level bound was stated at maximizers of the exact
score, and at a tie the floating-point search need not land on the index the
exact one returns, so nothing connected the value the head approximates to the
value `combined.resolve` writes out.

`fp_head_tie_resolves` is the composition, and it costs one hypothesis:
`fp_walk_sound` needs no exactness of the score routine, only `2δ < 1` and a
bound on the keys, so the head's output is within `ε·C` of what the running
code resolves under a condition on the precision alone.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 70-83 and 268-306.
-/

import Transformer.ALM.HullHead
import Transformer.ALM.FloatWalk

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **The head returns what the running code resolved.**  Both keys are named
by positions the *floating-point* merge loops accepted — `fpTieSet`, the `==`
of the implementation — and `fp_walk_sound` puts those positions among the
exact maximizers, where `argmaxTie_head_resolves` applies.  Nothing is assumed
exact: the score routine may round, provided it rounds by less than half the
gap it is comparing (`2δ < 1`), and the breakpoints by less than half a unit of
the key lattice (`F.u * B < 1/2`). -/
theorem fp_head_tie_resolves (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (Kv : Fin n → EucSpace 1) (qv : EucSpace 1) (q : ℤ) (hq : qv 0 = (q : ℝ))
    (hK : ∀ i, ∃ z : ℤ, Kv i 0 = (z : ℝ)) (B : ℝ)
    (hbd : ∀ j ≤ keyCard (fun j => Kv j 0) - 1, |sortedKey (fun j => Kv j 0) j| ≤ B)
    (hu : F.u * B < 1 / 2) (hδ : 2 * S.δ < 1)
    (β ε C : ℝ) (V : Fin n → ℝ × ℝ) (Mt : ℕ → Meta) (p r : ℕ) (sp sr : ℤ)
    (i₁ i₂ : Fin n) (b c : ℕ)
    (hb : b ∈ fpTieSet S F (fun j => Kv j 0) (q : ℝ))
    (hc : c ∈ fpTieSet S F (fun j => Kv j 0) (q : ℝ))
    (h₁ : Kv i₁ 0 = sortedKey (fun j => Kv j 0) b)
    (h₂ : Kv i₂ 0 = sortedKey (fun j => Kv j 0) c)
    (hne : i₁ ≠ i₂) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : Mt p = Meta.empty.add (V i₁) sp) (hMr : Mt r = Meta.empty.add (V i₂) sr)
    (hmass : 1 - ε
      ≤ Real.exp (β * score qv (Kv i₁)) / ∑ k, Real.exp (β * score qv (Kv k))
          + Real.exp (β * score qv (Kv i₂)) / ∑ k, Real.exp (β * score qv (Kv k)))
    (hC : ∀ j, ‖V j - (((V i₁).1 + (V i₂).1) / 2, ((V i₁).2 + (V i₂).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score qv (Kv j)) / ∑ k, Real.exp (β * score qv (Kv k))) • V j)
        - (scanCombined Mt p r).resolveAverage‖ ≤ ε * C := by
  have hsub := fp_walk_sound S F (fun j => Kv j 0) hK q B hbd hu hδ
  have hbA : b ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0)
      (keyCard (fun j => Kv j 0) - 1) := by rw [hq]; exact hsub hb
  have hcA : c ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0)
      (keyCard (fun j => Kv j 0) - 1) := by rw [hq]; exact hsub hc
  exact argmaxTie_head_resolves Kv qv β ε C V Mt p r sp sr i₁ i₂ b c hbA hcA h₁ h₂ hne
    hsp hsr hMp hMr hmass hC

/-- The hypotheses are satisfiable together, and the tie is the code's own:
two keys of equal value at the query they both maximize, the position the
floating-point probe returns (`fpProbe_mem_fpTieSet`, so the merge loops really
do accept it), exact arithmetic for both routines, and the bound
`exists_bound_sortedKey` supplies. -/
example :
    ‖(∑ _j : Fin 2, (Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
            / ∑ _k : Fin 2, Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)))
          • ((0 : ℝ), (0 : ℝ)))
        - (scanCombined (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0).resolveAverage‖
      ≤ 1 * 0 := by
  obtain ⟨B, hB⟩ := exists_bound_sortedKey
    (fun j : Fin 2 => (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
  obtain ⟨i, hi⟩ := exists_eq_sortedKey
    (fun j : Fin 2 => (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
    (fpProbe_le exactArith
      (fun j : Fin 2 => (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      (((0 : ℤ) : ℝ)))
  refine fp_head_tie_resolves exactScore exactArith
    (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
    (WithLp.toLp 2 ![(0 : ℝ)]) 0 (by norm_num) (fun _ => ⟨0, by norm_num⟩) B hB
    (by simp [exactArith]) (by norm_num [exactScore]) 1 1 0 (fun _ => ((0 : ℝ), (0 : ℝ)))
    (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0 0 0 i (Fin.rev i)
    (fpProbe exactArith
      (fun j : Fin 2 => (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      (((0 : ℤ) : ℝ)))
    (fpProbe exactArith
      (fun j : Fin 2 => (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      (((0 : ℤ) : ℝ)))
    (fpProbe_mem_fpTieSet _ _ _ _) (fpProbe_mem_fpTieSet _ _ _ _) hi hi
    ((by decide : ∀ j : Fin 2, j ≠ Fin.rev j) i) le_rfl le_rfl rfl rfl ?_ ?_
  · have h0 := softmax_weight_nonneg (n := 2) 1
      (fun _ => score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) 0
    simp only [sub_self] at *
    linarith
  · intro j
    norm_num

end ALM
end Transformer
