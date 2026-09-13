/-
# The other tie-break mode, at the tie the hull found

`Transformer.ALM.HullHead` bounds the head against what `resolve` writes out
under `TieBreak::AVERAGE`, at any two keys the tie set contains.  The machine's
other mode returns one of the two payloads instead
(`Transformer.ALM.HullResolve.scanCombined_resolveLatest_of_ne`), and
`Transformer.ALM.SoftmaxLatest` prices the difference: half the distance
between the payloads, on top of the `ε·C` of the averaging bound.

`argmaxTie_head_resolves_latest` is that price at the hull's own ties — the
`LATEST` counterpart of `argmaxTie_head_resolves`, with the scores again
supplied by the geometry rather than assumed, so that the machine's two modes
are both accounted for where the machine actually has to choose between them.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 70-83 and 276-306.
-/

import Transformer.ALM.HullHead
import Transformer.ALM.SoftmaxLatest

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **And under the other tie-break mode, at the same two lines.**  `LATEST`
hands back one of the two payloads rather than their mean, so the head is off
by half the distance between them and no more
(`softmax_head_resolves_latest`); the keys are again given by positions the
tie set contains, so nothing here names a score either. -/
theorem argmaxTie_head_resolves_latest [Nonempty (Fin n)] (Kv : Fin n → EucSpace 1)
    (qv : EucSpace 1) (β ε C : ℝ) (V : Fin n → ℝ × ℝ) (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ)
    (i₁ i₂ : Fin n) (b c : ℕ)
    (hb : b ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (hc : c ∈ argmaxSet (sortedKey fun j => Kv j 0) (qv 0) (keyCard (fun j => Kv j 0) - 1))
    (h₁ : Kv i₁ 0 = sortedKey (fun j => Kv j 0) b)
    (h₂ : Kv i₂ 0 = sortedKey (fun j => Kv j 0) c)
    (hne : i₁ ≠ i₂) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr) (hsne : sp ≠ sr)
    (hMp : M p = Meta.empty.add (V i₁) sp) (hMr : M r = Meta.empty.add (V i₂) sr)
    (hmass : 1 - ε
      ≤ Real.exp (β * score qv (Kv i₁)) / ∑ k, Real.exp (β * score qv (Kv k))
          + Real.exp (β * score qv (Kv i₂)) / ∑ k, Real.exp (β * score qv (Kv k)))
    (hC : ∀ j, ‖V j - (((V i₁).1 + (V i₂).1) / 2, ((V i₁).2 + (V i₂).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score qv (Kv j)) / ∑ k, Real.exp (β * score qv (Kv k))) • V j)
        - (scanCombined M p r).resolveLatest‖ ≤ ε * C + ‖V i₁ - V i₂‖ / 2 :=
  softmax_head_resolves_latest β (fun j => score qv (Kv j)) V i₁ i₂ hne
    (score qv (Kv i₁)) rfl
    ((score_eq_of_mem_argmaxSet Kv qv i₂ c hc h₂).trans
      (score_eq_of_mem_argmaxSet Kv qv i₁ b hb h₁).symm)
    M p r sp sr hsp hsr hsne hMp hMr ε C hmass hC

/-- The hypotheses are satisfiable, and the tie is real: two lines carrying
the same key tie at every query, an append-only log numbers them `0` and `1`,
and with equal payloads the two tie-break modes return the same vector — the
half-spread term vanishes. -/
example :
    ‖(∑ _j : Fin 2, (Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
            / ∑ _k : Fin 2, Real.exp (1 * score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
              (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)))
          • ((0 : ℝ), (0 : ℝ)))
        - Meta.resolveLatest
            (scanCombined (fun i : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) (i : ℤ)) 0 1)‖
      ≤ 1 * 0 + ‖((0 : ℝ), (0 : ℝ)) - ((0 : ℝ), (0 : ℝ))‖ / 2 := by
  refine argmaxTie_head_resolves_latest
    (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1))
    (WithLp.toLp 2 ![(0 : ℝ)]) 1 1 0 (fun _ => ((0 : ℝ), (0 : ℝ)))
    (fun i : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) (i : ℤ)) 0 1 0 1
    (hullIdx (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    (Fin.rev (hullIdx (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0)))
    (hullProbe (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    (hullProbe (fun j : Fin 2 => (fun _ : Fin 2 =>
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) j 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    (hullProbe_mem_argmaxSet _ _) (hullProbe_mem_argmaxSet _ _)
    (hullIdx_spec (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    (hullIdx_spec (fun _ : Fin 2 => (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0)
      ((WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1) 0))
    ((by decide : ∀ i : Fin 2, i ≠ Fin.rev i) _) le_rfl (by norm_num) (by decide) rfl rfl ?_ ?_
  · have h0 := softmax_weight_nonneg (n := 2) 1
      (fun _ => score (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)
        (WithLp.toLp 2 ![(0 : ℝ)] : EucSpace 1)) 0
    simp only [sub_self] at *
    linarith
  · intro j
    norm_num

end ALM
end Transformer
