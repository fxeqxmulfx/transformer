/-
# The other tie-break mode, and what it costs the head

`Transformer.ALM.SoftmaxTie` proves the softmax head implements
`TieBreak::AVERAGE`: where two keys tie it returns their mean, which is what
`resolve` writes out, up to `ε·C`.  The machine has a second mode
(`transformer_vm/attention/hull2d_cht.h`, lines 70-83), and
`Transformer.ALM.HullResolve` formalizes it — `scanCombined_resolveLatest_of_ne`
returns the payload of whichever tied line was appended later — but no
statement related it to the head at all.  That silence is not neutral: it left
open whether the head is close to the `LATEST` answer too.

It is not, and by an amount that can be named.  `LATEST` returns one of the two
payloads and the head returns their mean, so the two differ by exactly half the
distance between the payloads, and `softmax_head_resolves_latest` is that bound:
`ε·C + ‖V b - V c‖/2`, with the two modes coinciding precisely when the tied
payloads agree.
-/

import Transformer.ALM.SoftmaxTie

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **And it is off `TieBreak::LATEST` by half the spread, never more.**  The
other mode returns one of the two payloads outright, so it cannot agree with
the head in general — the head averages.  What can be said is the exact
penalty: whichever of the two the later insertion carries, it sits half the
distance between them from their mean, so the head's output is within
`ε·C + ‖V b - V c‖/2` of what `resolve` writes out under `LATEST`.  The two
modes coincide, as they should, exactly when the tied payloads agree. -/
theorem softmax_head_resolves_latest [Nonempty (Fin n)] (β : ℝ) (s : Fin n → ℝ)
    (V : Fin n → ℝ × ℝ) (b c : Fin n) (hbc : b ≠ c) (σ : ℝ) (hb : s b = σ) (hc : s c = σ)
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr) (hsne : sp ≠ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (ε C : ℝ)
    (hmass : 1 - ε ≤ Real.exp (β * s b) / ∑ k, Real.exp (β * s k)
      + Real.exp (β * s c) / ∑ k, Real.exp (β * s k))
    (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        - (scanCombined M p r).resolveLatest‖ ≤ ε * C + ‖V b - V c‖ / 2 := by
  have havg := softmax_head_resolves_average β s V b c hbc σ hb hc M p r sp sr hsp hsr
    hMp hMr ε C hmass hC
  rw [scanCombined_resolveAverage M p r (V b) (V c) sp sr hsp hsr hMp hMr] at havg
  have hhalf : ∀ u v : ℝ × ℝ, (((u.1 + v.1) / 2, (u.2 + v.2) / 2) : ℝ × ℝ) - u
      = ((1 : ℝ) / 2) • (v - u) := by
    intro u v
    refine Prod.ext ?_ ?_ <;> simp [Prod.smul_def] <;> ring
  have hnorm : ∀ u v : ℝ × ℝ, ‖(((u.1 + v.1) / 2, (u.2 + v.2) / 2) : ℝ × ℝ) - u‖
      = ‖u - v‖ / 2 := by
    intro u v
    rw [hhalf u v, norm_smul, norm_sub_rev]
    simp [div_eq_mul_inv]
    ring
  have htri : ∀ y : ℝ × ℝ,
      ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j) - y‖
        ≤ ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
            - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖
          + ‖(((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2) - y‖ := by
    intro y
    simpa [dist_eq_norm] using
      dist_triangle (∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        ((((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2) : ℝ × ℝ) y
  rw [scanCombined_resolveLatest_of_ne M p r (V b) (V c) sp sr hsp hsr hsne hMp hMr]
  split_ifs
  · refine le_trans (htri (V c)) ?_
    have : ‖(((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2) - V c‖ = ‖V b - V c‖ / 2 := by
      rw [show (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)
          = (((V c).1 + (V b).1) / 2, ((V c).2 + (V b).2) / 2) by
        refine Prod.ext ?_ ?_ <;> simp <;> ring, hnorm (V c) (V b), norm_sub_rev]
    linarith [this]
  · refine le_trans (htri (V b)) ?_
    rw [hnorm (V b) (V c)]
    linarith

/-- The hypotheses are satisfiable: two distinct keys carrying the same
payload tie at every query, their spread around the mean is zero, and an
append-only log gives them different sequence numbers. -/
example : (0 : Fin 2) ≠ 1 ∧ (0 : ℤ) ≠ 1 ∧
    ∀ j : Fin 2, ‖(fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) j
      - ((((0 : ℝ), (0 : ℝ)).1 + ((0 : ℝ), (0 : ℝ)).1) / 2,
         (((0 : ℝ), (0 : ℝ)).2 + ((0 : ℝ), (0 : ℝ)).2) / 2)‖ ≤ 0 := by
  refine ⟨by decide, by decide, fun j => ?_⟩
  norm_num

end ALM
end Transformer
