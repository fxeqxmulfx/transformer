/-
# The other tie-break mode, priced by the data too

`Transformer.ALM.SoftmaxMass` and `Transformer.ALM.SoftmaxTieInt` discharge the
mass hypothesis of the averaging head bound — first from a score gap, then from
the integer lattice, where no gap has to be supplied.  `TieBreak::LATEST` was
left where it started: `softmax_head_resolves_latest` still took `1 - ε ≤ w b +
w c` on faith, so of the machine's two tie-break modes one was priced by the
keys and the other by an assumption.

The two theorems here remove that asymmetry.  `softmax_head_resolves_latest_of_gap`
is the `LATEST` bound with `ε = (n-2)e^{-βδ}/2` supplied by a gap, and
`softmax_head_resolves_latest_of_int` is it on distinct integer keys, where the
gap is `1` by `sScore_tie_gap_one`.  In both, the extra `‖V b - V c‖/2` stays:
that term is not an error at all but the exact price of returning one payload
where the head returns their mean.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 70-83 and 276-306.
-/

import Transformer.ALM.SoftmaxTieInt
import Transformer.ALM.SoftmaxLatest

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **The `LATEST` head bound, priced by the keys.**  Two keys tie, every other
key scores at least `δ` below, and the head's output is within
`((n-2)e^{-βδ}/2)·C + ‖V b - V c‖/2` of the payload the log appended later. -/
theorem softmax_head_resolves_latest_of_gap [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (s : Fin n → ℝ) (V : Fin n → ℝ × ℝ) (b c : Fin n) (hbc : b ≠ c)
    (σ : ℝ) (hb : s b = σ) (hc : s c = σ) (δ : ℝ)
    (hgap : ∀ j, j ≠ b → j ≠ c → s j + δ ≤ σ)
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr) (hsne : sp ≠ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (C : ℝ) (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        - Meta.resolveLatest (scanCombined M p r)‖
      ≤ (((n : ℝ) - 2) * Real.exp (-(β * δ)) / 2) * C + ‖V b - V c‖ / 2 :=
  softmax_head_resolves_latest β s V b c hbc σ hb hc M p r sp sr hsp hsr hsne hMp hMr _ C
    (softmax_tie_mass_ge β hβ s b c hbc σ hb hc δ hgap) hC

/-- **And on the stored integer keys, with no gap supplied.**  The lattice
gives `δ = 1`, so both of the machine's tie-break modes are now bounded against
the head by quantities read off the data alone. -/
theorem softmax_head_resolves_latest_of_int [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Fin n → ℤ) (hinj : Function.Injective K) (q : ℤ) (V : Fin n → ℝ × ℝ)
    (b c : Fin n) (hbc : b ≠ c) (htie : sScore q (K c) = sScore q (K b))
    (hmax : ∀ j, sScore q (K j) ≤ sScore q (K b))
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr) (hsne : sp ≠ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (C : ℝ) (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * sScore q (K j)) / ∑ k, Real.exp (β * sScore q (K k))) • V j)
        - Meta.resolveLatest (scanCombined M p r)‖
      ≤ (((n : ℝ) - 2) * Real.exp (-(β * 1)) / 2) * C + ‖V b - V c‖ / 2 :=
  softmax_head_resolves_latest_of_gap β hβ (fun j => sScore q (K j)) V b c hbc
    (sScore q (K b)) rfl htie 1
    (fun j hjb hjc => sScore_tie_gap_one K hinj q b c hbc htie hmax j hjb hjc)
    M p r sp sr hsp hsr hsne hMp hMr C hC

/-- The hypotheses are satisfiable, and the log really has a later line to
prefer: the keys `0` and `2` tie at the query `1`, the aggregates are appended
at `0` and `1`, and with equal payloads the two modes agree. -/
example :
    ‖(∑ j : Fin 2, (Real.exp (1 * sScore 1 (![0, 2] j))
            / ∑ k : Fin 2, Real.exp (1 * sScore 1 (![0, 2] k)))
          • (fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) j)
        - Meta.resolveLatest
            (scanCombined (fun i : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) (i : ℤ)) 0 1)‖
      ≤ ((((2 : ℕ) : ℝ) - 2) * Real.exp (-(1 * 1)) / 2) * 0
        + ‖((0 : ℝ), (0 : ℝ)) - ((0 : ℝ), (0 : ℝ))‖ / 2 := by
  refine softmax_head_resolves_latest_of_int 1 zero_le_one ![0, 2] (by decide) 1
    (fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) 0 1 (by decide) ?_ ?_
    (fun i : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) (i : ℤ)) 0 1 0 1 le_rfl (by norm_num)
    (by decide) rfl rfl 0 (fun j => by norm_num)
  · norm_num [sScore]
  · intro j
    fin_cases j <;> norm_num [sScore]

end ALM
end Transformer
