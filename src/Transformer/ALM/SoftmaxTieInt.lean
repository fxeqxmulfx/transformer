/-
# The tie bound on the data the machine stores

`Transformer.ALM.SoftmaxMass` derives the mass on a tie from a score gap `δ`,
and leaves `δ` to be supplied.  On the keys the machine actually holds it need
not be supplied at all: scalar integer keys score integers against an integer
query, so a key that does not tie loses by at least a full unit — the same
observation `score_gap_one_of_int` makes for a single winner, at a tie.

Which keys tie is itself determined: `sScore_eq_iff` of
`Transformer.ALM.TieBreak` says two distinct integer keys score alike against
`q` exactly when they are reflections of each other in `q`, so a tie has
exactly two members and every other key is a clear `1` below.  `softmax_tie_mass_int` is the mass that follows and
`softmax_head_resolves_average_of_int` the head bound it buys: on distinct
integer keys, at a genuine tie, the head's output is within
`((n-2)·e^{-β}/2)·C` of what `resolve` writes out, with no hypothesis about
the softmax, the gap, or the mass.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 70-83; the lattice
separation is `Transformer.ALM.Basic.one_le_dist_sq_of_int` in dimension one.
-/

import Transformer.ALM.SoftmaxMass
import Transformer.ALM.TieBreak

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **So every key outside the tie loses by a full unit.**  With two distinct
keys tied at the maximum, a third key neither equals nor reflects them, so its
score differs — and integer scores that differ differ by at least `1`. -/
theorem sScore_tie_gap_one (K : Fin n → ℤ) (hinj : Function.Injective K) (q : ℤ)
    (b c : Fin n) (hbc : b ≠ c) (htie : sScore q (K c) = sScore q (K b))
    (hmax : ∀ j, sScore q (K j) ≤ sScore q (K b))
    (j : Fin n) (hjb : j ≠ b) (hjc : j ≠ c) :
    sScore q (K j) + 1 ≤ sScore q (K b) := by
  have hint : ∀ k : ℤ, sScore q k = ((2 * k * q - k ^ 2 : ℤ) : ℝ) := by
    intro k; unfold sScore; push_cast; ring
  have hKcb : K c ≠ K b := fun h => hbc (hinj h).symm
  have hrefl : 2 * q = K c + K b := (sScore_eq_iff q (K c) (K b) hKcb).mp htie
  have hne : sScore q (K j) ≠ sScore q (K b) := by
    intro h
    rcases eq_or_ne (K j) (K b) with h' | h'
    · exact hjb (hinj h')
    · exact hjc (hinj (by have := (sScore_eq_iff q (K j) (K b) h').mp h; omega))
  have hlt : ((2 * K j * q - K j ^ 2 : ℤ) : ℝ) < ((2 * K b * q - K b ^ 2 : ℤ) : ℝ) := by
    rw [← hint, ← hint]
    exact lt_of_le_of_ne (hmax j) hne
  have hz : 2 * K j * q - K j ^ 2 + 1 ≤ 2 * K b * q - K b ^ 2 := by
    have : (2 * K j * q - K j ^ 2 : ℤ) < 2 * K b * q - K b ^ 2 := by exact_mod_cast hlt
    omega
  rw [hint (K j), hint (K b)]
  exact_mod_cast hz

/-- **And so the tie carries all but an exponentially small mass.**  No gap is
assumed: the lattice supplies `δ = 1`, exactly as it supplies the single-winner
bound of `Transformer.ALM.SoftmaxIndex`. -/
theorem softmax_tie_mass_int (β : ℝ) (hβ : 0 ≤ β) (K : Fin n → ℤ)
    (hinj : Function.Injective K) (q : ℤ) (b c : Fin n) (hbc : b ≠ c)
    (htie : sScore q (K c) = sScore q (K b))
    (hmax : ∀ j, sScore q (K j) ≤ sScore q (K b)) :
    1 - ((n : ℝ) - 2) * Real.exp (-(β * 1)) / 2
      ≤ Real.exp (β * sScore q (K b)) / ∑ k, Real.exp (β * sScore q (K k))
        + Real.exp (β * sScore q (K c)) / ∑ k, Real.exp (β * sScore q (K k)) :=
  softmax_tie_mass_ge β hβ (fun j => sScore q (K j)) b c hbc (sScore q (K b)) rfl htie 1
    (fun j hjb hjc => sScore_tie_gap_one K hinj q b c hbc htie hmax j hjb hjc)

/-- **The head against `resolve`, on integer keys and nothing else.**  Two
distinct keys tie at the maximum, the payloads spread by at most `C`, and the
log holds each key's aggregate; then the head's output is within
`((n-2)e^{-β}/2)·C` of the vector the machine writes out under
`TieBreak::AVERAGE`.  Every hypothesis is about the stored data. -/
theorem softmax_head_resolves_average_of_int [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Fin n → ℤ) (hinj : Function.Injective K) (q : ℤ) (V : Fin n → ℝ × ℝ)
    (b c : Fin n) (hbc : b ≠ c) (htie : sScore q (K c) = sScore q (K b))
    (hmax : ∀ j, sScore q (K j) ≤ sScore q (K b))
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (C : ℝ) (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * sScore q (K j)) / ∑ k, Real.exp (β * sScore q (K k))) • V j)
        - Meta.resolveAverage (scanCombined M p r)‖
      ≤ (((n : ℝ) - 2) * Real.exp (-(β * 1)) / 2) * C :=
  softmax_head_resolves_average_of_gap β hβ (fun j => sScore q (K j)) V b c hbc
    (sScore q (K b)) rfl htie 1
    (fun j hjb hjc => sScore_tie_gap_one K hinj q b c hbc htie hmax j hjb hjc)
    M p r sp sr hsp hsr hMp hMr C hC

/-- The hypotheses are satisfiable, and the tie is a real one: the keys `0`
and `2` are reflections in the query `1`, so they score alike and each is the
maximum. -/
example :
    ‖(∑ j : Fin 2, (Real.exp (1 * sScore 1 (![0, 2] j))
            / ∑ k : Fin 2, Real.exp (1 * sScore 1 (![0, 2] k)))
          • (fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) j)
        - Meta.resolveAverage
            (scanCombined (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0)‖
      ≤ ((((2 : ℕ) : ℝ) - 2) * Real.exp (-(1 * 1)) / 2) * 0 := by
  refine softmax_head_resolves_average_of_int 1 zero_le_one ![0, 2] (by decide) 1
    (fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) 0 1 (by decide) ?_ ?_
    (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0 0 0 le_rfl le_rfl rfl rfl 0
    (fun j => by norm_num)
  · norm_num [sScore]
  · intro j
    fin_cases j <;> norm_num [sScore]

end ALM
end Transformer
