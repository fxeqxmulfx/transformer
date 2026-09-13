/-
# And so the bounds are bounds on standard attention

`Transformer.ALM.SAHead` identifies the hand-written softmax head with
`Transformer.XSA.SAOutput` at coordinate-projection weights.  That is only
worth stating if the bounds move across it, and they do, with no work beyond
the rewrite: every theorem of this development that bounds the hand-written
sum is now a theorem about causal self-attention.

`SAOutput_close_of_mass` is the general transfer — whatever lower-bounds the
weight at a slot upper-bounds the distance from the attention output to the
value stored there — and `sa_head_output_at_index` is the one that matters:
`head_output_at_index` (`Transformer.ALM.SoftmaxValue`) applies to an ordinary
attention head, so a trained head and a lookup head are the same operator at
different weights.  That is what licenses the hybrid layer the authors
describe, where the executor and the language model share an architecture.

Source: Percepta, *Can LLMs Be Computers?* (2026-03-11), §"a dedicated fast
path paired with a slower, more general model".
-/

import Transformer.ALM.SAHead

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-- **Mass at a slot is proximity to its value, for the attention head.**  The
transfer of `softmax_output_close` across the identification: no hypothesis
here mentions the lookup, only the weight the head puts on the slot. -/
theorem SAOutput_close_of_mass [Nonempty (Idx n)] (β : ℝ) (q k : Idx n → EucSpace m)
    (Vm : ParamMatrix ((m + 1) + (m + 1))) (i : Idx n) (hi : ∀ j : Idx n, (j : ℕ) ≤ (i : ℕ))
    (i₀ : Idx n) (C ε : ℝ)
    (hC : ∀ j, ‖Vm (residual (q j) (k j)) - Vm (residual (q i₀) (k i₀))‖ ≤ C)
    (hw : 1 - ε ≤ Real.exp (β * score (q i) (k i₀)) / ∑ l, Real.exp (β * score (q i) (k l))) :
    ‖XSA.SAOutput ((m + 1) + (m + 1)) n (β • queryProj) keyProj Vm
          (fun j => residual (q j) (k j)) i
        - Vm (residual (q i₀) (k i₀))‖ ≤ ε * C := by
  rw [SAOutput_eq_softmax_head β q k Vm i hi]
  exact softmax_output_close β (fun j => score (q i) (k j))
    (fun j => Vm (residual (q j) (k j))) i₀ C ε hC hw

/-- The hypotheses are satisfiable: at `ε = 1` the weight bound is free, and a
single stored slot has no spread around itself. -/
example (β : ℝ) (q k : Idx 2 → EucSpace m) :
    1 - 1 ≤ Real.exp (β * score (q 0) (k 0)) / ∑ l, Real.exp (β * score (q 0) (k l)) := by
  simpa using softmax_weight_nonneg β (fun j => score (q 0) (k j)) 0

/-- **The index's answer, returned by an ordinary attention head.**  This is
`head_output_at_index` with the hand-written sum replaced by
`XSA.SAOutput`: on the lookup path the running attention head's output differs
from the value of the slot the index names by at most `(n-1)·e^{-β}·C`.

The query token reads its own query block, the stored slots supply the keys,
and the mask is definitional — `i` is the decoding position, the last of the
stored prefix. -/
theorem sa_head_output_at_index (I : NNIndex) [Nonempty (Idx n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Idx n → (Fin m → ℤ)) (hinj : Function.Injective K) (i₀ i : Idx n)
    (hi : ∀ j : Idx n, (j : ℕ) ≤ (i : ℕ))
    (q : Idx n → EucSpace m) (hq : q i = embInt (K i₀))
    (Vm : ParamMatrix ((m + 1) + (m + 1))) (C : ℝ)
    (hC : ∀ j, ‖Vm (residual (q j) (embInt (K j)))
      - Vm (residual (q i₀) (embInt (K i₀)))‖ ≤ C) :
    ‖XSA.SAOutput ((m + 1) + (m + 1)) n (β • queryProj) keyProj Vm
          (fun j => residual (q j) (embInt (K j))) i
        - Vm (residual (q (I.ans (fun j => embInt (K j)) (embInt (K i₀))))
            (embInt (K (I.ans (fun j => embInt (K j)) (embInt (K i₀))))))‖
      ≤ ((n : ℝ) - 1) * Real.exp (-(β * 1)) * C := by
  rw [SAOutput_eq_softmax_head β q (fun j => embInt (K j)) Vm i hi, hq]
  exact head_output_at_index I β hβ K hinj i₀
    (fun j => Vm (residual (q j) (embInt (K j)))) C hC

/-- The hypotheses are satisfiable together, and by the data the machine runs
on: one stored slot, the key it holds, the query that reads it, and no spread
at all. -/
example (Vm : ParamMatrix ((1 + 1) + (1 + 1))) :
    Function.Injective (fun _ : Idx 1 => fun _ : Fin 1 => (0 : ℤ)) ∧
      (∀ j : Idx 1, (j : ℕ) ≤ ((0 : Idx 1) : ℕ)) ∧
      (fun _ : Idx 1 => embInt (fun _ : Fin 1 => (0 : ℤ))) 0
        = embInt (fun _ : Fin 1 => (0 : ℤ)) ∧
      ∀ j : Idx 1, ‖Vm (residual ((fun _ : Idx 1 => embInt (fun _ : Fin 1 => (0 : ℤ))) j)
            (embInt ((fun _ : Idx 1 => fun _ : Fin 1 => (0 : ℤ)) j)))
          - Vm (residual (embInt (fun _ : Fin 1 => (0 : ℤ)))
            (embInt (fun _ : Fin 1 => (0 : ℤ))))‖ ≤ 0 :=
  ⟨fun a b _ => Subsingleton.elim a b, by decide, rfl, fun j => by fin_cases j; simp⟩

end ALM
end Transformer
