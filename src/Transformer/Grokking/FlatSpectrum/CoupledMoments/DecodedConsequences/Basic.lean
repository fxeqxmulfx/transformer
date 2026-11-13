module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Certificate

/-!
# Actual observed row consequences identify decoded answers

Source: `review/flat_spectrum.md`, CM29. The original matrix has only observed
cell/class equations, without gauges or extra anchors. A finite row program
certifies a query by producing its actual coefficient row. Unknown structural
realizability and class-code injectivity then identify the decoded answer.
No cyclicity, unique code coordinates or solver-reported rank is required.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences

open CompositeDiscovery

variable {R n C S : Type*} [CommRing R]
variable [Fintype n] [Fintype C] [DecidableEq n] [DecidableEq C]

/-- The actual coefficient row for one proposed decoded answer, CM29. -/
def queryRow (i j : n) (c : C) : n ⊕ (n ⊕ C) → R :=
  fun v => coordinate (.inl i) v + coordinate (.inr (.inl j)) v -
    coordinate (.inr (.inr c)) v

/-- Only original observed equations; no gauge or unseen-answer pins, CM29. -/
def observedMatrix (cell : S → n × n) (labels : S → C) : S → n ⊕ (n ⊕ C) → R :=
  fun t => queryRow (cell t).1 (cell t).2 (labels t)

/-- Evaluating the three nonzero query coefficients, CM29. -/
theorem query_action (i j : n) (c : C) (x : n ⊕ (n ⊕ C) → R) :
    (∑ v, queryRow i j c v * x v) =
      x (.inl i) + x (.inr (.inl j)) - x (.inr (.inr c)) := by
  simp [queryRow, coordinate, sub_mul, add_mul, Finset.sum_sub_distrib,
    Finset.sum_add_distrib, ite_mul]

/-- Every actual code/phase fit annihilates the original observed matrix,
CM29. This premise contains only observed labels. -/
theorem observed_zero {cell : S → n × n} {labels : S → C}
    (e : C → R) (r s : n → R)
    (hfit : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) :
    ∀ t, rowAction (observedMatrix cell labels) (pack r s e) t = 0 := by
  intro t
  simpa only [rowAction, observedMatrix, query_action, pack, Sum.elim_inl, Sum.elim_inr]
    using sub_eq_zero.mpr (hfit t).symm

variable [DecidableEq S]

/-- A replayed original row consequence identifies its proposed answer
under an unknown injective-code target model, CM29. The certificate consists
only of actual coefficients, not a hypothesis naming the unknown answer. -/
theorem decoded_of_program {cell : S → n × n} {labels : S → C}
    (e : C → R) (hinj : Function.Injective e) (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (ops : List (RowOp S R)) (slot : S) (i j : n) (c : C)
    (hrow : ∀ v, reduceRows ops (observedMatrix cell labels) slot v = queryRow i j c v) :
    y i j = c := by
  have hfit : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2 := by
    intro t
    rw [← hobs t]
    exact hreal _ _
  have hz := reduceRows_zero ops (observed_zero e r s hfit) slot
  simp only [rowAction, hrow, query_action, pack, Sum.elim_inl, Sum.elim_inr] at hz
  apply hinj
  rw [hreal]
  exact sub_eq_zero.mp hz

example : ∀ t : ZMod 4 × ZMod 4,
    rowAction (observedMatrix id (fun t => t.1 + t.2)) (pack id id id) t = 0 :=
  observed_zero (R := ZMod 4) id id id (fun _ => rfl)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences
