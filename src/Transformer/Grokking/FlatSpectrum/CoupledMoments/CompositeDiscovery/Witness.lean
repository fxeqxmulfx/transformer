module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Completion

/-!
# A composite categorical table with repeated axes and held-out cells

Source: `review/flat_spectrum.md`, CM28. Seventeen samples identify a `7 × 7`
table over six latent states. The code swaps the class names zero and one;
rows zero and six, and columns zero and six, share a phase. Thirty-two cells
are held out. The observable joint kernel is computed below from the actual
seventeen equations, without a discovered-code or correct-completion premise.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

/-- The seventeen categorical cell identities supplied to the learner, CM28. -/
def sixCells : Fin 17 → Fin 7 × Fin 7 :=
  ![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2),
    (1, 3), (1, 4), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (0, 6)]

/-- The seventeen observed class names, CM28. -/
def sixLabels : Fin 17 → ZMod 6 :=
  ![1, 0, 2, 3, 4, 5, 0, 2, 3, 4, 5, 2, 3, 4, 5, 1, 1]

/-- The target code, used in the correctness statement but not by the learner. -/
def sixCode : ZMod 6 ≃ ZMod 6 := Equiv.swap 0 1

/-- The target phases repeat at the distinct input names zero and six. -/
def sixPhase (i : Fin 7) : ZMod 6 := (i.val : ZMod 6)

/-- The observations are those of the unknown composite cyclic model, CM28. -/
theorem six_real (t : Fin 17) :
    sixCode (sixLabels t) = sixPhase (sixCells t).1 + sixPhase (sixCells t).2 := by
  revert t
  decide

/-- A generator for the joint module, not an input to `discover`, CM28. -/
def sixVector : Fin 7 ⊕ (Fin 7 ⊕ ZMod 6) → ZMod 6 :=
  pack sixPhase sixPhase sixCode

/-- The displayed vector satisfies all actual equations and both pins, CM28. -/
theorem six_mem : sixVector ∈ jointKernel (ZMod 6) sixCells sixLabels 0 1 := by
  refine ⟨fun t => (six_real t).symm, ?_, ?_⟩ <;> decide

/-- Eliminate the actual seventeen observed equations: every solution is a
scalar multiple of the displayed vector over `ZMod 6`, CM28. -/
theorem six_cyclic : Cyclic (jointKernel (ZMod 6) sixCells sixLabels 0 1) := by
  refine ⟨sixVector, six_mem, fun x hx => ?_⟩
  let r : Fin 7 → ZMod 6 := fun i => x (.inl i)
  let s : Fin 7 → ZMod 6 := fun j => x (.inr (.inl j))
  let z : ZMod 6 → ZMod 6 := fun c => x (.inr (.inr c))
  change (∀ t, r (sixCells t).1 + s (sixCells t).2 = z (sixLabels t)) ∧
    r 0 = 0 ∧ z 1 = 0 at hx
  obtain ⟨h, hr0, hz1⟩ := hx
  have h00 : r 0 + s 0 = z 1 := h 0
  have h01 : r 0 + s 1 = z 0 := h 1
  have h02 : r 0 + s 2 = z 2 := h 2
  have h03 : r 0 + s 3 = z 3 := h 3
  have h04 : r 0 + s 4 = z 4 := h 4
  have h05 : r 0 + s 5 = z 5 := h 5
  have h10 : r 1 + s 0 = z 0 := h 6
  have h11 : r 1 + s 1 = z 2 := h 7
  have h12 : r 1 + s 2 = z 3 := h 8
  have h13 : r 1 + s 3 = z 4 := h 9
  have h14 : r 1 + s 4 = z 5 := h 10
  have h20 : r 2 + s 0 = z 2 := h 11
  have h30 : r 3 + s 0 = z 3 := h 12
  have h40 : r 4 + s 0 = z 4 := h 13
  have h50 : r 5 + s 0 = z 5 := h 14
  have h60 : r 6 + s 0 = z 1 := h 15
  have h06 : r 0 + s 6 = z 1 := h 16
  have hz2 : z 2 = 2 * z 0 := by
    linear_combination -h11 + h10 + h01 - h00 - hz1
  have hz3 : z 3 = 3 * z 0 := by
    linear_combination -h12 + h10 + h02 - h00 - hz1 + hz2
  have hz4 : z 4 = 4 * z 0 := by
    linear_combination -h13 + h10 + h03 - h00 - hz1 + hz3
  have hz5 : z 5 = 5 * z 0 := by
    linear_combination -h14 + h10 + h04 - h00 - hz1 + hz4
  have hz (c : ZMod 6) : z c = z 0 * sixCode c := by
    fin_cases c
    · change z 0 = z 0 * 1
      ring
    · change z 1 = z 0 * 0
      rw [hz1, mul_zero]
    · change z 2 = z 0 * 2
      linear_combination hz2
    · change z 3 = z 0 * 3
      linear_combination hz3
    · change z 4 = z 0 * 4
      linear_combination hz4
    · change z 5 = z 0 * 5
      linear_combination hz5
  have hr (i : Fin 7) : r i = z (sixCode.symm (sixPhase i)) := by
    fin_cases i
    · change r 0 = z 1
      linear_combination hr0 - hz1
    · change r 1 = z 0
      linear_combination h10 - h00 + hr0 - hz1
    · change r 2 = z 2
      linear_combination h20 - h00 + hr0 - hz1
    · change r 3 = z 3
      linear_combination h30 - h00 + hr0 - hz1
    · change r 4 = z 4
      linear_combination h40 - h00 + hr0 - hz1
    · change r 5 = z 5
      linear_combination h50 - h00 + hr0 - hz1
    · change r 6 = z 1
      linear_combination h60 - h00 + hr0 - hz1
  have hs (j : Fin 7) : s j = z (sixCode.symm (sixPhase j)) := by
    fin_cases j
    · change s 0 = z 1
      linear_combination h00 - hr0
    · change s 1 = z 0
      linear_combination h01 - hr0
    · change s 2 = z 2
      linear_combination h02 - hr0
    · change s 3 = z 3
      linear_combination h03 - hr0
    · change s 4 = z 4
      linear_combination h04 - hr0
    · change s 5 = z 5
      linear_combination h05 - hr0
    · change s 6 = z 1
      linear_combination h06 - hr0
  refine ⟨z 0, ?_⟩
  funext v
  rcases v with i | j | c
  · change r i = z 0 * sixPhase i
    rw [hr, hz, sixCode.apply_symm_apply]
  · change s j = z 0 * sixPhase j
    rw [hs, hz, sixCode.apply_symm_apply]
  · exact hz c

open Schatten

/-- The observed-only learner recovers every cell, including thirty-two
unobserved cells, in this nonconstant composite witness, CM28. -/
theorem six_recovers : discover 6 sixCells sixLabels 0 1 =
    some (phaseLabels sixCode sixPhase sixPhase) :=
  discover_eq_some sixCode sixPhase sixPhase six_real 0 1 six_cyclic

/-- Two genuinely held-out answers, with repeated input phases present in
the same table. The finite check does not evaluate the choice-based learner. -/
theorem six_hidden_answers :
    (∀ t, sixCells t ≠ (2, 3) ∧ sixCells t ≠ (2, 4)) ∧
    (discover 6 sixCells sixLabels 0 1).map (fun y => (y 2 3, y 2 4)) = some (5, 1) ∧
    sixPhase 0 = sixPhase 6 := by
  rw [six_recovers]
  decide

example := completion_eq_target sixCode sixPhase sixPhase six_real 0 1 six_cyclic
  sixCode sixPhase sixPhase six_real

example (y : Fin 7 → Fin 7 → ZMod 6) :=
  completes_iff sixCode sixPhase sixPhase six_real 0 1 six_cyclic y

example := discover_eq_some sixCode sixPhase sixPhase six_real 0 1 six_cyclic

example := conditions_of_discover six_recovers

example := discover_sound sixCode sixPhase sixPhase six_real six_recovers

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
