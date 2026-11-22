module

public import Transformer.Phasor.SharedValues

/-!
# The shared embedding gradient at the tied read and projection

Source: the tied-gradient accumulation trial in `experiments/phasor_value_transport/README.md`.
The same embedding is the terminal classifier and the value table. Its
pullback adds the two resulting gradients to the existing backbone gradient.
Every stored component can be updated by one worker, including masked batches.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {ι : Type*} {p D : ℕ}

/-- The upstream logit and hidden gradients enter through different uses of
each embedding. `feat` is the scaled residual seen by the tied projection,
and `weight` is the signed coefficient of that embedding in the value read. -/
noncomputable def tiedGradient (active : Finset ι) (base : Fin p → EucSpace D)
    (feat upstream : ι → EucSpace D) (logit weight : ι → Fin p → ℝ) (c : Fin p) : EucSpace D :=
  (base c + ∑ i ∈ active, logit i c • feat i) + ∑ i ∈ active, weight i c • upstream i

/-- Pairing the shared gradient with an arbitrary simultaneous perturbation
of the embeddings gives the sum of the backbone pairing, the tied logit
variation and the value-read variation at every active row. The result
allows negative weights and repeated uses of any embedding. -/
theorem tiedGradient_adjoint (active : Finset ι) (base change : Fin p → EucSpace D)
    (feat upstream : ι → EucSpace D) (logit weight : ι → Fin p → ℝ) :
    ∑ c, ⟪tiedGradient active base feat upstream logit weight c, change c⟫ =
      (∑ c, ⟪base c, change c⟫) +
      ∑ i ∈ active, ((∑ c, logit i c * ⟪change c, feat i⟫) +
        ⟪upstream i, ∑ c, weight i c • change c⟫) := by
  simp only [tiedGradient, inner_add_left, sum_inner, real_inner_smul_left,
    inner_sum, real_inner_smul_right, Finset.sum_add_distrib]
  rw [Finset.sum_comm (s := Finset.univ) (t := active)
        (f := fun c i => logit i c * ⟪feat i, change c⟫),
      Finset.sum_comm (s := Finset.univ) (t := active)
        (f := fun c i => weight i c * ⟪upstream i, change c⟫)]
  have flip (i : ι) (c : Fin p) : ⟪feat i, change c⟫ = ⟪change c, feat i⟫ := real_inner_comm _ _
  simp only [flip, add_assoc]

/-- A class absent from both paths keeps precisely the gradient from the
backbone, even when other classes receive contributions at those rows. -/
theorem tiedGradient_unread (active : Finset ι) (base : Fin p → EucSpace D)
    (feat upstream : ι → EucSpace D) (logit weight : ι → Fin p → ℝ) (c : Fin p)
    (hl : ∀ i ∈ active, logit i c = 0) (hw : ∀ i ∈ active, weight i c = 0) :
    tiedGradient active base feat upstream logit weight c = base c := by
  have l := Finset.sum_eq_zero (s := active) (f := fun i => logit i c • feat i)
    (fun i hi => by rw [hl i hi, zero_smul])
  have w := Finset.sum_eq_zero (s := active) (f := fun i => weight i c • upstream i)
    (fun i hi => by rw [hw i hi, zero_smul])
  simp [tiedGradient, l, w]

/-- Only the two active rows can contribute to the first class. -/
example (base : Fin 2 → EucSpace 8) (feat upstream : Fin 3 → EucSpace 8) :
    tiedGradient {0, 2} base feat upstream
      (fun i c => if c = 0 ∧ i = 1 then 1 else 0)
      (fun i c => if c = 1 ∨ i = 1 then -2 else 0) 0 = base 0 := by
  apply tiedGradient_unread <;> intro i hi <;> fin_cases i <;> norm_num at *

end Transformer.Phasor
