/-
# Initialization second moments

arXiv:2606.25971v2, §4.1.1 and Appendix D. Standard deviation `1/sqrt(d)`
for centered entries specifies a second moment, not an exact realized norm.
The norm identity below needs no independence between different entries.
-/

import Transformer.MagnitudeDirection.Section4_MuonScaling
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open MeasureTheory
open scoped BigOperators

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- The expected squared Frobenius norm is `mn/d` when each initialized entry
has second moment `1/d`. The paper's initialization “norm” is correctly
interpreted as this second-moment radius, rather than exact equality for
every random matrix. Source: arXiv:2606.25971v2, §4.1.1 and Appendix D. -/
theorem initialization_expected_energy (mu : Measure Ω)
    (X : Ω → Matrix (Fin m) (Fin n) ℝ) (d : ℕ)
    (hI : ∀ i j, Integrable (fun w => X w i j ^ 2) mu)
    (hM : ∀ i j, ∫ w, X w i j ^ 2 ∂mu = 1 / (d : ℝ)) :
    (∫ w, frobeniusNorm (X w) ^ 2 ∂mu) = (m : ℝ) * n / d := by
  simp_rw [frobeniusNorm_sq]
  rw [integral_finsetSum _ (fun i hi => integrable_finsetSum _ (fun j hj => hI i j))]
  simp_rw [integral_finsetSum _ (fun j hj => hI _ j), hM]
  simp
  ring

/-- The explicit entry-moment assumptions are satisfiable,
arXiv:2606.25971v2, Appendix D. -/
example :
    let X : Unit → Matrix (Fin 1) (Fin 1) ℝ := fun _ => 1
    (∀ i j : Fin 1, Integrable (fun w => X w i j ^ 2) (Measure.dirac ())) ∧
    (∀ i j : Fin 1, (∫ w, X w i j ^ 2 ∂Measure.dirac ()) = 1 / (1 : ℝ)) := by
  constructor
  · intro i j
    fin_cases i
    fin_cases j
    exact integrable_const _
  · intro i j
    fin_cases i
    fin_cases j
    simp

/-- Square root of the expected energy is exactly the general initialization
radius used in the shape correction, arXiv:2606.25971v2, Appendix D. -/
theorem initialization_secondMoment_radius (mu : Measure Ω)
    (X : Ω → Matrix (Fin m) (Fin n) ℝ) (d : ℕ)
    (hI : ∀ i j, Integrable (fun w => X w i j ^ 2) mu)
    (hM : ∀ i j, ∫ w, X w i j ^ 2 ∂mu = 1 / (d : ℝ)) :
    Real.sqrt (∫ w, frobeniusNorm (X w) ^ 2 ∂mu) = initializationRadius m n d := by
  rw [initialization_expected_energy mu X d hI hM]
  rfl

/-- The second-moment-radius hypotheses hold, arXiv:2606.25971v2, Appendix D. -/
example :
    let X : Unit → Matrix (Fin 1) (Fin 1) ℝ := fun _ => 1
    (∀ i j : Fin 1, Integrable (fun w => X w i j ^ 2) (Measure.dirac ())) ∧
    (∀ i j : Fin 1, (∫ w, X w i j ^ 2 ∂Measure.dirac ()) = 1 / (1 : ℝ)) := by
  constructor
  · intro i j
    fin_cases i
    fin_cases j
    exact integrable_const _
  · intro i j
    fin_cases i
    fin_cases j
    simp

end Transformer.MagnitudeDirection
