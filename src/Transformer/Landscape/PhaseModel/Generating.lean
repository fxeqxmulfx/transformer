module

public import Transformer.Landscape.PhaseModel.Division
public import Transformer.Landscape.PhaseModel.Descent

/-!
# Phase model — sets of rows generating the division table

When every row of the full division table is an integer combination of the
rows of a set `S` of pairs, the relations of `S` imply those of the full
table, so the energy `V_S` has the zeros of `V_full`: the phases of the
characters up to periods (`divisionEnergy_eq_zero_iff_of_mem_span`). Then
only `0` is orthogonal to every row of `S`
(`eq_zero_of_forall_inner_divisionRow_of_mem_span`), so every zero of `V_S`
is nondegenerate (`injective_fderiv_gradient_divisionEnergy_of_mem_span`),
and the Łojasiewicz inequality holds there with exponent `1/2`
(`isLojasiewiczAt_divisionEnergy_of_mem_span`). In particular the Hessian of
`V_full` is injective at the phases of every character up to periods
(`injective_fderiv_gradient_divisionEnergy_univ`), where the Łojasiewicz
inequality holds with exponent `1/2` (`isLojasiewiczAt_divisionEnergy_univ`).
When only `0` is orthogonal to every row of `S`, for `0 < η` with
`9 |S| η < 1`, an orbit of gradient descent on `V_S` with the phases of a
character as a limit point up to periods converges to them up to a period
(`exists_tendsto_characterPhase`).
-/

@[expose] public section

open Filter Real
open scoped Topology

namespace Transformer.Landscape.PhaseModel

open Landscape.SaddleAvoidance (gradientStep)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ} {S : Finset (Fin n × Fin n)} {η : ℝ}

/-- **When the rows of `S` generate those of the full division table, only
`0` is orthogonal to every row of `S`.** -/
theorem eq_zero_of_forall_inner_divisionRow_of_mem_span
    (hS : ∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n))))
    (h : EucSpace n) (hh : ∀ p ∈ S, ⟪intVec (divisionRow p), h⟫ = 0) : h = 0 :=
  eq_zero_of_forall_inner_divisionRow h fun p => AddSubgroup.mem_bot.1
    (inner_intVec_mem_of_mem_span ⊥ (by
      rintro _ ⟨q, hq, rfl⟩
      exact AddSubgroup.mem_bot.2 (hh q hq)) (hS p))

/-- **When the rows of `S` generate those of the full division table, the
zeros of `V_S` are the phases of the characters up to periods.** -/
theorem divisionEnergy_eq_zero_iff_of_mem_span
    (hS : ∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n))))
    (ψ : EucSpace n) :
    divisionEnergy S ψ = 0 ↔ ∃ j : ℤ, ψ - characterPhase n j ∈ periods n := by
  rw [← divisionEnergy_univ_eq_zero_iff]
  simp only [divisionEnergy, phaseEnergy_eq_zero_iff]
  refine ⟨fun h r => ?_, fun h r => h ⟨r.1, Finset.mem_univ _⟩⟩
  obtain ⟨k, hk⟩ := AddSubgroup.mem_zmultiples_iff.1
    (inner_intVec_mem_of_mem_span (AddSubgroup.zmultiples (2 * π)) (v := ψ) (by
      rintro _ ⟨q, hq, rfl⟩
      obtain ⟨k, hk⟩ := h ⟨q, hq⟩
      exact AddSubgroup.mem_zmultiples_iff.2 ⟨k, by rw [hk, zsmul_eq_mul]⟩) (hS r.1))
  exact ⟨k, by rw [← hk, zsmul_eq_mul]⟩

/-- **When the rows of `S` generate those of the full division table, every
zero of `V_S` is nondegenerate**: the Hessian there is injective. -/
theorem injective_fderiv_gradient_divisionEnergy_of_mem_span
    (hS : ∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n))))
    {z : EucSpace n} (hz : divisionEnergy S z = 0) :
    Function.Injective (fderiv ℝ (gradient (divisionEnergy S)) z) :=
  injective_fderiv_gradient_of_phaseEnergy_eq_zero _ hz fun h hh =>
    eq_zero_of_forall_inner_divisionRow_of_mem_span hS h fun p hp => hh ⟨p, hp⟩

/-- **When the rows of `S` generate those of the full division table, the
Łojasiewicz inequality holds with exponent `1/2` at every zero of `V_S`**: a
zero is a critical point where the Hessian is injective, so
`Landscape.SaddleAvoidance.isLojasiewiczAt_of_injective` applies. -/
theorem isLojasiewiczAt_divisionEnergy_of_mem_span
    (hS : ∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n))))
    {z : EucSpace n} (hz : divisionEnergy S z = 0) :
    Landscape.SaddleAvoidance.IsLojasiewiczAt (divisionEnergy S) z (1 / 2) :=
  Landscape.SaddleAvoidance.isLojasiewiczAt_of_injective
    ((contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero)
    (hasFDerivAt_gradient_phaseEnergy _ z).differentiableAt.hasFDerivAt
    (gradient_eq_zero_of_phaseEnergy_eq_zero _ hz)
    (injective_fderiv_gradient_divisionEnergy_of_mem_span hS hz)

/-- The rows of the full division table generate themselves. -/
theorem divisionRow_mem_span_univ (p : Fin n × Fin n) :
    divisionRow p ∈ Submodule.span ℤ
      (divisionRow '' ((Finset.univ : Finset (Fin n × Fin n)) : Set (Fin n × Fin n))) :=
  Submodule.subset_span ⟨p, Finset.mem_coe.2 (Finset.mem_univ p), rfl⟩

/-- **The Hessian of `V_full` at the phases of a character is injective**, up
to periods: if `ψ - (2πjx/n)_x ∈ 2πℤⁿ`, then `ψ` is a zero of `V_full`
(`divisionEnergy_univ_eq_zero_iff`), and the rows `e_x - e_y - e_{x-y}` of the
full table span `ℝⁿ` (`eq_zero_of_forall_inner_divisionRow`). -/
theorem injective_fderiv_gradient_divisionEnergy_univ {ψ : EucSpace n} {j : ℤ}
    (hψ : ψ - characterPhase n j ∈ periods n) :
    Function.Injective
      (fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n)))) ψ) :=
  injective_fderiv_gradient_divisionEnergy_of_mem_span divisionRow_mem_span_univ
    ((divisionEnergy_univ_eq_zero_iff ψ).2 ⟨j, hψ⟩)

/-- **The Łojasiewicz inequality holds with exponent `1/2` for `V_full` at the
phases of a character**, up to periods. -/
theorem isLojasiewiczAt_divisionEnergy_univ {ψ : EucSpace n} {j : ℤ}
    (hψ : ψ - characterPhase n j ∈ periods n) :
    Landscape.SaddleAvoidance.IsLojasiewiczAt (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ
      (1 / 2) :=
  isLojasiewiczAt_divisionEnergy_of_mem_span divisionRow_mem_span_univ
    ((divisionEnergy_univ_eq_zero_iff ψ).2 ⟨j, hψ⟩)

/-- **An orbit with the phases of a character as a limit point converges to
them** up to a period, for gradient descent on `V_S` with `0 < η` and
`9 |S| η < 1`, when only `0` is orthogonal to every row of `S` (as when the
rows of `S` generate those of the full table,
`eq_zero_of_forall_inner_divisionRow_of_mem_span`). -/
theorem exists_tendsto_characterPhase
    (hspan : ∀ h, (∀ p ∈ S, ⟪intVec (divisionRow p), h⟫ = 0) → h = 0) (hη : 0 < η)
    (hηS : 9 * S.card * η < 1) {x : EucSpace n} {j : ℤ}
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
      ‖(gradientStep (divisionEnergy S) η)^[k] x + v - characterPhase n j‖ < ε) :
    ∃ w, Tendsto (fun k => (gradientStep (divisionEnergy S) η)^[k] x) atTop (𝓝 w) ∧
      characterPhase n j - w ∈ periods n := by
  have hηL : η * ∑ p : S, ‖intVec (divisionRow p.1)‖ ^ 2 < 1 := by
    nlinarith [sum_norm_intVec_divisionRow_sq_le S]
  obtain ⟨w, hw, -, hzw⟩ := exists_tendsto_phaseEnergy_eq_zero (fun p : S => divisionRow p.1)
    hη hηL (divisionEnergy_characterPhase S j) (fun h hh => hspan h fun p hp => hh ⟨p, hp⟩) hz
  exact ⟨w, hw, hzw⟩

/-- The hypotheses of the theorems above hold for the full table of every
order `n`, with `η = 1 / (9 n² + 1)`: its rows generate themselves, only `0`
is orthogonal to all of them, the phases of every character are a zero, and
the orbit starting there stays there. -/
example (n : ℕ) (j : ℤ) : let S := (Finset.univ : Finset (Fin n × Fin n))
    let η : ℝ := 1 / (9 * n ^ 2 + 1)
    (∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n)))) ∧
      (∀ h, (∀ p ∈ S, ⟪intVec (divisionRow p), h⟫ = 0) → h = 0) ∧
      divisionEnergy S (characterPhase n j) = 0 ∧ 0 < η ∧ 9 * (S.card : ℝ) * η < 1 ∧
      ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
        ‖(gradientStep (divisionEnergy S) η)^[k] (characterPhase n j) + v -
          characterPhase n j‖ < ε := by
  intro S η
  have hV := divisionEnergy_characterPhase S j
  have hg : gradient (divisionEnergy S) (characterPhase n j) = 0 :=
    gradient_eq_zero_of_phaseEnergy_eq_zero _ hV
  have h0 : gradientStep (divisionEnergy S) η (characterPhase n j) = characterPhase n j := by
    simp [Landscape.SaddleAvoidance.gradientStep, hg]
  have hcard : (S.card : ℝ) = n ^ 2 := by simp [S, sq]
  refine ⟨divisionRow_mem_span_univ,
    fun h hh => eq_zero_of_forall_inner_divisionRow h fun p => hh p (Finset.mem_univ p), hV,
    by simp only [η]; positivity, ?_, fun ε hε => ?_⟩
  · rw [hcard]
    simp only [η]
    rw [mul_one_div, div_lt_one (by positivity)]
    linarith
  · refine Frequently.of_forall fun k => ⟨0, fun _ => ⟨0, by simp⟩, ?_⟩
    rw [Function.iterate_fixed h0, add_zero, sub_self, norm_zero]
    exact hε

/-- The hypothesis of `injective_fderiv_gradient_divisionEnergy_univ` and
`isLojasiewiczAt_divisionEnergy_univ` holds for the phases of every character
translated by the period `(2π, …, 2π)`. -/
example (n : ℕ) (j : ℤ) :
    (characterPhase n j + WithLp.toLp 2 fun _ => 2 * π) - characterPhase n j ∈ periods n :=
  fun _ => ⟨1, by simp⟩

end Transformer.Landscape.PhaseModel
