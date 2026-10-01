/-
# Accumulating local weak errors

arXiv:2506.12543v1, Section 4.3, Theorem 1.
A contraction and a local defect of order eta^2 give an error of order eta
on a finite horizon. The comparison operator and its local error are
explicit hypotheses; no unproved diffusion theorem is used here.
-/

import Transformer.BatchSize.Section4_ErrorFunction

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Bounded measurable observables for Markov expectations,
Section 4.3, Theorem 1. -/
abbrev BoundedObservable (E : Type*) [MeasurableSpace E] :=
  {φ : E → ℝ // Measurable φ ∧ ∃ C : ℝ, ∀ x, |φ x| ≤ C}

/-- Integration against a probability measure contracts the uniform
error between observables; Section 4.3, Theorem 1's weak-error argument. -/
theorem probability_expectation_contraction {E : Type*} [MeasurableSpace E]
    (P : Measure E) [IsProbabilityMeasure P] (φ ψ : BoundedObservable E)
    (r : ℝ) (hclose : ∀ x, |φ.val x - ψ.val x| ≤ r) :
    |(∫ x, φ.val x ∂P) - ∫ x, ψ.val x ∂P| ≤ r := by
  have hi (q : BoundedObservable E) : Integrable q.val P := by
    obtain ⟨C, hC⟩ := q.property.2
    apply (integrable_const C).mono' q.property.1.aestronglyMeasurable
    exact ae_of_all _ (fun x => by simpa [Real.norm_eq_abs] using hC x)
  rw [← integral_sub (hi φ) (hi ψ)]
  have h := norm_integral_le_of_norm_le_const (f := fun x => φ.val x - ψ.val x)
    (ae_of_all P (fun x => by simpa [Real.norm_eq_abs] using hclose x))
  simpa [Real.norm_eq_abs] using h

/-- Joint nonvacuity of the expectation-contraction hypotheses,
Section 4.3: the Dirac probability law and the zero observable. -/
example : ∃ P : Measure Unit, IsProbabilityMeasure P ∧
    ∃ φ ψ : BoundedObservable Unit, ∀ x, |φ.val x - ψ.val x| ≤ (0 : ℝ) := by
  refine ⟨Measure.dirac (), inferInstance, ?_⟩
  let φ : BoundedObservable Unit := ⟨fun _ => 0, measurable_const, 0, by simp⟩
  exact ⟨φ, φ, by simp⟩

/-- A local discrepancy accumulates at most linearly in the number
of steps under a Markov contraction, Section 4.3, Theorem 1.
The hypothesis is needed only along the comparison operator's orbit
before step n, rather than over arbitrarily long time intervals. -/
theorem weak_error_telescope {E : Type*} [MeasurableSpace E]
    (A S : BoundedObservable E → BoundedObservable E) (φ : BoundedObservable E)
    (δ : ℝ) (hδ : 0 ≤ δ)
    (hcontract : ∀ ψ χ : BoundedObservable E, ∀ r : ℝ, 0 ≤ r →
      (∀ x, |ψ.val x - χ.val x| ≤ r) → ∀ x, |(A ψ).val x - (A χ).val x| ≤ r)
    (n : ℕ) (hlocal : ∀ j : ℕ, j < n → ∀ x,
      |(A ((S^[j]) φ)).val x - (S ((S^[j]) φ)).val x| ≤ δ) :
    ∀ x, |((A^[n]) φ).val x - ((S^[n]) φ).val x| ≤ (n : ℝ) * δ := by
  revert hlocal
  induction n with
  | zero => intro hlocal; simp
  | succ n ih =>
    intro hlocal x
    have ih' := ih (fun j hj => hlocal j (hj.trans (Nat.lt_succ_self n)))
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    have ha := hcontract ((A^[n]) φ) ((S^[n]) φ) (n * δ) (by positivity) ih' x
    calc
      |(A ((A^[n]) φ)).val x - (S ((S^[n]) φ)).val x| ≤
          |(A ((A^[n]) φ)).val x - (A ((S^[n]) φ)).val x| +
          |(A ((S^[n]) φ)).val x - (S ((S^[n]) φ)).val x| := abs_sub_le _ _ _
      _ ≤ n * δ + δ := add_le_add ha (hlocal n (Nat.lt_succ_self n) x)
      _ = (n.succ : ℝ) * δ := by push_cast; ring

/-- Joint nonvacuity of the telescope hypotheses, Section 4.3:
both operators are the identity and the local defect is zero. -/
example : (0 : ℝ) ≤ 0 ∧
    (∀ ψ χ : BoundedObservable Unit, ∀ r : ℝ, 0 ≤ r →
      (∀ x, |ψ.val x - χ.val x| ≤ r) → ∀ x, |(id ψ).val x - (id χ).val x| ≤ r) ∧
    (∀ φ : BoundedObservable Unit, ∀ n j : ℕ, j < n → ∀ x,
      |(id ((id^[j]) φ)).val x - (id ((id^[j]) φ)).val x| ≤ (0 : ℝ)) := by
  refine ⟨le_rfl, fun ψ χ r hr h => h, ?_⟩
  simp

/-- Conditional finite-horizon conclusion of Section 4.3, Theorem 1:
an O(eta^2) defect on propagated observables gives O(eta) weak error.
Constructing the diffusion operator and verifying its local defect remain
separate requirements; they are not assumed via a sorried theorem. -/
theorem finite_horizon_weak_error {E : Type*} [MeasurableSpace E]
    (A S : BoundedObservable E → BoundedObservable E) (φ : BoundedObservable E)
    (C η T : ℝ) (hC : 0 ≤ C) (hη : 0 ≤ η)
    (hcontract : ∀ ψ χ : BoundedObservable E, ∀ r : ℝ, 0 ≤ r →
      (∀ x, |ψ.val x - χ.val x| ≤ r) → ∀ x, |(A ψ).val x - (A χ).val x| ≤ r)
    (n : ℕ) (horizon : (n : ℝ) * η ≤ T)
    (hlocal : ∀ j : ℕ, j < n → ∀ x,
      |(A ((S^[j]) φ)).val x - (S ((S^[j]) φ)).val x| ≤ C * η ^ 2) :
    ∀ x, |((A^[n]) φ).val x - ((S^[n]) φ).val x| ≤ C * T * η := by
  intro x
  refine (weak_error_telescope A S φ (C * η ^ 2) (by positivity)
    hcontract n hlocal x).trans ?_
  calc
    (n : ℝ) * (C * η ^ 2) = (C * η) * ((n : ℝ) * η) := by ring
    _ ≤ (C * η) * T := mul_le_mul_of_nonneg_left horizon (mul_nonneg hC hη)
    _ = C * T * η := by ring

/-- Joint nonvacuity of the finite-horizon estimate, Section 4.3:
identity operators, zero defect, ten steps of length 1/1000 on horizon one. -/
example : ∃ (A S : BoundedObservable Unit → BoundedObservable Unit)
    (φ : BoundedObservable Unit) (C η T : ℝ) (n : ℕ),
    0 ≤ C ∧ 0 ≤ η ∧ (n : ℝ) * η ≤ T ∧
    (∀ ψ χ : BoundedObservable Unit, ∀ r : ℝ, 0 ≤ r →
      (∀ x, |ψ.val x - χ.val x| ≤ r) → ∀ x, |(A ψ).val x - (A χ).val x| ≤ r) ∧
    (∀ j : ℕ, j < n → ∀ x,
      |(A ((S^[j]) φ)).val x - (S ((S^[j]) φ)).val x| ≤ C * η ^ 2) := by
  let φ : BoundedObservable Unit := ⟨fun _ => 0, measurable_const, 0, by simp⟩
  refine ⟨id, id, φ, 0, 1 / 1000, 1, 10, le_rfl, by norm_num,
    by norm_num, fun ψ χ r hr h => h, ?_⟩
  simp

end Transformer.BatchSize
