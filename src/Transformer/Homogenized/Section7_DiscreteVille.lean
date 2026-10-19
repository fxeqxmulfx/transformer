module

public import Mathlib.Probability.Martingale.OptionalStopping

/-!
# Ville's maximal inequality from actual supermartingale data

ArXiv:2604.01978v1, §7, the maximal inequality used after
eq:Gronwall_E_selfcontained.
This module proves the discrete nonnegative supermartingale
inequality needed before passing to continuous noise paths.
It uses Mathlib's bounded optional stopping theorem on the actual
first threshold crossing, not an assumed tail or moment bound.

Both finite and infinite natural time horizons are treated. The
infinite horizon follows by continuity of the measures of the
actual increasing crossing events. The initial expectation stays explicit.
The example is a strictly
decreasing positive supermartingale, so none of the positivity,
integrability or stopping-time premises is contradictory.
-/

@[expose] public section

open MeasureTheory Set
open scoped ENNReal

namespace Transformer.Homogenized

/-- Bounded optional stopping decreases the actual expectation of a
supermartingale. Source: arXiv:2604.01978v1, §7, the maximal inequality;
the optional stopping input is Mathlib's expected_stoppedValue_mono. -/
theorem supermartingale_stopped_expectation_le {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℕ m) (Z : ℕ → Ω → ℝ)
    (hZ : Supermartingale Z ℱ P) (τ : Ω → WithTop ℕ) (hτ : IsStoppingTime ℱ τ)
    (N : ℕ) (hbdd : ∀ ω, τ ω ≤ N) :
    (∫ ω, stoppedValue Z τ ω ∂P) ≤ ∫ ω, Z 0 ω ∂P := by
  have hm := hZ.neg.expected_stoppedValue_mono (isStoppingTime_const ℱ 0) hτ
    (fun ω => bot_le) hbdd
  rw [stoppedValue_const, stoppedValue_neg] at hm
  change (∫ ω, -Z 0 ω ∂P) ≤ ∫ ω, -stoppedValue Z τ ω ∂P at hm
  rw [integral_neg, integral_neg] at hm
  exact neg_le_neg_iff.mp hm

/-- The finite-horizon crossing probability is bounded by the actual
initial expectation divided by the threshold, in multiplication form.
Source: arXiv:2604.01978v1, §7, the martingale maximal-inequality step. -/
theorem nonnegative_supermartingale_ville_finite {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℕ m) (Z : ℕ → Ω → ℝ)
    (hZ : Supermartingale Z ℱ P) (hpos : ∀ k ω, 0 ≤ Z k ω)
    (a : ℝ) (ha : 0 ≤ a) (N : ℕ) :
    ENNReal.ofReal a * P {ω | ∃ k ≤ N, a ≤ Z k ω} ≤ ENNReal.ofReal (∫ ω, Z 0 ω ∂P) := by
  classical
  let A : Set Ω := {ω | ∃ k ≤ N, a ≤ Z k ω}
  let τ : Ω → WithTop ℕ := fun ω => (hittingBtwn Z (Ici a) 0 N ω : ℕ)
  have hτ : IsStoppingTime ℱ τ := hZ.stronglyAdapted.adapted.isStoppingTime_hittingBtwn measurableSet_Ici
  have hbdd : ∀ ω, τ ω ≤ N := fun ω => by
    change ((hittingBtwn Z (Ici a) 0 N ω : ℕ) : WithTop ℕ) ≤ (N : WithTop ℕ)
    exact_mod_cast hittingBtwn_le (u := Z) (s := Ici a) (n := 0) (m := N) ω
  have hInt : Integrable (stoppedValue Z τ) P := integrable_stoppedValue ℕ hτ hZ.integrable hbdd
  have hmean := supermartingale_stopped_expectation_le P ℱ Z hZ τ hτ N hbdd
  have hA : MeasurableSet A := by
    have hAeq : A = ⋃ k : ℕ, ⋃ hk : k ≤ N, {ω | a ≤ Z k ω} := by ext ω; simp [A]
    rw [hAeq]
    apply MeasurableSet.iUnion
    intro k
    apply MeasurableSet.iUnion
    intro hk
    exact measurableSet_le measurable_const ((hZ.stronglyMeasurable k).measurable.le (ℱ.le k))
  have hpoint : ∀ ω, A.indicator (fun _ => a) ω ≤ stoppedValue Z τ ω := by
    intro ω
    by_cases hω : ω ∈ A
    · rw [indicator_of_mem hω]
      obtain ⟨k, hk, hka⟩ := hω
      exact stoppedValue_hittingBtwn_mem (s := Ici a) ⟨k, ⟨Nat.zero_le _, hk⟩, hka⟩
    · rw [indicator_of_notMem hω]
      exact hpos _ _
  have hi := integral_mono ((integrable_const a).indicator hA) hInt hpoint
  have hreal : a * P.real A ≤ ∫ ω, Z 0 ω ∂P := by
    simpa only [integral_indicator hA, setIntegral_const, smul_eq_mul, mul_comm] using hi.trans hmean
  have hconvert : ENNReal.ofReal (a * P.real A) = ENNReal.ofReal a * P A := by
    rw [ENNReal.ofReal_mul ha, Measure.real, ENNReal.ofReal_toReal (measure_ne_top P A)]
  rw [← hconvert]
  exact ENNReal.ofReal_le_ofReal hreal

/-- Ville's bound holds over all natural times, for the actual crossing
event. Source: arXiv:2604.01978v1, §7, the maximal-inequality step. -/
theorem nonnegative_supermartingale_ville_nat {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℕ m) (Z : ℕ → Ω → ℝ)
    (hZ : Supermartingale Z ℱ P) (hpos : ∀ k ω, 0 ≤ Z k ω) (a : ℝ) (ha : 0 ≤ a) :
    ENNReal.ofReal a * P {ω | ∃ k, a ≤ Z k ω} ≤ ENNReal.ofReal (∫ ω, Z 0 ω ∂P) := by
  let A : ℕ → Set Ω := fun N => {ω | ∃ k ≤ N, a ≤ Z k ω}
  have hmono : Monotone A := by
    intro N M hNM ω hω
    obtain ⟨k, hk, hka⟩ := hω
    exact ⟨k, hk.trans hNM, hka⟩
  have heq : {ω | ∃ k, a ≤ Z k ω} = ⋃ N, A N := by
    ext ω
    simp only [mem_iUnion, A, mem_ofPred_eq]
    constructor
    · rintro ⟨k, hk⟩
      exact ⟨k, k, le_rfl, hk⟩
    · rintro ⟨N, k, hk, hka⟩
      exact ⟨k, hka⟩
  rw [heq, hmono.measure_iUnion, ENNReal.mul_iSup]
  exact iSup_le fun N => nonnegative_supermartingale_ville_finite P ℱ Z hZ hpos a ha N

/-- At a positive threshold the same result has the usual division form.
Source: arXiv:2604.01978v1, §7, the martingale maximal inequality. -/
theorem nonnegative_supermartingale_ville_nat_div {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℕ m) (Z : ℕ → Ω → ℝ)
    (hZ : Supermartingale Z ℱ P) (hpos : ∀ k ω, 0 ≤ Z k ω) (a : ℝ) (ha : 0 < a) :
    P {ω | ∃ k, a ≤ Z k ω} ≤ ENNReal.ofReal (∫ ω, Z 0 ω ∂P) / ENNReal.ofReal a := by
  apply (ENNReal.le_div_iff_mul_le (.inl (by positivity)) (.inl ENNReal.ofReal_ne_top)).2
  simpa only [mul_comm] using nonnegative_supermartingale_ville_nat P ℱ Z hZ hpos a ha.le

/-- A bound for the actual initial expectation gives the corresponding
maximal bound, without any assumption on later tail probabilities.
Source: arXiv:2604.01978v1, §7, the martingale maximal inequality. -/
theorem nonnegative_supermartingale_ville_nat_of_initial_le
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℕ m) (Z : ℕ → Ω → ℝ) (hZ : Supermartingale Z ℱ P)
    (hpos : ∀ k ω, 0 ≤ Z k ω) (a : ℝ) (ha : 0 ≤ a) (C : ℝ)
    (hC : (∫ ω, Z 0 ω ∂P) ≤ C) :
    ENNReal.ofReal a * P {ω | ∃ k, a ≤ Z k ω} ≤ ENNReal.ofReal C := by
  exact (nonnegative_supermartingale_ville_nat P ℱ Z hZ hpos a ha).trans
    (ENNReal.ofReal_le_ofReal hC)

/-- A strictly decreasing positive supermartingale and an actual bounded
stopping time satisfy every positivity and optional-stopping premise. -/
example : ∃ Z : ℕ → Unit → ℝ,
    Supermartingale Z (Filtration.const (ι := ℕ) ⊥ bot_le) (Measure.dirac ()) ∧
    (∀ k ω, 0 ≤ Z k ω) ∧
    IsStoppingTime (Filtration.const (ι := ℕ) (m := inferInstance) ⊥ bot_le)
      (fun _ : Unit => (1 : WithTop ℕ)) ∧
    (∀ ω : Unit, (fun _ : Unit => (1 : WithTop ℕ)) ω ≤ 1) ∧
    (0 : ℝ) < 1 ∧ Z 0 () = 1 ∧ Z 1 () < 1 := by
  let Z : ℕ → Unit → ℝ := fun k _ => Real.exp (-(k : ℝ))
  let ℱ := Filtration.const (ι := ℕ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  have hZ : Supermartingale Z ℱ (Measure.dirac ()) := by
    refine ⟨fun k => stronglyMeasurable_const, ?_, fun k => integrable_const _⟩
    intro i j hij
    rw [condExp_const (ℱ.le i)]
    apply ae_of_all
    intro ω
    apply Real.exp_le_exp.mpr
    have hijR : (i : ℝ) ≤ j := by exact_mod_cast hij
    linarith
  refine ⟨Z, hZ, fun k ω => Real.exp_nonneg _, ?_, fun ω => le_rfl, one_pos, ?_, ?_⟩
  · exact isStoppingTime_const ℱ (1 : ℕ)
  · simp [Z]
  · norm_num [Z]

end Transformer.Homogenized
