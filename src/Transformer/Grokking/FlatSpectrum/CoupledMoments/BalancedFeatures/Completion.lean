module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement

/-!
# Explicit completion of zero features and unobserved phase axes

Source: `review/flat_spectrum.md`, CM24. A declared constant feature row makes
normalization total. Phase choices outside the observed axes preserve actual
observed CE and the same latent spectral budgets, without identifying labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures

open Finset Matrix DensityExtraction DensityBounds Schatten ObservedRefinement

variable {ι X : Type*} [Fintype ι]

/-- Normalize nonzero rows; explicitly replace zero rows by all ones, CM24. -/
noncomputable def completeBalance (B : Matrix X ι ℂ) : Matrix X ι ℂ :=
  fun x i => if rowEnergy B x = 0 then 1 else normalize B x i

/-- Even a completed row has the declared measured energy, CM24. The zero-row
branch is a chosen feature convention, not a recovered hidden Fourier row. -/
theorem completed_energy (B : Matrix X ι ℂ) (x : X) :
    rowEnergy (completeBalance B) x = Fintype.card ι := by
  by_cases h : rowEnergy B x = 0
  · change (∑ i, Complex.normSq (completeBalance B x i)) = _
    simp [completeBalance, h]
  · have he : rowEnergy (completeBalance B) x = rowEnergy (normalize B) x := by
      change (∑ i, Complex.normSq (completeBalance B x i)) = _
      simp only [completeBalance, h, ↓reduceIte]
      rfl
    rw [he]
    exact normalized_energy B x (lt_of_le_of_ne (energy_nonneg B x) (Ne.symm h))

/-- The explicitly completed domain always admits the uniform density, CM24. -/
theorem completed_feasible [DecidableEq ι] [Nonempty ι] (B : Matrix X ι ℂ) :
    DensityExtraction.Feasible (completeBalance B) uniformDensity := by
  refine ⟨uniform_density, fun x => ?_⟩
  rw [uniform_reading, completed_energy]
  have hd : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [hd]

variable {S n C : Type*} [Fintype S] [Fintype n] [DecidableEq n]
  [Fintype C] [DecidableEq C] {q : ℕ} [NeZero q]

/-- A declared fallback fills only unobserved axis entries, CM24. -/
noncomputable def completeAxis (axis : S → n) (code fallback : n → ZMod q) : n → ZMod q :=
  fun i => if ∃ t, axis t = i then code i else fallback i

omit [Fintype n] [NeZero q] in
/-- Completion preserves every observed phase exactly, CM24. -/
theorem completeAxis_observed (axis : S → n) (code fallback : n → ZMod q) (t : S) :
    completeAxis axis code fallback (axis t) = code (axis t) := by
  simp [completeAxis, show ∃ u, axis u = axis t from ⟨t, rfl⟩]

/-- Arbitrary changes outside the observed axes preserve ordinary full-vocabulary
CE and remain in the same convex decoded spectral hull, CM24. -/
theorem phase_completion (d : ZMod q → C) (r s r' s' : n → ZMod q)
    (cell : S → n × n) (labels : S → C)
    (hr : ∀ t, r' (cell t).1 = r (cell t).1)
    (hs : ∀ t, s' (cell t).2 = s (cell t).2) :
    pointMass (fun i j => d (r' i + s' j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
      loss cell labels (pointMass (fun i j => d (r' i + s' j))) =
        loss cell labels (pointMass (fun i j => d (r i + s j))) := by
  refine ⟨phase_member d r' s', ?_⟩
  rw [pointMass_loss, pointMass_loss]
  congr 1
  unfold errors
  apply sum_congr rfl
  intro t ht
  simp only [hr t, hs t]

/-- The observable fallback construction satisfies the preceding CE equality;
no decoder coverage or true unseen phase is supplied as a hypothesis, CM24. -/
theorem completed_phase_loss (d : ZMod q → C) (r s fr fs : n → ZMod q)
    (cell : S → n × n) (labels : S → C) :
    loss cell labels (pointMass (fun i j =>
      d (completeAxis (fun t => (cell t).1) r fr i +
        completeAxis (fun t => (cell t).2) s fs j))) =
      loss cell labels (pointMass (fun i j => d (r i + s j))) := by
  exact (phase_completion d r s _ _ cell labels
    (fun t => completeAxis_observed _ r fr t)
    (fun t => completeAxis_observed _ s fs t)).2

example := completed_energy (fun x i : Bool => if x then (if i then 3 else 4) else 0) false
example := completed_feasible (fun x i : Bool => if x then (if i then 3 else 4) else 0)
example := completeAxis_observed (fun _ : Unit => false)
  (fun i : Bool => if i then (1 : ZMod 2) else 0) (fun _ => 0) ()
example := phase_completion (id : ZMod 2 → ZMod 2)
  (fun _ : Bool => 0) (fun _ => 0) (fun i => if i then 1 else 0) (fun _ => 0)
  (fun _ : Unit => (false, false)) (fun _ => 0) (by simp) (by simp)
example := completed_phase_loss (id : ZMod 2 → ZMod 2)
  (fun i : Bool => if i then 1 else 0) (fun _ => 0) (fun _ => 0) (fun _ => 1)
  (fun _ : Unit => (false, false)) (fun _ => 0)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures
