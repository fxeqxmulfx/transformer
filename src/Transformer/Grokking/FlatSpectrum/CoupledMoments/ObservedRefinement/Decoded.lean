module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement.Assignment

/-!
# Decoder count updates and feasible phase refinement

Source: `review/flat_spectrum.md`, CM23. Count-maximizing decoder updates and
certified axis assignments decrease actual observed CE. All resulting phase
tables remain in the same convex decoded spectral hull, including imperfect
fits. The discrete search itself is not a jointly convex parametrization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement

open Finset Schatten

variable {L C S : Type*} [Fintype L] [DecidableEq L]
  [DecidableEq C] [Fintype S]

/-- Observed label counts in each proposed latent state (CM23). -/
noncomputable def labelCount (latent : S → L) (labels : S → C) (l : L) (c : C) : ℝ :=
  ∑ t, if latent t = l then (if labels t = c then 1 else 0) else 0

/-- Correct decoder counts and actual errors partition all observations (CM23). -/
theorem counts_errors (latent : S → L) (labels : S → C) (d : L → C) :
    (∑ l, labelCount latent labels l (d l)) + errors (fun t => d (latent t)) labels =
      Fintype.card S := by
  have hg : (∑ l, labelCount latent labels l (d l)) =
      ∑ t, if labels t = d (latent t) then (1 : ℝ) else 0 := by
    unfold labelCount
    rw [sum_comm]
    apply sum_congr rfl
    intro t ht
    simp
  rw [hg, errors, ← sum_add_distrib]
  calc
    _ = ∑ t : S, (1 : ℝ) := by
      apply sum_congr rfl
      intro t ht
      by_cases h : d (latent t) = labels t <;> simp [mistake, h, Ne.symm]
    _ = _ := by simp

/-- Replacing each decoder entry by an observed count maximizer cannot
increase error, including latent states with no observations (CM23). -/
theorem decoder_descent (latent : S → L) (labels : S → C) (a b : L → C)
    (hcount : ∀ l, labelCount latent labels l (b l) ≤ labelCount latent labels l (a l)) :
    errors (fun t => a (latent t)) labels ≤ errors (fun t => b (latent t)) labels := by
  have h := sum_le_sum fun l (_ : l ∈ (univ : Finset L)) => hcount l
  have ha := counts_errors latent labels a
  have hb := counts_errors latent labels b
  linarith

variable {q : ℕ} [NeZero q] {n : Type*} [Fintype n] [DecidableEq n]

/-- Every refined phase table remains in the common convex decoded family,
without any training-fit or subspace-exactness assumption (CM23). -/
theorem phase_member (d : ZMod q → C) (r s : n → ZMod q) :
    pointMass (fun i j => d (r i + s j)) ∈
      DecoderHull (phaseCode (Equiv.refl (ZMod q)))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n) := by
  apply decoderHull_member d _ _ _
  refine ⟨pointMass (phaseLabels (Equiv.refl (ZMod q)) r s),
    phaseCandidate_budget (Equiv.refl (ZMod q)) r s, ?_⟩
  simp only [decode_pointMass, phaseLabels, Equiv.refl_symm, Equiv.refl_apply]

variable [Fintype C]

/-- An actual row-assignment certificate yields feasible ordinary-CE descent.
The candidate row phases are categorical assignments through `code` (CM23). -/
theorem row_step (d : ZMod q → C) (code s : n → ZMod q)
    (cell : S → n × n) (labels : S → C) (u v : n → ℝ) (π σ : Equiv.Perm n)
    (hdual : ∀ i j, u i + v j ≤ assignmentCost (fun t => (cell t).1)
      (fun t k => d (code k + s (cell t).2)) labels i j)
    (htight : ∀ i, assignmentCost (fun t => (cell t).1)
      (fun t k => d (code k + s (cell t).2)) labels i (π i) = u i + v (π i)) :
    pointMass (fun i j => d (code (π i) + s j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
      loss cell labels (pointMass (fun i j => d (code (π i) + s j))) ≤
        loss cell labels (pointMass (fun i j => d (code (σ i) + s j))) := by
  refine ⟨phase_member d (fun i => code (π i)) s, ?_⟩
  apply (loss_le_iff cell labels _ _).mpr
  exact assignment_errors _ _ labels u v hdual π htight σ

/-- The independently certified column update has the same CE and feasibility
guarantee, with the two observation axes exchanged (CM23). -/
theorem column_step (d : ZMod q → C) (r code : n → ZMod q)
    (cell : S → n × n) (labels : S → C) (u v : n → ℝ) (π σ : Equiv.Perm n)
    (hdual : ∀ i j, u i + v j ≤ assignmentCost (fun t => (cell t).2)
      (fun t k => d (r (cell t).1 + code k)) labels i j)
    (htight : ∀ i, assignmentCost (fun t => (cell t).2)
      (fun t k => d (r (cell t).1 + code k)) labels i (π i) = u i + v (π i)) :
    pointMass (fun i j => d (r i + code (π j))) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
      loss cell labels (pointMass (fun i j => d (r i + code (π j)))) ≤
        loss cell labels (pointMass (fun i j => d (r i + code (σ j)))) := by
  refine ⟨phase_member d r (fun j => code (π j)), ?_⟩
  apply (loss_le_iff cell labels _ _).mpr
  exact assignment_errors _ _ labels u v hdual π htight σ

/-- A decoder refit changes no latent moments and gives observed CE descent
inside the same prediction hull (CM23). -/
theorem decoder_step (r s : n → ZMod q) (a b : ZMod q → C)
    (cell : S → n × n) (labels : S → C)
    (hcount : ∀ l, labelCount (fun t => r (cell t).1 + s (cell t).2) labels l (b l) ≤
      labelCount (fun t => r (cell t).1 + s (cell t).2) labels l (a l)) :
    pointMass (fun i j => a (r i + s j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
      loss cell labels (pointMass (fun i j => a (r i + s j))) ≤
        loss cell labels (pointMass (fun i j => b (r i + s j))) := by
  refine ⟨phase_member a r s, ?_⟩
  apply (loss_le_iff cell labels _ _).mpr
  exact decoder_descent _ labels a b hcount

example := counts_errors (id : Bool → Bool) not not
example := decoder_descent (id : Bool → Bool) not not id
  (by intro l; cases l <;> norm_num [labelCount])
example := phase_member cornerDecoder cornerPhase cornerPhase

example := row_step (fun z : ZMod 2 => z) (fun i : ZMod 2 => i) (fun _ => 0)
  (fun t : ZMod 2 => (t, 0)) (fun t => t) (fun _ => 0) (fun _ => 0)
  (Equiv.refl _) (Equiv.refl _)
  (by intro i j; simp [assignmentCost, mistake]; split_ifs <;> norm_num)
  (by intro i; simp [assignmentCost, mistake])
example := decoder_step (fun i : ZMod 2 => i) (fun _ => 0)
  (id : ZMod 2 → ZMod 2) (fun _ => 0) (fun t : ZMod 2 => (t, 0)) id
  (by intro l; simp [labelCount]; split_ifs <;> norm_num)
example := column_step (fun z : ZMod 2 => z) (fun _ => 0) (fun i : ZMod 2 => i)
  (fun t : ZMod 2 => (0, t)) (fun t => t) (fun _ => 0) (fun _ => 0)
  (Equiv.refl _) (Equiv.refl _)
  (by intro i j; simp [assignmentCost, mistake]; split_ifs <;> norm_num)
  (by intro i; simp [assignmentCost, mistake])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement
