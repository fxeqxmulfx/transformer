module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Decoded
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PhaseCandidate

/-!
# Latent phase tables with a fixed output decoder

Source: `review/flat_spectrum.md`, CM15. The latent phase construction survives
arbitrary fixed categorical output merging. A concrete decoded table lies
outside the direct binary rank-one budget. Two distinct decoded CE minima
show why the earlier observed-latent-label recovery theorem cannot be applied
to observed decoded labels without new identification hypotheses.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {q : ℕ} [NeZero q] {C n S : Type*}
  [Fintype C] [DecidableEq C] [Fintype n] [DecidableEq n] [Fintype S]

/-- Decoded training phase consistency constructs a feasible global ordinary-CE
minimum with the original output vocabulary and latent spectral budgets (CM15). -/
theorem decodedPhase_minimum (d : ZMod q → C) (r s : n → ZMod q)
    (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, d (r (cell t).1 + s (cell t).2) = labels t) :
    DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n)
        (pointMass (fun i j => d (r i + s j))) ∧
      IsMinOn (loss cell labels)
        {Q | DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q}
        (pointMass (fun i j => d (r i + s j))) := by
  have hp := phaseCandidate_budget (Equiv.refl (ZMod q)) r s
  have hf : ∀ t, decode d (pointMass (phaseLabels (Equiv.refl (ZMod q)) r s))
      (cell t).1 (cell t).2 = ConvexRecall.targetLogits (labels t) := by
    intro t
    rw [decode_pointMass]
    funext c
    simp [pointMass, phaseLabels, hfit t, ConvexRecall.targetLogits]
  simpa only [decode_pointMass, phaseLabels, Equiv.refl_symm, Equiv.refl_apply] using
    decoded_fit_minimum d _ _ _ cell labels hp hf

/-- The categorical three-equal-corner table (CM15). -/
def cornerLabels (i j : Fin 2) : Bool := decide (i = 0 ∨ j = 0)

/-- No injective additive class code represents the corner table (CM15).
This is a capacity obstruction for every latent additive group, not an optimizer claim. -/
theorem corner_no_injective_phase {A : Type*} [AddCancelCommMonoid A]
    (e : Bool → A) (he : Function.Injective e) (r s : Fin 2 → A) :
    ¬ ∀ i j, e (cornerLabels i j) = r i + s j := by
  intro h
  have h00 : e true = r 0 + s 0 := by simpa [cornerLabels] using h 0 0
  have h01 : e true = r 0 + s 1 := by simpa [cornerLabels] using h 0 1
  have h10 : e true = r 1 + s 0 := by simpa [cornerLabels] using h 1 0
  have h11 : e false = r 1 + s 1 := by simpa [cornerLabels] using h 1 1
  have hr : r 0 = r 1 := add_right_cancel (h00.symm.trans h10)
  have heq := he (h01.trans (hr ▸ h11.symm))
  cases heq

/-- The direct binary sign moment's nuclear budget excludes that table (CM15). -/
theorem corner_direct_excluded :
    ¬ BudgetFeasible (fun _ : Fin 1 => signCode) (fun _ => 2) (fun _ => 2)
      (pointMass cornerLabels) := by
  intro h
  have hx := rankOne_missing_probability (rows_pointMass cornerLabels)
    (by simp [pointMass, cornerLabels]) (by simp [pointMass, cornerLabels])
    (by simp [pointMass, cornerLabels]) (h.2 0)
  norm_num [pointMass, cornerLabels] at hx

/-- Three latent states merge into two original output labels (CM15). -/
def cornerDecoder (z : ZMod 3) : Bool := decide (z ≠ 2)

/-- Two distinct categorical phases, without prescribing input arithmetic (CM15). -/
def cornerPhase (i : Fin 2) : ZMod 3 := i.val

private theorem corner_decoding :
    (fun i j => cornerDecoder (cornerPhase i + cornerPhase j)) = cornerLabels := by
  funext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [cornerDecoder, cornerPhase, cornerLabels] <;> decide

private theorem corner_observed_fit (t : Fin 3) :
    cornerDecoder (cornerPhase (threeCells t).1 + cornerPhase (threeCells t).2) = true := by
  fin_cases t <;> norm_num [cornerDecoder, cornerPhase, threeCells] <;> decide

/-- The fixed three-state latent spectral family admits the excluded table (CM15). -/
theorem corner_decoded_admitted :
    DecodedFeasible cornerDecoder (phaseCode (Equiv.refl (ZMod 3)))
      (fun _ => 2) (fun _ => 2) (pointMass cornerLabels) := by
  simpa only [corner_decoding, Fintype.card_fin, Nat.cast_ofNat] using
    (decodedPhase_minimum cornerDecoder cornerPhase cornerPhase
      threeCells (fun _ => true) corner_observed_fit).1

/-- Same decoder and bounds, same observed ordinary CE, different unseen label (CM15).
Observed decoded labels do not determine a unique latent phase completion. -/
theorem corner_decoded_ambiguity :
    let domain := {Q : Table (Fin 2) (Fin 2) Bool |
      DecodedFeasible cornerDecoder (phaseCode (Equiv.refl (ZMod 3)))
        (fun _ => 2) (fun _ => 2) Q}
    pointMass cornerLabels ∈ domain ∧ pointMass (fun _ _ => true) ∈ domain ∧
      IsMinOn (loss threeCells (fun _ => true)) domain (pointMass cornerLabels) ∧
      IsMinOn (loss threeCells (fun _ => true)) domain (pointMass (fun _ _ => true)) ∧
      (pointMass cornerLabels : Table (Fin 2) (Fin 2) Bool) ≠ pointMass (fun _ _ => true) := by
  have hc := decodedPhase_minimum cornerDecoder cornerPhase cornerPhase
    threeCells (fun _ => true) corner_observed_fit
  have hz := decodedPhase_minimum cornerDecoder (fun _ : Fin 2 => 0) (fun _ => 0)
    threeCells (fun _ => true) (by intro t; norm_num [cornerDecoder]; decide)
  simp only [corner_decoding, Fintype.card_fin, Nat.cast_ofNat] at hc
  norm_num [cornerDecoder, show (0 : ZMod 3) ≠ 2 by decide] at hz
  refine ⟨hc.1, hz.1, hc.2, hz.2, ?_⟩
  intro he
  have hx := congrArg (fun P => P 1 1 true) he
  norm_num [pointMass, cornerLabels] at hx

/-- The ambiguity uses a connected observed graph with a perfect matching (CM15).
The missing assumption from CM13 is latent-label identification, not graph coverage. -/
theorem corner_decoded_graph_coverage :
    (patternGraph (Set.range threeCells)).Connected ∧
      ∃ σ : Equiv.Perm (Fin 2), ∀ i, (i, σ i) ∈ Set.range threeCells := by
  constructor
  · apply (rigid_iff_connected _).mp
    intro r s h i j
    have h00 := h (0, 0) ⟨0, rfl⟩
    have h01 := h (0, 1) ⟨1, rfl⟩
    have h10 := h (1, 0) ⟨2, rfl⟩
    fin_cases i <;> fin_cases j
    · exact h00
    · exact h01
    · exact h10
    · change r 1 + s 1 = 0
      linear_combination h01 + h10 - h00
  · refine ⟨Equiv.swap 0 1, ?_⟩
    intro i
    fin_cases i
    · exact ⟨1, by simp [threeCells]⟩
    · exact ⟨2, by simp [threeCells]⟩

example := decodedPhase_minimum cornerDecoder cornerPhase cornerPhase
  threeCells (fun _ => true) corner_observed_fit
example := corner_observed_fit 0
example := decoded_fit_minimum cornerDecoder (phaseCode (Equiv.refl (ZMod 3)))
  (fun _ => 2) (fun _ => 2) threeCells (fun _ => true)
  (by simpa using phaseCandidate_budget (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)
  (by
    intro t
    rw [decode_pointMass]
    funext c
    simp [pointMass, phaseLabels, corner_observed_fit t, ConvexRecall.targetLogits])
example := corner_no_injective_phase (fun c : Bool => if c then (1 : ℤ) else 0)
  (by intro a b h; cases a <;> cases b <;> norm_num at *)
  (fun i : Fin 2 => i.val) (fun j : Fin 2 => j.val)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
