module

public import Transformer.Grokking.GradientEvidence.RelativeLoss

/-!
# Query and fixed-subset cross-entropy bounds from the actual readout step

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC3–JC4.
The signed joint drift and explicit mixed/rounding allowance control actual
CE, and their ceilings average over any fixed query subset. A bottom decile
selected by old margins is one such subset, not a new-state selection.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes Landscape.Neural

variable {C K : Type*} [Fintype C] [Fintype K]

/-- The joint parameter/feature budget bounds the query's actual CE change
through old probability weights, JC3. -/
theorem loss_change_le_joint_step (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C) :
    crossEntropy (readoutLogits (fun c => scoreStep (E c) (P c) (ρ c))
      (fun k => x k + dx k) e') t - crossEntropy (readoutLogits E x e) t ≤
      Real.log (weightedExp (readoutLogits E x e)
        (fun c => jointGapDrift E P x dx t c + gapStepRadius P ρ x dx e e' t c)) := by
  apply loss_change_le_relative_bound
  intro c
  have h := new_gap_le_joint_upper E P ρ x dx e e' t c
  unfold jointGapUpper at h
  linarith

/-- A concrete joint-update moment condition is sufficient for query CE
nonincrease, JC3. Feature displacement remains an explicit controlled input. -/
theorem loss_nonincrease_of_joint_step (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C)
    (h : weightedExp (readoutLogits E x e)
      (fun c => jointGapDrift E P x dx t c + gapStepRadius P ρ x dx e e' t c) ≤ 1) :
    crossEntropy (readoutLogits (fun c => scoreStep (E c) (P c) (ρ c))
      (fun k => x k + dx k) e') t ≤ crossEntropy (readoutLogits E x e) t := by
  have hl := loss_change_le_joint_step E P ρ x dx e e' t
  have hu := Real.log_le_log (weightedExp_pos (readoutLogits E x e) _ t) h
  rw [Real.log_one] at hu
  linarith

variable [DecidableEq C]

/-- The actual CE is bounded by all new competitor-gap ceilings, JC3. -/
theorem crossEntropy_le_joint_step (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C) :
    crossEntropy (readoutLogits (fun c => scoreStep (E c) (P c) (ρ c))
      (fun k => x k + dx k) e') t ≤
        Real.log (1 + oddsEnvelope (jointGapUpper E P ρ x dx e e' t) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact new_gap_le_joint_upper E P ρ x dx e e' t c

/-- A lower CE bound also detects deterioration on difficult queries, JC3.
Its inputs are the same signed drift and explicit remainder. -/
theorem crossEntropy_ge_joint_step (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C) :
    Real.log (1 + oddsEnvelope (jointGapLower E P ρ x dx e e' t) t) ≤
      crossEntropy (readoutLogits (fun c => scoreStep (E c) (P c) (ρ c))
        (fun k => x k + dx k) e') t := by
  rw [crossEntropy_eq_log_odds]
  apply Real.log_le_log (by have := oddsEnvelope_nonneg (jointGapLower E P ρ x dx e e' t) t; linarith)
  have h : oddsEnvelope (jointGapLower E P ρ x dx e e' t) t ≤
      oddsEnvelope (fun c => readoutLogits (fun v => scoreStep (E v) (P v) (ρ v))
        (fun k => x k + dx k) e' c - readoutLogits (fun v => scoreStep (E v) (P v) (ρ v))
        (fun k => x k + dx k) e' t) t := by
    apply sum_le_sum
    intro c hc
    exact Real.exp_le_exp.mpr (joint_lower_le_new_gap E P ρ x dx e e' t c)
  linarith

/-- Independently valid ceilings control the mean loss on any fixed subset,
JC4. Old errors and difficult rows need not be omitted or assumed correct. -/
theorem meanCrossEntropy_le_joint_step {I : Type*} (rows : Finset I)
    (E P ρ : C → K → ℝ) (x dx : I → K → ℝ) (e e' : I → C → ℝ) (t : I → C) :
    meanCrossEntropy rows
      (fun i => readoutLogits (fun c => scoreStep (E c) (P c) (ρ c))
        (fun k => x i k + dx i k) (e' i)) t ≤
      (∑ i ∈ rows, Real.log (1 + oddsEnvelope
        (jointGapUpper E P ρ (x i) (dx i) (e i) (e' i) (t i)) (t i))) / rows.card := by
  apply meanCrossEntropy_le_of_contrast_bounds
  intro i hi c hc
  exact new_gap_le_joint_upper E P ρ (x i) (dx i) (e i) (e' i) (t i) c

example := loss_nonincrease_of_joint_step
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun c _ => if c then 1 else 0) (fun _ _ => 0)
  (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 0) true
  (by
    apply (weightedExp_mono _ _ (fun _ => 0) ?_).trans
      (le_of_eq (weightedExp_zero _ true))
    intro c
    cases c <;> norm_num [jointGapDrift, gapStepRadius, scoreStepRadius,
      jointScoreDrift, classContrast, linearScore, pairBudget])

end Transformer.Grokking.GradientEvidence
