module

public import Transformer.Grokking.GradientEvidence.JointScoreStep

/-!
# All-competitor margins under joint parameter and feature changes

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC2.
Every competitor has its own signed increment and remainder. A bound for
each competitor controls the worst one even when its identity changes.
Execution errors and parameter rounding are distinct explicit inputs.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open FlatSpectrum.FeatureCodes

variable {C K : Type*} [Fintype K]

/-- Executed readout scores with an explicit numerical discrepancy, JC2. -/
def readoutLogits (E : C → K → ℝ) (x : K → ℝ) (e : C → ℝ) (c : C) : ℝ :=
  linearScore (E c) x + e c

/-- Competitor minus target, applied to weights, increments or rounding, JC2. -/
def classContrast (E : C → K → ℝ) (t c : C) (k : K) : ℝ := E c k - E t k

/-- The actual score gap uses the difference of the two rows, JC2. -/
theorem readout_gap (E : C → K → ℝ) (x : K → ℝ) (e : C → ℝ) (t c : C) :
    readoutLogits E x e c - readoutLogits E x e t =
      linearScore (classContrast E t c) x + (e c - e t) := by
  unfold readoutLogits classContrast linearScore
  simp only [sub_mul, Finset.sum_sub_distrib]
  ring

omit [Fintype K] in
/-- Contrast updates keep both rows' own increments, JC2. -/
theorem classContrast_step (E P ρ : C → K → ℝ) (t c : C) :
    classContrast (fun v => scoreStep (E v) (P v) (ρ v)) t c =
      scoreStep (classContrast E t c) (classContrast P t c) (classContrast ρ t c) := by
  funext k
  unfold classContrast scoreStep
  ring

/-- Signed joint contribution for one competing class, JC2. -/
def jointGapDrift (E P : C → K → ℝ) (x dx : K → ℝ) (t c : C) : ℝ :=
  jointScoreDrift (classContrast E t c) (classContrast P t c) x dx

/-- All finite-step remainders for one executed class contrast, JC2. -/
def gapStepRadius (P ρ : C → K → ℝ) (x dx : K → ℝ) (e e' : C → ℝ) (t c : C) : ℝ :=
  scoreStepRadius (classContrast P t c) (classContrast ρ t c) x dx
    (e c - e t) (e' c - e' t)

/-- Upper new gap from old scores and explicit changes, JC2. -/
def jointGapUpper (E P ρ : C → K → ℝ) (x dx : K → ℝ) (e e' : C → ℝ)
    (t c : C) : ℝ :=
  readoutLogits E x e c - readoutLogits E x e t + jointGapDrift E P x dx t c +
    gapStepRadius P ρ x dx e e' t c

/-- Lower new gap from the same signed drift and remainder, JC2. -/
def jointGapLower (E P ρ : C → K → ℝ) (x dx : K → ℝ) (e e' : C → ℝ)
    (t c : C) : ℝ :=
  readoutLogits E x e c - readoutLogits E x e t + jointGapDrift E P x dx t c -
    gapStepRadius P ρ x dx e e' t c

/-- Both sides of the per-class interval follow from the joint remainder,
JC2. No old or new winning class is assumed in this inequality. -/
theorem gap_step_error_bound (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t c : C) :
    |((readoutLogits (fun v => scoreStep (E v) (P v) (ρ v))
          (fun k => x k + dx k) e' c -
        readoutLogits (fun v => scoreStep (E v) (P v) (ρ v))
          (fun k => x k + dx k) e' t) -
      (readoutLogits E x e c - readoutLogits E x e t)) - jointGapDrift E P x dx t c| ≤
      gapStepRadius P ρ x dx e e' t c := by
  simp only [readout_gap, classContrast_step, jointGapDrift, gapStepRadius]
  exact score_step_error_bound _ _ _ x dx _ _

/-- The upper interval endpoint bounds every new competitor gap, JC2. -/
theorem new_gap_le_joint_upper (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t c : C) :
    readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' c -
      readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' t ≤
        jointGapUpper E P ρ x dx e e' t c := by
  have h := (abs_le.mp (gap_step_error_bound E P ρ x dx e e' t c)).2
  unfold jointGapUpper
  linarith

/-- The lower endpoint also controls each new gap, JC2. This permits
certifying deterioration as well as preservation, without dropping errors. -/
theorem joint_lower_le_new_gap (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t c : C) :
    jointGapLower E P ρ x dx e e' t c ≤
      readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' c -
        readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' t := by
  have h := (abs_le.mp (gap_step_error_bound E P ρ x dx e e' t c)).1
  unfold jointGapLower
  linarith

/-- A common margin floor holds against all competitors, including a new
worst competitor, JC2. The hypothesis is an explicit old-state/change budget. -/
theorem all_competitor_margin (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C) (m : ℝ)
    (h : ∀ c, c ≠ t → jointGapUpper E P ρ x dx e e' t c ≤ -m) :
    ∀ c, c ≠ t → m ≤
      readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' t -
        readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' c := by
  intro c hc
  have hu := new_gap_le_joint_upper E P ρ x dx e e' t c
  have hm := h c hc
  linarith

/-- Strict correctness from all negative upper gaps, JC2. This condition
does not require monotone residual norms or improved mean margins. -/
theorem strict_winner_of_joint_step (E P ρ : C → K → ℝ) (x dx : K → ℝ)
    (e e' : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → jointGapUpper E P ρ x dx e e' t c < 0) :
    ∀ c, c ≠ t →
      readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' c <
        readoutLogits (fun v => scoreStep (E v) (P v) (ρ v)) (fun k => x k + dx k) e' t := by
  intro c hc
  have hu := new_gap_le_joint_upper E P ρ x dx e e' t c
  have hh := h c hc
  linarith

example := all_competitor_margin
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun c _ => if c then 1 else 0) (fun _ _ => 0)
  (fun _ => 2) (fun _ => -1) (fun _ => 0) (fun _ => 0) true 2
  (by intro c hc; cases c <;> simp_all [jointGapUpper, jointGapDrift, gapStepRadius,
    scoreStepRadius, jointScoreDrift, readoutLogits, classContrast, linearScore, pairBudget]; norm_num)

example := strict_winner_of_joint_step
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun c _ => if c then 1 else 0) (fun _ _ => 0)
  (fun _ => 2) (fun _ => -1) (fun _ => 0) (fun _ => 0) true
  (by intro c hc; cases c <;> simp_all [jointGapUpper, jointGapDrift, gapStepRadius,
    scoreStepRadius, jointScoreDrift, readoutLogits, classContrast, linearScore, pairBudget]; norm_num)

end Transformer.Grokking.GradientEvidence
