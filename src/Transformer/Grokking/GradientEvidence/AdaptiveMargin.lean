module

public import Transformer.Grokking.GradientEvidence.JointLoss

/-!
# Joint margin control for the actual buffered Adam update

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC2–JC3;
native optimizer source 14cff3b5, engine::update and product::moment.
Moments are the updated buffers. Each class retains its own preconditioner.
Parameter rounding and execution discrepancies remain explicit; no stored
moment is identified with the partial readout gradient.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

variable {C K : Type*}

/-- Actual ideal-real Adam readout increment, including weight decay, JC2.
It uses the updated buffers, not a substituted current gradient. -/
def adamReadoutIncrement (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v : C → K → ℝ) (c : C) (k : K) : ℝ :=
  -η * (decay * E c k + adamDiagonal b2 eps n (v c k) * correctedMoment b1 n (m c k))

/-- The declared increment reconstructs the source buffered update, JC2. -/
theorem adam_increment_update (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v : C → K → ℝ) (c : C) (k : K) :
    E c k + adamReadoutIncrement b1 b2 eps η decay n E m v c k =
      bufferedAdamUpdate b1 b2 eps η decay n (E c k) (m c k) (v c k) := by
  rw [bufferedAdamUpdate_factor]
  unfold adamReadoutIncrement
  ring

/-- Contrast forcing subtracts two separately preconditioned class rows,
JC2. No common diagonal scale is assumed. -/
theorem adam_increment_contrast (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v : C → K → ℝ) (t c : C) (k : K) :
    classContrast (adamReadoutIncrement b1 b2 eps η decay n E m v) t c k =
      -η * (decay * classContrast E t c k + bufferedContrastDirection b1 b2 eps n m v c t k) := by
  unfold classContrast adamReadoutIncrement bufferedContrastDirection
  ring

/-- Exact-real source update plus the retained parameter rounding, JC2. -/
theorem adam_readout_step (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v ρ : C → K → ℝ) :
    (fun c => scoreStep (E c) (adamReadoutIncrement b1 b2 eps η decay n E m v c) (ρ c)) =
      (fun c k => bufferedAdamUpdate b1 b2 eps η decay n (E c k) (m c k) (v c k) + ρ c k) := by
  funext c k
  unfold scoreStep
  rw [adam_increment_update]

variable [Fintype K]

/-- The actual buffered step satisfies the joint competitor-gap bound,
JC2. Feature movement and both numerical discrepancies are retained. -/
theorem adam_gap_le_joint_upper (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v ρ : C → K → ℝ) (x dx : K → ℝ) (e e' : C → ℝ) (t c : C) :
    readoutLogits (fun a k => bufferedAdamUpdate b1 b2 eps η decay n
      (E a k) (m a k) (v a k) + ρ a k) (fun k => x k + dx k) e' c -
      readoutLogits (fun a k => bufferedAdamUpdate b1 b2 eps η decay n
        (E a k) (m a k) (v a k) + ρ a k) (fun k => x k + dx k) e' t ≤
      jointGapUpper E (adamReadoutIncrement b1 b2 eps η decay n E m v) ρ x dx e e' t c := by
  rw [← adam_readout_step]
  exact new_gap_le_joint_upper E _ ρ x dx e e' t c

/-- A joint budget controls every actual new competitor margin under Adam,
JC2. It is a sufficient local condition, not a convergence assertion. -/
theorem adam_margin_of_joint_condition (b1 b2 eps η decay : ℝ) (n : ℕ)
    (E m v ρ : C → K → ℝ) (x dx : K → ℝ) (e e' : C → ℝ) (t : C) (margin : ℝ)
    (h : ∀ c, c ≠ t →
      jointGapUpper E (adamReadoutIncrement b1 b2 eps η decay n E m v) ρ x dx e e' t c ≤ -margin) :
    ∀ c, c ≠ t → margin ≤
      readoutLogits (fun a k => bufferedAdamUpdate b1 b2 eps η decay n
        (E a k) (m a k) (v a k) + ρ a k) (fun k => x k + dx k) e' t -
      readoutLogits (fun a k => bufferedAdamUpdate b1 b2 eps η decay n
        (E a k) (m a k) (v a k) + ρ a k) (fun k => x k + dx k) e' c := by
  rw [← adam_readout_step]
  exact all_competitor_margin E _ ρ x dx e e' t margin h

example := adam_margin_of_joint_condition 0 0 1 (1 / 10) 1 1
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun c _ => if c then 1 else 2) (fun c _ => if c then 1 else 4)
  (fun _ _ => 0) (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 0) true 2
  (by intro c hc; cases c <;> simp_all [jointGapUpper, jointGapDrift, gapStepRadius,
    scoreStepRadius, jointScoreDrift, readoutLogits, classContrast, adamReadoutIncrement,
    FlatSpectrum.FeatureCodes.linearScore, pairBudget, adamDiagonal, correctedMoment]; norm_num)

end Transformer.Grokking.GradientEvidence
