module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TrainingTransport
public import Transformer.Grokking.Geometry.ProbabilityTransfer

/-!
# Training-anchor odds from actual linear-readout geometry

Source: `review/flat_spectrum.md`, *Probability transfer*, PT3.
The query state and training states use the same actual final normalization.
Cauchy–Schwarz bounds each change of class contrast. Training confidence and
the state distance then give a probability/loss bound through PT2. Anchor
selection can depend on the unlabeled query state and differ by competitor.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset Geometry Landscape.Neural

variable {I C K : Type*} [Fintype K]

/-- A reference formed by averaging normalized training states has the mean
training logit, PT3. Softmax itself does not commute with that average. -/
theorem linearScore_mean [Fintype I] (w : K → ℝ) (T : I → K → ℝ) :
    linearScore w (fun k => (∑ i, T i k) / Fintype.card I) =
      (∑ i, linearScore w (T i)) / Fintype.card I := by
  unfold linearScore
  simp_rw [← mul_div_assoc]
  rw [← sum_div]
  congr 1
  simp_rw [mul_sum]
  exact sum_comm

/-- Training log odds plus a contrast-specific distance penalty, PT3. The
selected reference may also be a computed center of training states. -/
def anchorLogOddsBound (E : C → K → ℝ) (T : I → K → ℝ) (x : K → ℝ)
    (t : C) (anchor : C → I) (c : C) : ℝ :=
  linearScore (E c) (T (anchor c)) - linearScore (E t) (T (anchor c)) +
    Real.sqrt ((∑ k, (x k - T (anchor c) k) ^ 2) * ∑ k, (E c k - E t k) ^ 2)

/-- The relative query logit is bounded from states and training logits;
the right side does not evaluate a query class score, PT3. -/
theorem relative_logit_le_anchor (E : C → K → ℝ) (T : I → K → ℝ) (x : K → ℝ)
    (t : C) (anchor : C → I) (c : C) :
    linearScore (E c) x - linearScore (E t) x ≤ anchorLogOddsBound E T x t anchor c := by
  have h := (abs_le.mp (center_contrast_error_le (T (anchor c)) x
    (fun k => E c k - E t k))).2
  simp only [linearScore, sub_mul, sum_sub_distrib] at h
  unfold anchorLogOddsBound linearScore
  linarith

/-- **A probability transfer bound derived from training-state geometry**,
PT3. No query label, query probability error or inverse condition is a premise. -/
theorem probability_ge_anchor_bound [Fintype C] [DecidableEq C]
    (E : C → K → ℝ) (T : I → K → ℝ) (x : K → ℝ) (t : C) (anchor : C → I) :
    1 / (1 + oddsEnvelope (anchorLogOddsBound E T x t anchor) t) ≤
      probability (fun c => linearScore (E c) x) t := by
  apply probability_ge_of_contrast_bounds
  intro c hc
  exact relative_logit_le_anchor E T x t anchor c

/-- The training-anchor geometry also controls the query's actual CE, PT3. -/
theorem crossEntropy_le_anchor_bound [Fintype C] [DecidableEq C]
    (E : C → K → ℝ) (T : I → K → ℝ) (x : K → ℝ) (t : C) (anchor : C → I) :
    crossEntropy (fun c => linearScore (E c) x) t ≤
      Real.log (1 + oddsEnvelope (anchorLogOddsBound E T x t anchor) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact relative_logit_le_anchor E T x t anchor c

/-- The state distance is nonzero in this positive geometric witness, PT5. -/
theorem nonzero_anchor_geometry_witness :
    let E : Bool → Fin 1 → ℝ := fun c _ => if c then 2 else 0
    let T : Fin 1 → Fin 1 → ℝ := fun _ _ => 2
    let x : Fin 1 → ℝ := fun _ => 3 / 2
    anchorLogOddsBound E T x true (fun _ => 0) false = -3 ∧
      oddsEnvelope (anchorLogOddsBound E T x true (fun _ => 0)) true < 1 := by
  norm_num [anchorLogOddsBound, linearScore, oddsEnvelope]

example := relative_logit_le_anchor (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℝ)) (fun _ => 3 / 2) true (fun _ => 0) false
example := probability_ge_anchor_bound (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℝ)) (fun _ => 3 / 2) true (fun _ => 0)
example := crossEntropy_le_anchor_bound (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℝ)) (fun _ => 3 / 2) true (fun _ => 0)
example := linearScore_mean (fun _ : Fin 1 => (3 : ℝ))
  (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1)

end Transformer.Grokking.FlatSpectrum.FeatureCodes
