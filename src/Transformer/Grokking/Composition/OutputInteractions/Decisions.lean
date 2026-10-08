import Transformer.Grokking.Composition.OutputInteractions.NonIdentification
import Transformer.Grokking.Geometry.Decisions

/-!
# Decision meaning of the hypothetical additive output

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition; Prieto et al., arXiv:2501.04697v1 §4.2, confidence versus
decisions. The three-corner reconstruction is an explicit diagnostic,
not an architecture or a learned circuit identified by those papers.

Zero class-centered interaction preserves actual CE and every strict
correct-class comparison at the measured state. A positive current
reference margin survives sufficiently small absolute per-row interaction
energy. Neither a mean energy over examples nor a future margin is supplied
by that theorem. A concrete positive-interaction example nevertheless
preserves its correct decision, so interaction is not a decision change.

No numerical rounding bound, transformer training convergence or causal
grokking detector follows from this finite-output comparison.
-/

namespace Transformer.Grokking.Composition.OutputInteractions

open Transformer.Grokking.Geometry Transformer.Grokking.NaiveLoss
open scoped BigOperators

variable {C : Type*}

/-- Output reconstructed from the other three actual gate corners.
Source: the explicit pre-loss refinement of arXiv:2301.05217v1's appendix
composition diagnostic; this is hypothetical arithmetic on fixed outputs. -/
def additiveReconstruction (f : ℝ → ℝ → C → ℝ) (a b : ℝ) (k : C) : ℝ :=
  f 0 b k + f a 0 k - f 0 0 k

/-- The actual raw interaction is exactly reconstruction error. Source:
arXiv:2301.05217v1 appendix-inspired output diagnostic; the three-corner
map is computed rather than defined to satisfy a task loss or prediction. -/
theorem scoreInteraction_eq_reconstruction_error (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (k : C) :
    scoreInteraction (fun u v => f u v k) a b = f a b k - additiveReconstruction f a b k := by
  unfold scoreInteraction additiveReconstruction
  ring

/-- Reconstruction retains a fixed class bias while raw interaction
cancels it. Source: arXiv:2301.05217v1 §5.1's reference-scoring policy;
this keeps the decision-relevant bias in the hypothetical output. -/
theorem additiveReconstruction_class_bias (f : ℝ → ℝ → C → ℝ) (bias : C → ℝ)
    (a b : ℝ) (k : C) :
    additiveReconstruction (fun u v c => f u v c + bias c) a b k =
      additiveReconstruction f a b k + bias k := by
  unfold additiveReconstruction
  ring

/-- Zero energy makes reconstruction differ only by the computed common
row offset. Source: arXiv:2301.05217v1 appendix-inspired diagnostic and
the actual class-centering refinement; all classes are measured here. -/
theorem zero_energy_reconstruction_shift [Fintype C] (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (he : interactionEnergy Finset.univ f a b = 0) :
    f a b = fun k => additiveReconstruction f a b k +
      meanOver Finset.univ (fun c => scoreInteraction (fun u v => f u v c) a b) := by
  funext k
  have hk := (interactionEnergy_zero_iff _ _ _ _).mp he k (Finset.mem_univ k)
  unfold centeredInteraction centerClasses at hk
  dsimp only at hk
  rw [scoreInteraction_eq_reconstruction_error] at hk
  linarith

example : interactionEnergy Finset.univ hiddenGateScores 1 1 = 0 :=
  zero_endpoint_interaction_with_positive_interior.1

/-- Zero centered interaction preserves every strict target comparison.
Source: arXiv:2501.04697v1 §4.2's decision/offset distinction, applied
to the explicit arXiv:2301.05217v1-inspired three-corner diagnostic. -/
theorem zero_energy_strict_decisions_iff [Fintype C] (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (y : C) (he : interactionEnergy Finset.univ f a b = 0) :
    StrictCorrect (f a b) y ↔ StrictCorrect (additiveReconstruction f a b) y := by
  rw [zero_energy_reconstruction_shift f a b he]
  exact strictCorrect_shift_iff _ _ _

example : interactionEnergy Finset.univ hiddenGateScores 1 1 = 0 :=
  zero_endpoint_interaction_with_positive_interior.1

/-- Actual finite-class CE is also unchanged at zero centered interaction.
Source: arXiv:2501.04697v1 §4.2, common logit offsets; the finite loss
retains its original target and no success is assumed. -/
theorem zero_energy_CE_eq [Fintype C] (f : ℝ → ℝ → C → ℝ) (a b : ℝ) (y : C)
    (he : interactionEnergy Finset.univ f a b = 0) :
    crossEntropy (f a b) y = crossEntropy (additiveReconstruction f a b) y := by
  rw [zero_energy_reconstruction_shift f a b he]
  exact crossEntropy_shift _ _ _

example : interactionEnergy Finset.univ hiddenGateScores 1 1 = 0 :=
  zero_endpoint_interaction_with_positive_interior.1

/-- Aligning reconstruction by the computed raw row offset leaves exactly
the interaction energy as squared error. Source: arXiv:2301.05217v1 §5.1's
reference/error policy, applied to the explicit appendix-inspired output
reconstruction. This is a per-row sum, not mean energy over examples. -/
theorem aligned_reconstruction_error_energy [Fintype C] (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) :
    energyOver Finset.univ (fun k => additiveReconstruction f a b k +
      meanOver Finset.univ (fun c => scoreInteraction (fun u v => f u v c) a b) - f a b k) =
        interactionEnergy Finset.univ f a b := by
  unfold energyOver interactionEnergy
  apply Finset.sum_congr rfl
  intro k _
  unfold centeredInteraction centerClasses additiveReconstruction scoreInteraction
  dsimp only
  ring

/-- A verified positive current reference margin survives a small
absolute interaction energy. Source: the restricted-output correctness
condition of arXiv:2301.05217v1 §5.1, now for the explicit reconstruction.
The current margin and strict per-row bound remain hypotheses. -/
theorem reconstruction_correct_of_margin_energy [Fintype C] (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (y : C) (margin : ℝ) (hm : MarginAtLeast (f a b) y margin)
    (hp : 0 < margin) (he : 2 * interactionEnergy Finset.univ f a b < margin ^ 2) :
    StrictCorrect (additiveReconstruction f a b) y := by
  let shift := meanOver Finset.univ (fun c => scoreInteraction (fun u v => f u v c) a b)
  have hs := strictCorrect_of_margin_and_energy
    (fun k => additiveReconstruction f a b k + shift) (f a b) y margin hm hp
    (by simpa only [shift, aligned_reconstruction_error_energy] using he)
  exact (strictCorrect_shift_iff (additiveReconstruction f a b) y shift).mp hs

example :
    MarginAtLeast (fun k : Fin 2 => if k = 0 then (3 + (1 : ℝ) / 4) else 0) 0 3 ∧
      0 < (3 : ℝ) ∧ 2 * interactionEnergy Finset.univ
        (fun (u v : ℝ) (k : Fin 2) => if k = 0 then 3 + u * v / 4 else 0) 1 1 < 3 ^ 2 := by
  refine ⟨?_, by norm_num, ?_⟩
  · intro k hk
    fin_cases k <;> norm_num at *
  · norm_num [interactionEnergy, energyOver, centeredInteraction, centerClasses,
      scoreInteraction, meanOver, Fin.sum_univ_two]

/-- Positive interaction and unchanged correct decisions coexist.
Source: arXiv:2501.04697v1 §4.2's decision/confidence distinction,
specialized to the arXiv:2301.05217v1-inspired output diagnostic. -/
theorem positive_interaction_without_decision_change :
    0 < interactionEnergy Finset.univ
      (fun (u v : ℝ) (k : Fin 2) => if k = 0 then 3 + u * v / 4 else 0) 1 1 ∧
      StrictCorrect (fun k : Fin 2 => if k = 0 then (3 + (1 : ℝ) / 4) else 0) 0 ∧
      StrictCorrect (additiveReconstruction
        (fun (u v : ℝ) (k : Fin 2) => if k = 0 then 3 + u * v / 4 else 0) 1 1) 0 := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [interactionEnergy, energyOver, centeredInteraction, centerClasses,
      scoreInteraction, meanOver, Fin.sum_univ_two]
  · intro k hk
    fin_cases k <;> norm_num at *
  · intro k hk
    fin_cases k <;> norm_num [additiveReconstruction] at *

end Transformer.Grokking.Composition.OutputInteractions
