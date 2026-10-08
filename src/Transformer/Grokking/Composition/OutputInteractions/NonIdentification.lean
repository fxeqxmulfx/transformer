import Transformer.Grokking.Composition.OutputInteractions.Basic

/-!
# What four output corners identify and what they miss

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition; diagnostic refinement of AdditiveContrast at 6ca0b3a.

Additive class scores, including nonlinear common row shifts, have zero
class-centered contrast. A genuine bilinear binary logit has energy
`(a*b)^2/2`. Conversely, a stated polynomial gate map has zero interaction
at all removal endpoints while its interior interaction is nonzero. Thus
zero endpoint contrast does not establish global additivity. Positive
energy likewise does not identify attention instead of downstream FFN or
normalization nonlinearities. These are explicit finite-output examples.

Neither probability of initialization, transformer training, held-out
division accuracy nor a numerical-kernel certificate is assumed or proved.
-/

namespace Transformer.Grokking.Composition.OutputInteractions

open Transformer.Grokking.Geometry
open scoped BigOperators

variable {C : Type*}

/-- Affine class outputs have zero centered interaction even with arbitrary
nonlinear common row offsets. Source: the pre-loss refinement of
arXiv:2301.05217v1's appendix composition diagnostic; this concerns the
specified output map rather than the transformer's trained parameters. -/
theorem centeredInteraction_affine_shift (cs : Finset C) (base left right : C → ℝ)
    (shift : ℝ → ℝ → ℝ) (a b : ℝ) (hc : cs.Nonempty) (c : C) :
    centeredInteraction cs
      (fun u v k => base k + u * left k + v * right k + shift u v) a b c = 0 := by
  rw [centeredInteraction_shift cs _ shift a b hc]
  have hz : (fun k => scoreInteraction
      (fun u v => base k + u * left k + v * right k) a b) = fun _ => 0 := by
    funext k
    unfold scoreInteraction
    ring
  unfold centeredInteraction
  rw [hz]
  unfold centerClasses
  rw [meanOver_const cs 0 hc]
  ring

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨false, by norm_num⟩

/-- The corresponding actual squared energy is zero. Source:
arXiv:2301.05217v1 appendix-inspired interaction refinement; affine
outputs are a sufficient condition, not a consequence of one zero result. -/
theorem interactionEnergy_affine_shift (cs : Finset C) (base left right : C → ℝ)
    (shift : ℝ → ℝ → ℝ) (a b : ℝ) (hc : cs.Nonempty) :
    interactionEnergy cs
      (fun u v k => base k + u * left k + v * right k + shift u v) a b = 0 := by
  apply (interactionEnergy_zero_iff _ _ _ _).mpr
  intro c _
  exact centeredInteraction_affine_shift cs base left right shift a b hc c

example : ({0, 1} : Finset ℕ).Nonempty := by
  exact ⟨1, by norm_num⟩

/-- Actual two-class logits from a scalar product. Source: the explicit
bilinear CE specialization of arXiv:2301.05217v1's appendix composition
hypothesis; the class center and energy are computed separately. -/
def binaryProductScores (a b : ℝ) (k : Fin 2) : ℝ :=
  if k = 0 then a * b else 0

/-- The four raw product-logit corners retain the product. Source:
arXiv:2301.05217v1 appendix compositional hypothesis, explicit two-class
specialization; this identifies the chosen score map, not a learned head. -/
theorem binaryProductScores_contrast (a b : ℝ) (k : Fin 2) :
    scoreInteraction (fun u v => binaryProductScores u v k) a b =
      binaryProductScores a b k := by
  by_cases h : k = 0
  · simp [scoreInteraction, binaryProductScores, h]
  · simp [scoreInteraction, binaryProductScores, h]

/-- Actual finite-class centering gives the product energy with factor one
half. Source: arXiv:2301.05217v1 appendix-inspired score diagnostic;
the factor is derived from both class coordinates and their actual mean. -/
theorem binaryProductScores_energy (a b : ℝ) :
    interactionEnergy Finset.univ binaryProductScores a b = (a * b) ^ 2 / 2 := by
  unfold interactionEnergy energyOver centeredInteraction centerClasses
  simp only [binaryProductScores_contrast]
  unfold meanOver binaryProductScores
  simp only [Fin.sum_univ_two]
  norm_num
  ring

/-- Product interaction is positive precisely when both components supply
a nonzero product. Source: arXiv:2301.05217v1 appendix composition
hypothesis, explicit two-class map; positivity alone is not task success. -/
theorem binaryProductScores_energy_pos_iff (a b : ℝ) :
    0 < interactionEnergy Finset.univ binaryProductScores a b ↔ a * b ≠ 0 := by
  rw [binaryProductScores_energy]
  constructor
  · intro h hz
    rw [hz] at h
    norm_num at h
  · intro h
    have hs := sq_pos_iff.mpr h
    positivity

/-- A stated nonlinear gate map with zero outputs at binary endpoints.
Source comparison: arXiv:2301.05217v1 appendix composition diagnostic;
explicit deviation: polynomial gates, rather than actual attention heads. -/
def hiddenGateScores (a b : ℝ) (k : Fin 2) : ℝ :=
  if k = 0 then a * b * (1 - a) * (1 - b) else 0

/-- All four raw corners still compute the chosen polynomial, because
each coordinate axis output is zero. Source: the explicit counterexample
to global identification from the appendix-inspired diagnostic. -/
theorem hiddenGateScores_contrast (a b : ℝ) (k : Fin 2) :
    scoreInteraction (fun u v => hiddenGateScores u v k) a b =
      hiddenGateScores a b k := by
  by_cases h : k = 0
  · simp [scoreInteraction, hiddenGateScores, h]
  · simp [scoreInteraction, hiddenGateScores, h]

/-- Compute the actual centered energy of the nonlinear gate example.
Source: the explicit four-corner identification counterexample compared
with arXiv:2301.05217v1's appendix composition hypothesis. -/
theorem hiddenGateScores_energy (a b : ℝ) :
    interactionEnergy Finset.univ hiddenGateScores a b =
      (a * b * (1 - a) * (1 - b)) ^ 2 / 2 := by
  unfold interactionEnergy energyOver centeredInteraction centerClasses
  simp only [hiddenGateScores_contrast]
  unfold meanOver hiddenGateScores
  simp only [Fin.sum_univ_two]
  norm_num
  ring

/-- Removal endpoints give zero while an interior gate pair gives strictly
positive interaction in the same forward map. Source: the explicit
counterexample to global additivity identification from four gate states. -/
theorem zero_endpoint_interaction_with_positive_interior :
    interactionEnergy Finset.univ hiddenGateScores 1 1 = 0 ∧
      interactionEnergy Finset.univ hiddenGateScores (1 / 2) (1 / 2) = 1 / 512 := by
  rw [hiddenGateScores_energy, hiddenGateScores_energy]
  constructor <;> norm_num

/-- An interior pair of gates can interact when the complete endpoint
removal diagnostic vanishes. Source: arXiv:2301.05217v1 appendix-inspired
diagnostic counterexample; no inference to all gate values is permitted. -/
theorem endpoint_zero_does_not_imply_interior_zero :
    interactionEnergy Finset.univ hiddenGateScores 1 1 = 0 ∧
      ∃ a b : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧
        0 < interactionEnergy Finset.univ hiddenGateScores a b := by
  refine ⟨zero_endpoint_interaction_with_positive_interior.1, 1 / 2, 1 / 2,
    by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  rw [zero_endpoint_interaction_with_positive_interior.2]
  norm_num

end Transformer.Grokking.Composition.OutputInteractions
