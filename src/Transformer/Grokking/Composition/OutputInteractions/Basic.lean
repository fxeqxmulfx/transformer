import Transformer.Grokking.Composition.AdditiveContrast
import Transformer.Grokking.Geometry.RowAlignment

/-!
# Class-centered output interactions before cross-entropy

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition; scalar diagnostic counterexample in AdditiveContrast at
6ca0b3a. The source does not supply this identification criterion.

Explicit refinement: compute the four actual output corners before CE,
then subtract the arithmetic mean over classes. Independent common row
offsets are softmax-invisible and must not count as output interaction.
Squared centered energy vanishes exactly when the raw contrast is a
common class offset on the measured class set. Empty sets give no evidence.

These are finite-output identities, not proofs of a particular attention
algorithm, training dynamics or generalization. They do not identify all
interior gate values from four removal states or certify numerical kernels.
-/

namespace Transformer.Grokking.Composition.OutputInteractions

open Transformer.Grokking.Geometry
open scoped BigOperators

variable {C : Type*}

/-- Actual class mean subtraction. Source: row centering in the fixed-orbit
adaptation of arXiv:2301.05217v1 §5.1 and the appendix composition diagnostic;
the selected finite class set remains an explicit argument. -/
noncomputable def centerClasses (cs : Finset C) (z : C → ℝ) (c : C) : ℝ :=
  z c - meanOver cs z

/-- Four actual output corners, centered before nonlinear CE. Source:
the diagnostic refinement of arXiv:2301.05217v1's appendix composition
hypothesis; every corner uses the same explicitly supplied forward map. -/
noncomputable def centeredInteraction (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (c : C) : ℝ :=
  centerClasses cs (fun k => scoreInteraction (fun u v => f u v k) a b) c

/-- Unnormalized squared centered-output contrast. Source: the finite-output
refinement above; this is sensitivity energy, not a task success predicate. -/
noncomputable def interactionEnergy (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) : ℝ := energyOver cs (centeredInteraction cs f a b)

/-- Class mean subtraction commutes with addition. Source: arithmetic
row centering used in arXiv:2301.05217v1 §5.1's diagnostic adaptation;
this is linearity in observed outputs, not in trained parameters. -/
theorem centerClasses_add (cs : Finset C) (z w : C → ℝ) (c : C) :
    centerClasses cs (fun k => z k + w k) c =
      centerClasses cs z c + centerClasses cs w c := by
  unfold centerClasses
  rw [meanOver_add]
  ring

/-- Subtraction also commutes with actual row centering. Source:
arXiv:2301.05217v1 §5.1's adapted finite projection, used here for the
appendix-inspired four-corner diagnostic. -/
theorem centerClasses_sub (cs : Finset C) (z w : C → ℝ) (c : C) :
    centerClasses cs (fun k => z k - w k) c =
      centerClasses cs z c - centerClasses cs w c := by
  unfold centerClasses
  rw [meanOver_sub]
  ring

/-- A common row offset is removed, assuming a nonempty class set.
Source: arXiv:2301.05217v1 §5.1's row-shift policy, applied before
interpreting the appendix-inspired interaction diagnostic. -/
theorem centerClasses_shift (cs : Finset C) (z : C → ℝ) (shift : ℝ)
    (hc : cs.Nonempty) (c : C) :
    centerClasses cs (fun k => z k + shift) c = centerClasses cs z c := by
  unfold centerClasses
  rw [row_mean_after_shift cs z shift hc]
  ring

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨false, by norm_num⟩

/-- Centering the actual contrast equals contrasting the four centered
corners. Source: the explicit refinement of arXiv:2301.05217v1's appendix
composition diagnostic; CE and labels do not enter this identity. -/
theorem center_commutes_with_contrast (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (c : C) :
    centeredInteraction cs f a b c =
      scoreInteraction (fun u v => centerClasses cs (f u v) c) a b := by
  unfold centeredInteraction scoreInteraction
  rw [centerClasses_add, centerClasses_sub, centerClasses_sub]

/-- Arbitrary common offsets at all four corners leave centered interaction
unchanged. Source: arXiv:2301.05217v1 appendix composition diagnostic,
with the softmax-invisible row offsets removed rather than counted. -/
theorem centeredInteraction_shift (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (shift : ℝ → ℝ → ℝ) (a b : ℝ) (hc : cs.Nonempty) (c : C) :
    centeredInteraction cs (fun u v k => f u v k + shift u v) a b c =
      centeredInteraction cs f a b c := by
  rw [center_commutes_with_contrast, center_commutes_with_contrast]
  unfold scoreInteraction
  simp only [centerClasses_shift cs _ _ hc]

example : ({0, 1} : Finset ℕ).Nonempty := by
  exact ⟨0, by norm_num⟩

/-- An input-independent class bias cancels from the contrast even though
it can alter task decisions. Source: arXiv:2301.05217v1 §5.1's policy of
retaining class bias when scoring references; the appendix-inspired
interaction energy must therefore be paired with actual decision metrics. -/
theorem centeredInteraction_class_bias (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (bias : C → ℝ) (a b : ℝ) (c : C) :
    centeredInteraction cs (fun u v k => f u v k + bias k) a b c =
      centeredInteraction cs f a b c := by
  have hf : (fun k => scoreInteraction (fun u v => f u v k + bias k) a b) =
      fun k => scoreInteraction (fun u v => f u v k) a b := by
    funext k
    unfold scoreInteraction
    ring
  unfold centeredInteraction
  rw [hf]

/-- The measured squared interaction is nonnegative by actual finite sums.
Source: the pre-loss diagnostic refinement of arXiv:2301.05217v1's
appendix; nonnegativity does not make this a correctness certificate. -/
theorem interactionEnergy_nonneg (cs : Finset C) (f : ℝ → ℝ → C → ℝ) (a b : ℝ) :
    0 ≤ interactionEnergy cs f a b := by
  unfold interactionEnergy energyOver
  exact Finset.sum_nonneg (fun c _ => sq_nonneg _)

/-- Energy zero means every measured centered coordinate is zero.
Source: the finite-output refinement of arXiv:2301.05217v1's appendix
composition hypothesis; the conclusion is only on the selected class set. -/
theorem interactionEnergy_zero_iff (cs : Finset C) (f : ℝ → ℝ → C → ℝ) (a b : ℝ) :
    interactionEnergy cs f a b = 0 ↔ ∀ c ∈ cs, centeredInteraction cs f a b c = 0 := by
  unfold interactionEnergy energyOver
  rw [Finset.sum_eq_zero_iff_of_nonneg (fun c _ => sq_nonneg _)]
  constructor
  · intro h c hc
    exact sq_eq_zero_iff.mp (h c hc)
  · intro h c hc
    rw [h c hc]
    norm_num

/-- On a nonempty measured class set, zero centered energy is exactly
a common raw contrast offset. Source: arXiv:2301.05217v1 appendix-inspired
diagnostic; zero energy does not identify a globally additive architecture. -/
theorem interactionEnergy_zero_iff_constant (cs : Finset C) (f : ℝ → ℝ → C → ℝ)
    (a b : ℝ) (hc : cs.Nonempty) :
    interactionEnergy cs f a b = 0 ↔
      ∃ shift : ℝ, ∀ c ∈ cs, scoreInteraction (fun u v => f u v c) a b = shift := by
  rw [interactionEnergy_zero_iff]
  constructor
  · intro h
    refine ⟨meanOver cs (fun k => scoreInteraction (fun u v => f u v k) a b), ?_⟩
    intro c hmem
    have hz := h c hmem
    unfold centeredInteraction centerClasses at hz
    linarith
  · rintro ⟨shift, h⟩
    have hm : meanOver cs (fun k => scoreInteraction (fun u v => f u v k) a b) = shift := by
      rw [← meanOver_const cs shift hc]
      unfold meanOver
      congr 1
      exact Finset.sum_congr rfl h
    intro c hmem
    unfold centeredInteraction centerClasses
    dsimp only
    rw [h c hmem, hm]
    ring

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨true, by norm_num⟩

end Transformer.Grokking.Composition.OutputInteractions
