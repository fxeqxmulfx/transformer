module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Boundary
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-!
# Joint coordinates train both conditional stages at one fixed input

Source: `review/flat_spectrum.md`, CM74. Learn a strictly positive normalized
hidden-state/output table. Its row sums give a learned encoder; conditional
rows give a learned decoder. Their actual products recover the joint entries.
Answer-only CE on the resulting marginal is convex in all joint coordinates.
Neither stage is frozen, and no internal-path label is supplied to the loss.

This is exact for arbitrary positive two-stage distributions at one input.
It does not impose the earlier keep/flip tie. Nor does it prove that a decoder
can be shared across different inputs while retaining this convex domain.
Strict positivity excludes boundary optima; optimizer convergence is not claimed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Joint

open scoped BigOperators
open Transformer.Landscape.Neural (crossEntropy)

variable {H C : Type*} [Fintype H] [Fintype C]

/-- All complete path probabilities are optimized together, CM74. -/
def domain : Set (H → C → ℝ) :=
  {P | (∀ h c, 0 < P h c) ∧ ∑ h, ∑ c, P h c = 1}

/-- Learned hidden representation as the true joint row marginal, CM74. -/
noncomputable def encoder (P : H → C → ℝ) (h : H) : ℝ := ∑ c, P h c

/-- Learned output map as the true conditional row, CM74. It is coupled to
the encoder through the joint coordinates, not optimized as a separate factor. -/
noncomputable def decoder (P : H → C → ℝ) (h : H) (c : C) : ℝ :=
  P h c / encoder P h

/-- Actual two-stage inference retains every learned product and hidden state,
CM74. A marginal identity is proved below rather than substituted here. -/
noncomputable def forward (P : H → C → ℝ) (c : C) : ℝ :=
  ∑ h, encoder P h * decoder P h c

/-- Ordinary CE on normalized next-object log probabilities, CM74. -/
noncomputable def loss (c : C) (P : H → C → ℝ) : ℝ :=
  crossEntropy (fun d => Real.log (forward P d)) c

/-- The complete positive joint domain is convex and independent of labels,
CM74. All hidden and output coordinates participate in the same domain. -/
theorem domain_convex : Convex ℝ (domain : Set (H → C → ℝ)) := by
  intro P hP Q hQ a b ha hb hab
  refine ⟨?_, ?_⟩
  · intro h c
    change 0 < a * P h c + b * Q h c
    rcases lt_or_eq_of_le ha with ha | ha
    · exact add_pos_of_pos_of_nonneg (mul_pos ha (hP.1 h c)) (mul_nonneg hb (hQ.1 h c).le)
    · have hb' : 0 < b := by linarith
      exact add_pos_of_nonneg_of_pos (mul_nonneg ha.le (hP.1 h c).le) (mul_pos hb' (hQ.1 h c))
  · change (∑ h, ∑ c, (a * P h c + b * Q h c)) = 1
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
    rw [hP.2, hQ.2]
    linarith

/-- Every recovered encoder state is positive, CM74. -/
theorem encoder_pos [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain) (h : H) :
    0 < encoder P h :=
  Finset.sum_pos (fun c _ => hP.1 h c) Finset.univ_nonempty

/-- Expanding actual learned encoder/decoder products gives the output
marginal of the same joint table, CM74. -/
theorem forward_eq [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain) (c : C) :
    forward P c = ∑ h, P h c := by
  apply Finset.sum_congr rfl
  intro h _
  unfold decoder
  field_simp [(encoder_pos hP h).ne']

/-- Every answer has positive mass at an admitted point, CM74. -/
theorem forward_pos [Nonempty H] [Nonempty C] {P : H → C → ℝ}
    (hP : P ∈ domain) (c : C) : 0 < forward P c := by
  rw [forward_eq hP]
  exact Finset.sum_pos (fun h _ => hP.1 h c) Finset.univ_nonempty

/-- The actual two-stage output is normalized on the complete vocabulary,
CM74. No auxiliary target or missing-label assumption is used. -/
theorem forward_sum [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain) :
    ∑ c, forward P c = 1 := by
  simp_rw [forward_eq hP]
  rw [Finset.sum_comm]
  exact hP.2

/-- The computed full-vocabulary CE is exactly marginal NLL, CM74. -/
theorem loss_eq [Nonempty H] [Nonempty C] {P : H → C → ℝ}
    (hP : P ∈ domain) (c : C) : loss c P = -Real.log (forward P c) :=
  crossEntropy_log_probability _ (forward_pos hP) (forward_sum hP) c

/-- Ordinary answer-only CE is jointly convex in this whole two-stage
parameterization, CM74. The factors are recovered conditional distributions;
convexity in independent encoder/decoder factors is not asserted. -/
theorem loss_convex [Nonempty H] [Nonempty C] (c : C) :
    ConvexOn ℝ (domain : Set (H → C → ℝ)) (loss c) := by
  refine ⟨domain_convex, ?_⟩
  intro P hP Q hQ a b ha hb hab
  have hR := domain_convex hP hQ ha hb hab
  have hf : forward (a • P + b • Q) c = a * forward P c + b * forward Q c := by
    rw [forward_eq hR, forward_eq hP, forward_eq hQ]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib,
      ← Finset.mul_sum]
  rw [loss_eq hR, loss_eq hP, loss_eq hQ, hf]
  exact strictConcaveOn_log_Ioi.concaveOn.neg.2 (forward_pos hP c) (forward_pos hQ c) ha hb hab

end Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Joint
