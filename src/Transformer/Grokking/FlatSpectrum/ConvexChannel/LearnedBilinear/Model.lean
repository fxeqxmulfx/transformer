module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss

/-!
# Ordinary next-object cross-entropy through both recovered encoders

Source: `review/convex/moments.md`, CM86. One full rectangular complex matrix
per output object is jointly learned. Actual forward computation recovers both
representations and the readout. Its identity with the joint reading proves
ordinary full-vocabulary CE convexity on the entire coordinate space. There
are no observed-path targets, activation cones, or latent-label constraints.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear

open scoped BigOperators Matrix ComplexOrder Matrix.Norms.Elementwise
open Landscape.Neural (crossEntropy)

variable {Q D C : Type*} [Fintype Q] [Fintype D] [DecidableEq Q] [DecidableEq D]

/-- Every original output object has its complete rectangular joint state,
CM86. There is no independent whole-prefix table. -/
abbrev Parameters (Q D C : Type*) := C → Matrix Q D ℂ

/-- Reconstructing both representations after an arbitrary real parameter
combination gives the same complete complex combination, CM86. -/
theorem forward_combination (M N : Matrix Q D ℂ) (x : Q → ℂ) (y : D → ℂ) (a b : ℝ) :
    forward (a • M + b • N) x y = a • forward M x y + b • forward N x y := by
  simp only [forward_eq, Matrix.add_mulVec, Matrix.smul_mulVec,
    dotProduct_add, dotProduct_smul, Complex.real_smul]
  ring

/-- The real part is taken after the actual complex interaction and
learned readout, CM86. Every output object remains represented. -/
noncomputable def logits (p : Parameters Q D C) (x : Q → ℂ) (y : D → ℂ) : C → ℝ :=
  fun c => (forward (p c) x y).re

/-- The actual recovered logits retain the complete joint complex reading,
CM86. This identity is not substituted into inference's definition. -/
theorem logits_eq (p : Parameters Q D C) (x : Q → ℂ) (y : D → ℂ) (c : C) :
    logits p x y c = ((1 + Complex.I) * (star x ⬝ᵥ ((p c) *ᵥ y))).re := by
  unfold logits
  rw [forward_eq]

/-- All learned coordinates preserve real combinations at the output, CM86. -/
theorem logits_combination (p q : Parameters Q D C) (x : Q → ℂ) (y : D → ℂ) (a b : ℝ) :
    logits (a • p + b • q) x y = a • logits p x y + b • logits q x y := by
  funext c
  simp only [logits, Pi.add_apply, Pi.smul_apply, forward_combination]
  simp

variable [Fintype C]

/-- Only the ordinary answer loss receives the next original object, CM86. -/
noncomputable def loss (x : Q → ℂ) (y : D → ℂ) (target : C) (p : Parameters Q D C) : ℝ :=
  crossEntropy (logits p x y) target

/-- Ordinary CE is jointly convex in the entire rectangular model, CM86.
Convexity in independently optimized physical factors is not asserted. -/
theorem loss_convex (x : Q → ℂ) (y : D → ℂ) (target : C) :
    ConvexOn ℝ Set.univ (loss x y target : Parameters Q D C → ℝ) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [loss, logits_combination]
  exact (ConvexRecall.convex_crossEntropy target).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The actual output probabilities are positive, CM86. -/
theorem probability_pos (p : Parameters Q D C) (x : Q → ℂ) (y : D → ℂ) (c : C) :
    0 < Geometry.probability (logits p x y) c := Geometry.probability_pos _ _

/-- Ordinary softmax normalizes every retained original output class, CM86. -/
theorem probability_sum (p : Parameters Q D C) (x : Q → ℂ) (y : D → ℂ) (witness : C) :
    ∑ c, Geometry.probability (logits p x y) c = 1 := Geometry.sum_probability _ witness

/-- Full indexed mean minibatch, including repeated records and short tails,
CM86. The zero-record expression is not a prescribed update. -/
noncomputable def batchLoss {N : ℕ} (query : Fin N → Q → ℂ) (context : Fin N → D → ℂ)
    (target : Fin N → C) (p : Parameters Q D C) : ℝ :=
  (N : ℝ)⁻¹ * ∑ i, loss (query i) (context i) (target i) p

/-- Every indexed mean minibatch retains joint convexity, CM86. No optimizer
convergence or finite-difference derivative identity is assumed here. -/
theorem batchLoss_convex {N : ℕ} (query : Fin N → Q → ℂ) (context : Fin N → D → ℂ)
    (target : Fin N → C) :
    ConvexOn ℝ Set.univ (batchLoss query context target : Parameters Q D C → ℝ) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq a b ha hb hab
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) =>
    (loss_convex (query i) (context i) (target i)).2 hp hq ha hb hab)
  have hs := mul_le_mul_of_nonneg_left h (inv_nonneg.mpr (Nat.cast_nonneg N : (0 : ℝ) ≤ N))
  simpa only [batchLoss, smul_eq_mul, Finset.sum_add_distrib, ← Finset.mul_sum,
    mul_add, mul_left_comm] using hs

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear
