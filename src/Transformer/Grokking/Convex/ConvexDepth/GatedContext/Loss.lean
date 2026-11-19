module

public import Transformer.Grokking.Convex.ConvexDepth.GatedContext.Bounds

/-!
# Ordinary CE in every local context coefficient

Source: `review/convex/moments.md`, CM46. Gates, anchor, radius, bank and full
head are fixed. CE remains convex in the complete collection of local rows.
The explicit slopes retain the radius, gate and every original raw class.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.GatedContext

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open FlatSpectrum

variable {G J K X C : Type*} [Fintype G] [Fintype J] [Fintype K] [Fintype C]

/-- Ordinary full-vocabulary CE is convex in all local simplex rows,
CM46. The statement does not train the gate or complete head jointly. -/
theorem ce_convex (gate : X → G → ℝ) (bank : J → K → X → ℝ) (base : J → ℝ)
    (radius : ℝ) (W : K → C → ℝ) (x : X) (y : C) :
    ConvexOn ℝ rows
      (fun a => crossEntropy (headLogits (features gate bank base radius a) W x) y) := by
  refine ⟨rows_convex, ?_⟩
  intro a _ b _ t u ht hu htu
  change crossEntropy (headLogits (features gate bank base radius (t • a + u • b)) W x) y ≤
    t * crossEntropy (headLogits (features gate bank base radius a) W x) y +
    u * crossEntropy (headLogits (features gate bank base radius b) W x) y
  rw [logits_affine gate bank base radius a b W x t u htu]
  exact (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ _) (Set.mem_univ _) ht hu htu

/-- The actual local coefficient slope is radius times fixed gate times
the complete softmax slope, CM46. No raw class is removed. -/
noncomputable def slope (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (a : G → J → ℝ)
    (x : X) (y : C) (g : G) (j : J) : ℝ :=
  radius * gate x g * ContextMixture.slope bank W (weights gate base radius a x) x y j

/-- Pairing local slopes with a coefficient change gives exactly the
effective-weight linear model, CM46; the fixed anchor cancels. -/
theorem slope_pairing (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (a b : G → J → ℝ)
    (x : X) (y : C) :
    (∑ g, ∑ j, (b g j - a g j) * slope gate bank base radius W a x y g j) =
      ∑ j, (weights gate base radius b x j - weights gate base radius a x j) *
        ContextMixture.slope bank W (weights gate base radius a x) x y j := by
  simp only [slope, weights_sub, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro g _
  ring

/-- The explicit local slopes support ordinary actual CE, CM46. This
inequality holds on all local coefficients, including every feasible row. -/
theorem first_order (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (a b : G → J → ℝ)
    (x : X) (y : C) :
    crossEntropy (headLogits (features gate bank base radius a) W x) y +
        (∑ g, ∑ j, (b g j - a g j) * slope gate bank base radius W a x y g j) ≤
      crossEntropy (headLogits (features gate bank base radius b) W x) y := by
  rw [slope_pairing]
  exact ContextMixture.first_order bank W
    (weights gate base radius a x) (weights gate base radius b x) x y

/-- XE2 yields an actual local quadratic CE model with a radius-scaled,
gated displacement bound from the same complete head budget, CM46. -/
theorem ce_quadratic (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) {radius : ℝ} (hzero : 0 ≤ radius) (a b : G → J → ℝ)
    (x : X) (y : C) (hgate : ∀ g, 0 ≤ gate x g) {M R : ℝ} (hM : 0 ≤ M)
    (hbank : ∀ j k, |bank j k x| ≤ M) {W : K → C → ℝ} (hW : W ∈ headBudget R) :
    crossEntropy (headLogits (features gate bank base radius b) W x) y ≤
      crossEntropy (headLogits (features gate bank base radius a) W x) y +
        (∑ g, ∑ j, (b g j - a g j) * slope gate bank base radius W a x y g j) +
        (radius * (R * M) * ∑ g, gate x g * ∑ j, |b g j - a g j|)^2 / 2 := by
  rw [slope_pairing, ContextMixture.slope_pairing]
  exact Training.crossEntropy_le_add _ _ y
    (logits_change_bound gate bank base hzero a b x hgate hM hbank hW)

end Transformer.Grokking.Convex.ConvexDepth.GatedContext
