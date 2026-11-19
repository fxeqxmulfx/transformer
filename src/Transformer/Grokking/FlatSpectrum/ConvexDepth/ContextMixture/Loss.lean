module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture.Basic

/-!
# Ordinary cross-entropy in the complete context mixture

Source: `review/convex/moments.md`, CM45. With the bank and head frozen, actual
logits are linear in all mixture coefficients. Ordinary CE is convex and its
explicit coefficient slopes satisfy the full first-order inequality. The
target in each comparison is the query's raw label, not a neighbor's label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open Geometry (probability)

variable {J K X C : Type*} [Fintype J] [Fintype K] [Fintype C]

/-- Actual ordinary CE is convex in the complete simplex mixture, CM45.
The feature bank and complete complex head are fixed during this optimization. -/
theorem ce_convex (bank : J → K → X → ℝ) (W : K → C → ℝ) (x : X) (y : C) :
    ConvexOn ℝ simplex (fun a => crossEntropy (headLogits (features bank a) W x) y) := by
  refine ⟨simplex_convex, ?_⟩
  intro a _ b _ t u ht hu htu
  change crossEntropy (headLogits (features bank (t • a + u • b)) W x) y ≤
    t * crossEntropy (headLogits (features bank a) W x) y +
    u * crossEntropy (headLogits (features bank b) W x) y
  rw [logits_linear]
  exact (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ _) (Set.mem_univ _) ht hu htu

/-- Jensen for the actual bank products, CM45. Every term on the right
uses the same query target; no local correctness is supplied by flatness. -/
theorem ce_le_mix (bank : J → K → X → ℝ) {a : J → ℝ} (ha : a ∈ simplex)
    (W : K → C → ℝ) (x : X) (y : C) :
    crossEntropy (headLogits (features bank a) W x) y ≤
      ∑ j, a j * crossEntropy (headLogits (bank j) W x) y := by
  rw [logits]
  simpa only [smul_eq_mul] using
    (ConvexRecall.convex_crossEntropy y).map_sum_le
      (fun j (_ : j ∈ (Finset.univ : Finset J)) => ha.1 j) ha.2
      (fun j (_ : j ∈ (Finset.univ : Finset J)) => Set.mem_univ (headLogits (bank j) W x))

/-- The ordinary softmax coefficient slope, CM45. It uses every raw class
and the actual bank logits, with the complete source head held fixed. -/
noncomputable def slope (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (a : J → ℝ) (x : X) (y : C) (j : J) : ℝ :=
  (∑ c, probability (headLogits (features bank a) W x) c * headLogits (bank j) W x c) -
    headLogits (bank j) W x y

omit [Fintype C] in
/-- A change of mixture coefficients has exactly the declared change in
actual logits, CM45. No approximation or new decoder appears here. -/
theorem logits_sub (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (a b : J → ℝ) (x : X) (c : C) :
    headLogits (features bank b) W x c - headLogits (features bank a) W x c =
      ∑ j, (b j - a j) * headLogits (bank j) W x c := by
  rw [logits, logits]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Pairing coefficient slopes with an increment gives exactly the ordinary
full-logit linear model, CM45. Every real/imaginary product remains present. -/
theorem slope_pairing (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (a b : J → ℝ) (x : X) (y : C) :
    (∑ j, (b j - a j) * slope bank W a x y j) =
      (∑ c, probability (headLogits (features bank a) W x) c *
        (headLogits (features bank b) W x c - headLogits (features bank a) W x c)) -
      (headLogits (features bank b) W x y - headLogits (features bank a) W x y) := by
  simp only [slope, mul_sub, Finset.sum_sub_distrib, logits_sub, Finset.mul_sum]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The explicit coefficient slopes give a supporting hyperplane to raw
CE, CM45. This applies to every pair of mixtures, including all simplex points. -/
theorem first_order (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (a b : J → ℝ) (x : X) (y : C) :
    crossEntropy (headLogits (features bank a) W x) y +
        ∑ j, (b j - a j) * slope bank W a x y j ≤
      crossEntropy (headLogits (features bank b) W x) y := by
  rw [slope_pairing]
  exact Training.crossEntropy_first_order _ _ y

omit [Fintype C] in
/-- The original feature and complete-head bounds control every logit
increment by the mixture's L1 displacement, CM45. -/
theorem logits_change_bound (bank : J → K → X → ℝ) (a b : J → ℝ) (x : X)
    {M R : ℝ} (hM : 0 ≤ M) (hbank : ∀ j k, |bank j k x| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features bank b) W x c - headLogits (features bank a) W x c| ≤
      R * M * ∑ j, |b j - a j| := by
  rw [logits_sub]
  calc
    _ ≤ ∑ j, |(b j - a j) * headLogits (bank j) W x c| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, |b j - a j| * (R * M) := by
      apply Finset.sum_le_sum
      intro j _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (headLogits_bound _ hW hM x (hbank j) c)
        (abs_nonneg _)
    _ = _ := by rw [← Finset.sum_mul]; ring

/-- XE2 gives a finite quadratic upper model for actual mixture CE,
CM45. Its radius is derived from the same charged complex-head bound. -/
theorem ce_quadratic (bank : J → K → X → ℝ) (a b : J → ℝ) (x : X) (y : C)
    {M R : ℝ} (hM : 0 ≤ M) (hbank : ∀ j k, |bank j k x| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) :
    crossEntropy (headLogits (features bank b) W x) y ≤
      crossEntropy (headLogits (features bank a) W x) y +
        (∑ j, (b j - a j) * slope bank W a x y j) +
        (R * M * ∑ j, |b j - a j|)^2 / 2 := by
  rw [slope_pairing]
  exact Training.crossEntropy_le_add _ _ y
    (logits_change_bound bank a b x hM hbank hW)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture
