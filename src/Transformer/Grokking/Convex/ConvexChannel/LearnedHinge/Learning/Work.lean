module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.RawGradient

/-!
# Combining complete coordinate directions without dropping records

Source: `review/convex/moments.md`, CM82. The full raw-coordinate pairing is a
linear map of the complete gradient array. This supplies exact sums and scaling
for shared passes and indexed minibatches, including repeated records.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators

/-- Every native gradient array index is retained, CM82. -/
abbrev GradientArray (J O : Type*) (T : ℕ) := J → O → Bool → Bool → Coordinate O T → ℝ

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- Coordinate combinations preserve their complete work, CM82. -/
theorem rawWork_combination (g h : GradientArray J O T)
    (v : Parameters J O (ObjectWeights O T)) (a b : ℝ) :
    rawWork (a • g + b • h) v = a * rawWork g v + b * rawWork h v := by
  simp only [rawWork, Projection.dot, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    add_mul, Finset.sum_add_distrib, mul_assoc, ← Finset.mul_sum]

/-- Complete work as a linear map of update arrays, CM82. The direction
argument enters every original coordinate's inner product. -/
def rawWorkLinear (v : Parameters J O (ObjectWeights O T)) : GradientArray J O T →ₗ[ℝ] ℝ where
  toFun g := rawWork g v
  map_add' g h := by
    simpa only [one_smul, one_mul] using rawWork_combination g h v 1 1
  map_smul' a g := by
    change rawWork (a • g) v = a * rawWork g v
    simpa only [zero_smul, add_zero, zero_mul, smul_eq_mul] using rawWork_combination g 0 v a 0

/-- Every indexed record contributes to the full coordinate work, CM82. -/
theorem rawWork_sum {I : Type*} [Fintype I] (g : I → GradientArray J O T)
    (v : Parameters J O (ObjectWeights O T)) :
    rawWork (∑ i, g i) v = ∑ i, rawWork (g i) v :=
  map_sum (rawWorkLinear v) g Finset.univ

/-- Normalization scales the complete work once, CM82. -/
theorem rawWork_smul (a : ℝ) (g : GradientArray J O T)
    (v : Parameters J O (ObjectWeights O T)) : rawWork (a • g) v = a * rawWork g v :=
  map_smul (rawWorkLinear v) a g

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning
