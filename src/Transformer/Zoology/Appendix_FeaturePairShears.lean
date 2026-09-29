/-
# Arbitrary feature-pair shears without extra coordinates

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`
and `lmm: kaleido-coyote`. A nonzero shear coefficient is implemented by
diagonal conjugation of a fixed unit shear. Splitting any coefficient into
two nonzero parts gives six layers, including coefficient zero.
-/

import Transformer.Zoology.Appendix_FeaturePairPrimitives

noncomputable section

namespace Transformer.Zoology

/-- Diagonal conjugation of a fixed unit shear.
Source: Appendix `prop: butterfly-hyena`, alternative in-place construction. -/
def featurePairConjugateShearNetwork {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) : CyclicKCoyoteNetwork n k 1 :=
  ((featurePairDiagonalNetwork t
      (if upper then fun i q => (a i q)⁻¹ else fun _ _ => 1)
      (if upper then fun _ _ => 1 else fun i q => (a i q)⁻¹)).append
    (featurePairUnitShearNetwork t upper)).append
      (featurePairDiagonalNetwork t
        (if upper then a else fun _ _ => 1)
        (if upper then fun _ _ => 1 else a))

/-- Diagonal conjugation realizes every nonzero independently parameterized
shear. Source: Appendix `lmm:primitives`, exact same-shape composition. -/
theorem featurePairConjugateShearNetwork_correct {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) (ha : ∀ i q, a i q ≠ 0)
    (u : RealSequence n (butterflyWidth k)) :
    (featurePairConjugateShearNetwork t upper a).run u =
      featurePairRun t (fun i q v => pairShear upper (a i q) v) u := by
  simp only [featurePairConjugateShearNetwork, CyclicKCoyoteNetwork.run_append,
    featurePairDiagonalNetwork_correct, featurePairUnitShearNetwork_correct,
    featurePairRun_comp]
  apply congrArg (fun F => featurePairRun t F u)
  funext i q v
  cases upper <;> apply Prod.ext <;>
    simp [pairShear, pairUpperShear, pairLowerShear] <;>
      field_simp [ha i q]

/-- First nonzero part of an arbitrary real shear coefficient.
Source: Appendix `lmm: kaleido-coyote`, construction includes zero coefficients. -/
def shearFirstPart (a : ℝ) : ℝ := if a = 1 then 2 else 1

/-- The first part never vanishes.
Source: Appendix `lmm: kaleido-coyote`, conjugation denominator condition. -/
theorem shearFirstPart_ne_zero (a : ℝ) : shearFirstPart a ≠ 0 := by
  by_cases h : a = 1 <;> simp [shearFirstPart, h]

/-- The remainder never vanishes either.
Source: Appendix `lmm: kaleido-coyote`, construction for arbitrary coefficients. -/
theorem shearSecondPart_ne_zero (a : ℝ) : a - shearFirstPart a ≠ 0 := by
  by_cases h : a = 1
  · norm_num [shearFirstPart, h]
  · simp [shearFirstPart, h, sub_eq_zero]

/-- Six-layer shear, valid for every real coefficient at every pair.
Source: Appendix `prop: butterfly-hyena`, no extra sequence workspace. -/
def featurePairShearNetwork {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) : CyclicKCoyoteNetwork n k 1 :=
  (featurePairConjugateShearNetwork t upper (fun i q => shearFirstPart (a i q))).append
    (featurePairConjugateShearNetwork t upper (fun i q => a i q - shearFirstPart (a i q)))

/-- The two nonzero shears add to the requested coefficient exactly.
Source: Appendix `lmm: kaleido-coyote`, arbitrary block coefficients. -/
theorem featurePairShearNetwork_correct {n k : ℕ} (t : Fin k) (upper : Bool)
    (a u : RealSequence n (butterflyWidth k)) :
    (featurePairShearNetwork t upper a).run u =
      featurePairRun t (fun i q v => pairShear upper (a i q) v) u := by
  rw [featurePairShearNetwork, CyclicKCoyoteNetwork.run_append,
    featurePairConjugateShearNetwork_correct t upper _ (fun i q => shearFirstPart_ne_zero _),
    featurePairConjugateShearNetwork_correct t upper _ (fun i q => shearSecondPart_ne_zero _),
    featurePairRun_comp]
  apply congrArg (fun F => featurePairRun t F u)
  funext i q v
  cases upper <;> apply Prod.ext <;>
    simp [pairShear, pairUpperShear, pairLowerShear] <;> ring

/-- Nonzero conjugation takes exactly three layers.
Source: Appendix `lmm:primitives`, diagonal/projection/diagonal composition. -/
theorem featurePairConjugateShearNetwork_layerCount {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) :
    (featurePairConjugateShearNetwork t upper a).layerCount = 3 := by
  simp [featurePairConjugateShearNetwork, CyclicKCoyoteNetwork.layerCount_append,
    featurePairDiagonalNetwork_layerCount, featurePairUnitShearNetwork,
    featurePairProjectionNetwork_layerCount]

/-- All coefficients, including zero, use exactly six layers.
Source: Appendix `prop: butterfly-hyena`, constant-depth factor operation. -/
theorem featurePairShearNetwork_layerCount {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) :
    (featurePairShearNetwork t upper a).layerCount = 6 := by
  rw [featurePairShearNetwork, CyclicKCoyoteNetwork.layerCount_append,
    featurePairConjugateShearNetwork_layerCount, featurePairConjugateShearNetwork_layerCount]

/-- The conjugation premise is satisfied by the constant unit coefficient.
Source: Appendix `def: butterfly`, nonempty shear construction. -/
example : ∀ (_i : Fin 2) (_q : Fin 2), (1 : ℝ) ≠ 0 := by norm_num

end Transformer.Zoology
