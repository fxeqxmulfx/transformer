module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms.Head

/-!
# Exact complete products before input normalization

Source: review/flat_spectrum.md, CM63. These are real-arithmetic identities
for every saved coefficient, including coefficients perturbed by numerical
construction. A native implementation still has to certify its arithmetic.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ExactScoring

open scoped BigOperators

variable {I K C X : Type*} [Fintype I] [Fintype K]

/-- The full real tensor contracted before the common division, CM63.
No compact decoder or purported target answer substitutes for a coefficient. -/
noncomputable def numerator (w : I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (c : C) : ℝ :=
  ∑ k, W k c * ∑ i, w i * source k i

/-- Reordering both finite sums gives the same complete source product,
CM63. Signed input rows and arbitrary coefficient tensors are admitted. -/
theorem numerator_source (w : I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (c : C) :
    numerator w source W c = ∑ i, w i * headLogits source W i c := by
  unfold numerator headLogits
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Delay the single positive input normalizer until after all products,
CM63. This equality uses the actual complete head, not a rounded parameter map. -/
theorem normalized_head (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c : C) :
    headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x c =
      numerator (w x) source W c / SignedContext.normalizer (w x) := by
  rw [SignedContext.normalized_logits, numerator_source]

/-- Exact class ordering is determined before division by the common
positive normalizer, CM63. Flatness provides no extra answer information. -/
theorem normalized_lt_iff (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (a b : C) :
    headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x a <
        headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x b ↔
      numerator (w x) source W a < numerator (w x) source W b := by
  rw [normalized_head, normalized_head]
  exact div_lt_div_iff_of_pos_right (SignedContext.normalizer_pos (w x))

/-- An exact tie is also invariant under the common positive normalization,
CM63. A floating tolerance is not an equality hypothesis. -/
theorem normalized_eq_iff (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (a b : C) :
    headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x a =
        headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x b ↔
      numerator (w x) source W a = numerator (w x) source W b := by
  rw [normalized_head, normalized_head]
  exact div_left_inj' (ne_of_gt (SignedContext.normalizer_pos (w x)))

/-- Realification for a rectangular complex source bank, CM63. The sign of
the imaginary feature matches the actual complex multiplication. -/
def realSource (U : I → K → ℂ) (k : K × Bool) (i : I) : ℝ :=
  if k.2 then -(U i k.1).im else (U i k.1).re

/-- Independently multiplying the entire complex tensor produces the same
unnormalized numerator as the real/imaginary contraction, CM63. No unitarity,
decoder identity or exact coefficient-construction assumption is needed. -/
theorem complex_numerator (w : I → ℝ) (U : I → K → ℂ)
    (H : K → C → ℂ) (c : C) :
    numerator w (realSource U) (complexCoefficients H) c =
      (∑ k, (∑ i, (w i : ℂ) * U i k) * H k c).re := by
  simp only [numerator, Fintype.sum_prod_type, Fintype.sum_bool,
    realSource, complexCoefficients, Bool.false_eq_true, ite_false, ite_true,
    Complex.re_sum, Complex.im_sum, Complex.mul_re, Complex.mul_im,
    Complex.ofReal_re, Complex.ofReal_im, zero_mul, add_zero, sub_zero]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [mul_neg, Finset.sum_neg_distrib]
  ring

/-- Subtract a single class's exact numerator before converting the vector,
CM63. This is only a common logit shift after the same positive division. -/
theorem gauge_logits (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (r c : C) :
    (numerator (w x) source W c - numerator (w x) source W r) /
        SignedContext.normalizer (w x) =
      headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x c -
        headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x r := by
  rw [normalized_head, normalized_head, sub_div]

/-- Ordinary full-vocabulary CE is unchanged by that common shift, CM63.
This is an identity of the original objective, not a new surrogate loss. -/
theorem gauge_crossEntropy [Fintype C] (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (r y : C) :
    Landscape.Neural.crossEntropy
      (fun c => (numerator (w x) source W c - numerator (w x) source W r) /
        SignedContext.normalizer (w x)) y =
      Landscape.Neural.crossEntropy
        (headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x) y := by
  simp_rw [gauge_logits]
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  congr 1
  apply Finset.sum_congr rfl
  intro c hc
  congr 1
  ring

/-- Valid numerator enclosures certify a unique exact winner only when the
intervals separate, CM63. Constructing correct native enclosures remains an
implementation obligation; agreement of two rounded argmaxes is not a premise. -/
theorem enclosure_winner (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (lower upper : C → ℝ) (y : C)
    (hl : lower y ≤ numerator (w x) source W y)
    (hu : ∀ c, numerator (w x) source W c ≤ upper c)
    (hgap : ∀ c, c ≠ y → upper c < lower y) :
    ∀ c, c ≠ y →
      headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x c <
        headLogits (ContextInterpolation.features (fun x => SignedContext.normalized (w x)) source) W x y := by
  intro c hc
  apply (normalized_lt_iff w source W x c y).2
  exact lt_of_le_of_lt (hu c) (lt_of_lt_of_le (hgap c hc) hl)

/-- Two actual coefficient columns and a nonconstant input row satisfy the
interval certificate; its domain is not empty, CM63. -/
example :
    ∀ c : Fin 2, c ≠ 1 →
      headLogits (ContextInterpolation.features
        (fun _ : Unit => SignedContext.normalized (![1, 2] : Fin 2 → ℝ))
        (fun k i : Fin 2 => if k = i then (1 : ℝ) else 0))
        (fun k c : Fin 2 => (![![2, 1], ![-1, 1]] : Fin 2 → Fin 2 → ℝ) k c) () c <
      headLogits (ContextInterpolation.features
        (fun _ : Unit => SignedContext.normalized (![1, 2] : Fin 2 → ℝ))
        (fun k i : Fin 2 => if k = i then (1 : ℝ) else 0))
        (fun k c : Fin 2 => (![![2, 1], ![-1, 1]] : Fin 2 → Fin 2 → ℝ) k c) () 1 := by
  apply enclosure_winner _ _ _ _ (![-1, 2] : Fin 2 → ℝ) (![1, 4] : Fin 2 → ℝ) 1
  · norm_num [numerator, Fin.sum_univ_succ]
  · intro c
    fin_cases c <;> norm_num [numerator, Fin.sum_univ_succ]
  · intro c hc
    fin_cases c <;> norm_num at *

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ExactScoring
