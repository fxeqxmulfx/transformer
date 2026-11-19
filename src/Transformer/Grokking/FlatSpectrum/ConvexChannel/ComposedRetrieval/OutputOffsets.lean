module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.DepthCoordinate

/-!
# Bounded class offsets on the complete two-read score

Source: `review/convex/moments.md`, CM116. Every original class has a trainable
offset. The joint score is bounded. With other weights fixed, a labeled row
gives a cross-entropy derivative and strict classification constrains the
allowed displacement. Numerical score contrasts identify relative offsets.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Landscape.Neural

variable {O : Type*} {d n : ℕ}

/-- CM116. Apply the physical score scale before adding the output offset. -/
noncomputable def offsetScore (p : Parameters O d) (B : ℝ) (b : O → ℝ)
    (xs : List O) (c : O) : ℝ := B * score p xs c + b c

/-- The actual complete logit remains bounded on every prefix, CM116. -/
theorem offsetScore_bound (p : Parameters O d) (hp : Admissible p)
    (B D : ℝ) (hB : 0 ≤ B) (b : O → ℝ) (hb : ∀ c, |b c| ≤ D)
    (xs : List O) (c : O) : |offsetScore p B b xs c| ≤ B + D := by
  refine (abs_add_le _ _).trans (add_le_add ?_ (hb c))
  rw [abs_mul, abs_of_nonneg hB]
  simpa using mul_le_mul_of_nonneg_left (score_bound p hp xs c) hB

/-- The ordinary CE derivative in one output coordinate uses the complete
class partition and the original target, CM116. -/
theorem offset_derivative [Fintype O] [DecidableEq O] (p : Parameters O d)
    (B : ℝ) (b : O → ℝ) (xs : List O) (y k : O) :
    HasDerivAt (fun t => crossEntropy
      (fun c => offsetScore p B b xs c + if c = k then t else 0) y)
      (Geometry.probability (offsetScore p B b xs) k - if y = k then 1 else 0) 0 := by
  have h := GradientEvidence.readout_ce_line_hasDerivAt (offsetScore p B b xs)
    (fun c => if c = k then 1 else 0) y
  simpa only [mul_ite, mul_one, mul_zero, Fintype.sum_ite_eq'] using h

/-- At fixed base parameters a finite output-offset displacement satisfies a
supporting-plane bound on full CE, CM116. -/
theorem offset_tangent [Fintype O] (p : Parameters O d) (B : ℝ) (b v : O → ℝ)
    (xs : List O) (y : O) :
    crossEntropy (offsetScore p B b xs) y +
      ((∑ c, Geometry.probability (offsetScore p B b xs) c * v c) - v y) ≤
      crossEntropy (offsetScore p B (fun c => b c + v c) xs) y := by
  have h := Training.crossEntropy_first_order (offsetScore p B b xs)
    (offsetScore p B (fun c => b c + v c) xs) y
  have hd (c : O) :
      offsetScore p B (fun k => b k + v k) xs c - offsetScore p B b xs c = v c := by
    unfold offsetScore
    ring
  simpa only [hd] using h

/-- The same fixed-parameter lower bound sums over the observed training
occurrences and their original labels, CM116. -/
theorem summed_offset_tangent [Fintype O] (p : Parameters O d) (B : ℝ) (b v : O → ℝ)
    (xs : Fin n → List O) (y : Fin n → O) :
    (∑ i, crossEntropy (offsetScore p B b (xs i)) (y i)) +
      (∑ i, ((∑ c, Geometry.probability (offsetScore p B b (xs i)) c * v c) - v (y i))) ≤
    ∑ i, crossEntropy (offsetScore p B (fun c => b c + v c) (xs i)) (y i) := by
  simpa only [Finset.sum_add_distrib] using Finset.sum_le_sum (s := Finset.univ)
    (fun i _ => offset_tangent p B b v (xs i) (y i))

/-- At one prefix, relative logits identify the complete offset vector up to
a common additive constant when the other coordinates are fixed, CM116. -/
theorem offset_contrasts_iff (p : Parameters O d) (B : ℝ) (b u : O → ℝ)
    (xs : List O) (k : O) :
    (∀ c, offsetScore p B b xs c - offsetScore p B b xs k =
      offsetScore p B u xs c - offsetScore p B u xs k) ↔
    (∀ c, b c - u c = b k - u k) := by
  unfold offsetScore
  constructor <;> intro h c <;> linarith [h c]

/-- Exact finite-data conditions for strict winners when only the logit of
class `k` moves by `t`, CM116. Each index is an original supervised occurrence. -/
theorem strict_after_offset_iff [DecidableEq O] (z : Fin n → O → ℝ)
    (y : Fin n → O) (k : O) (t : ℝ) :
    (∀ i c, c ≠ y i →
      z i c + (if c = k then t else 0) < z i (y i) + (if y i = k then t else 0)) ↔
    (∀ i, y i = k → ∀ c, c ≠ k → z i c - z i k < t) ∧
    (∀ i, y i ≠ k → t < z i (y i) - z i k) ∧
    (∀ i, y i ≠ k → ∀ c, c ≠ y i → c ≠ k → z i c < z i (y i)) := by
  constructor
  · intro h
    refine ⟨?_, ?_, ?_⟩
    · intro i hy c hc
      have hw := h i c (by simpa [hy] using hc)
      simp only [hy, hc, ite_false, ite_true, add_zero] at hw
      linarith
    · intro i hy
      have hw := h i k (Ne.symm hy)
      simp only [hy, ite_false, ite_true, add_zero] at hw
      linarith
    · intro i hy c hcy hc
      simpa [hy, hc] using h i c hcy
  · rintro ⟨hp, hn, ho⟩ i c hcy
    by_cases hy : y i = k
    · have hc : c ≠ k := by simpa [hy] using hcy
      have hw := hp i hy c hc
      simp only [hy, hc, ite_false, ite_true, add_zero]
      linarith
    · by_cases hc : c = k
      · subst c
        have hw := hn i hy
        simp only [hy, ite_false, ite_true, add_zero]
        linarith
      · simpa [hy, hc] using ho i hy c hcy hc

/-- All initial errors can have the same true class while no unbounded
scalar shift of its output makes both training labels strict winners, CM116. -/
theorem concentrated_errors_insufficient :
    let z : Fin 2 → Fin 2 → ℝ := ![![0,2], ![0,1]]
    z 0 0 < z 0 1 ∧ z 1 0 < z 1 1 ∧ ∀ t : ℝ, ¬ ∀ i c, c ≠ i →
      z i c + (if c = 0 then t else 0) < z i i + (if i = 0 then t else 0) := by
  intro z
  refine ⟨by norm_num [z], by norm_num [z], ?_⟩
  intro t h
  obtain ⟨hp, hn, _⟩ := (strict_after_offset_iff z id 0 t).1 h
  have lower := hp 0 rfl 1 (by decide)
  have upper := hn 1 (by decide)
  norm_num [z] at lower upper
  linarith

example : |offsetScore (productParameters residue residue) 16 (fun _ => 3) [17,92] 12| ≤ 16+3 :=
  offsetScore_bound _ (product_admissible ..) 16 3 (by norm_num) _ (by intro c; norm_num) _ _

example : let z : Fin 2 → Fin 2 → ℝ := ![![0,1], ![0,3]]
    z 0 0 < z 0 1 ∧ ∀ i c, c ≠ i →
      z i c + (if c = 0 then 2 else 0) < z i i + (if i = 0 then 2 else 0) := by
  intro z
  constructor
  · norm_num [z]
  · apply (strict_after_offset_iff z id 0 2).2
    refine ⟨?_, ?_, ?_⟩
    · intro i hi c hc
      fin_cases i <;> fin_cases c <;> norm_num [z] at *
    · intro i hi
      fin_cases i <;> norm_num [z] at *
    · intro i hi c hcy hc
      fin_cases i <;> fin_cases c <;> norm_num [z] at *

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
