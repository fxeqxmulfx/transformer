module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation.Basic

/-!
# Raw-label CE and local margin transfer

Source: `review/flat_spectrum.md`, CM42. Jensen bounds ordinary full-vocabulary
cross-entropy at an interpolated context. Accuracy requires local raw-label
information, expressed separately from the source spectrum and observed fit.
The conjugate initializer yields weighted raw-label votes in the actual head.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {K I X C : Type*} [Fintype K] [Fintype I] [Fintype C]

/-- Jensen for actual full-vocabulary logits, CM42. The comparison losses
use the query's raw target, not each neighbor's potentially different label. -/
theorem ce_le_weighted (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (y : C)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) :
    crossEntropy (headLogits (features w source) W x) y ≤
      ∑ i, w x i * crossEntropy (headLogits source W i) y := by
  rw [logits]
  simpa only [smul_eq_mul] using
    (ConvexRecall.convex_crossEntropy y).map_sum_le
      (fun i (_ : i ∈ (Finset.univ : Finset I)) => hw i) hs
      (fun i (_ : i ∈ (Finset.univ : Finset I)) => Set.mem_univ (headLogits source W i))

/-- Local low CE transfers to the query, CM42. This is an explicit premise
about the query's raw target at every positively weighted source state. -/
theorem ce_le_of_local (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (y : C)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) {ε : ℝ}
    (hlocal : ∀ i, w x i ≠ 0 → crossEntropy (headLogits source W i) y ≤ ε) :
    crossEntropy (headLogits (features w source) W x) y ≤ ε := by
  calc
    _ ≤ ∑ i, w x i * crossEntropy (headLogits source W i) y :=
      ce_le_weighted w source W x y hw hs
    _ ≤ ∑ i, w x i * ε := by
      apply Finset.sum_le_sum
      intro i _
      by_cases hi : w x i = 0
      · simp [hi]
      · exact mul_le_mul_of_nonneg_left (hlocal i hi) (hw i)
    _ = ε := by rw [← Finset.sum_mul, hs, one_mul]

omit [Fintype C] in
/-- Positive local raw-label margins survive fixed context interpolation,
CM42. Distinct aliases need not have the same margin, even if codes coincide. -/
theorem margin_le (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (y c : C)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) {m : ℝ}
    (hlocal : ∀ i, w x i ≠ 0 → m ≤ headLogits source W i y - headLogits source W i c) :
    m ≤ headLogits (features w source) W x y - headLogits (features w source) W x c := by
  rw [logits]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, ← Finset.sum_sub_distrib, ← mul_sub]
  calc
    m = ∑ i, w x i * m := by rw [← Finset.sum_mul, hs, one_mul]
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro i _
      by_cases hi : w x i = 0
      · simp [hi]
      · exact mul_le_mul_of_nonneg_left (hlocal i hi) (hw i)

omit [Fintype K] [Fintype C] in
/-- Raw-label weight in a fixed interpolation row, CM42. Labels come from
the original observed states; no unseen answer is used in this construction. -/
noncomputable def labelMass [DecidableEq C] (w : X → I → ℝ) (labels : I → C)
    (x : X) (c : C) : ℝ := ∑ i, if labels i = c then w x i else 0

omit [Fintype K] [Fintype C] in
/-- The actual conjugate Fourier head becomes a weighted raw-label vote,
CM42. This identity retains every real/imaginary coefficient of the source head. -/
theorem initializer_logits [DecidableEq I] [DecidableEq C]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (w : X → I → ℝ) (labels : I → C) (a : ℝ) (x : X) (c : C) :
    headLogits (features w (complexFeatures U))
      (complexCoefficients (complexInitializer U labels a)) x c =
      a * labelMass w labels x c := by
  rw [logits]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul,
    complexInitializer_logits U hU, labelMass, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : labels i = c
  · simp [hi, mul_comm]
  · simp [hi, Ne.symm hi]

omit [Fintype K] [Fintype C] in
/-- A strict local raw-label vote gives the actual prediction margin,
CM42. The vote comparison is additional information, not a spectral consequence. -/
theorem initializer_margin [DecidableEq I] [DecidableEq C]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (w : X → I → ℝ) (labels : I → C) {a : ℝ} (ha : 0 < a) (x : X) (y c : C)
    (hvote : labelMass w labels x c < labelMass w labels x y) :
    headLogits (features w (complexFeatures U))
        (complexCoefficients (complexInitializer U labels a)) x c <
      headLogits (features w (complexFeatures U))
        (complexCoefficients (complexInitializer U labels a)) x y := by
  rw [initializer_logits U hU, initializer_logits U hU]
  exact mul_lt_mul_of_pos_left hvote ha

-- Dirac context weights are inhabited. These examples retain four raw labels.
example := ce_le_weighted (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 id 4)) 0 0
  (pointWeights_nonneg id 0) (pointWeights_sum id 0)
example := ce_le_of_local (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 id 4)) 0 0
  (pointWeights_nonneg id 0) (pointWeights_sum id 0)
  (ε := Real.log (1 + 3 * Real.exp (-4))) (by
    intro i hi
    have h : i = 0 := by simpa [pointWeights] using hi
    subst i
    convert (complexInitializer_ce fourier4 fourier4_unitary id 4 0).le using 1 <;> norm_num)
example := margin_le (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 id 4)) 0 0 1
  (pointWeights_nonneg id 0) (pointWeights_sum id 0) (m := 4) (by
    intro i hi
    have h : i = 0 := by simpa [pointWeights] using hi
    subst i
    rw [complexInitializer_logits fourier4 fourier4_unitary]
    norm_num)
example := initializer_logits fourier4 fourier4_unitary
  (pointWeights (id : Fin 4 → Fin 4)) id 4 0 0
example := initializer_margin fourier4 fourier4_unitary
  (pointWeights (id : Fin 4 → Fin 4)) id (by norm_num : (0 : ℝ) < 4) 0 0 1
  (by norm_num [labelMass, pointWeights, Fin.sum_univ_four])

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation
