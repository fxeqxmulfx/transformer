import Transformer.GPTMini.Convex.Structured.PointerValues
import Transformer.GPTMini.Convex.Structured.SharedPointer
import Transformer.GPTMini.Convex.Structured.OutputMargins

/-!
# True compact pointer values and the whole-vocabulary decoder

Source: the exact pointer joint/mean contraction at 5371b5f and actual
ten-axis whole-vocabulary margins at d84bd84. The real decoder reads
the small-channel pointer means, rather than the observed route or a
desired output. Its scalar score is proved to be the actual normalized
full Gibbs expectation of the same output code used during supervision.

Every raw matching/value/route potential is unrestricted. Full score
bounds and the 11p-10 margin use the entire latent distribution, so
arbitrary wrong routes and channel combinations are retained. These
local decoder results do not supply a correct raw Basis route or
assume that the model already has correct logits. Semantic finite
weight constructions must derive the actual selected probability.
Residual/prenorm/tied model integration remains a separate proof.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

variable {J : Type*} [Fintype J] [Nonempty J]

/-- The actual ten learned pointer-output coordinates are true compact channel means.
Source: PointerValues.pointerMean and the five output-code coordinate pairs, without any observed-label argument. -/
def pointerOutputCoordinates (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (i : Fin 10) : ℝ :=
  if i.val % 2 = 0 then pointerMean query key value bias ⟨i.val / 2, by omega⟩ quarterX
  else pointerMean query key value bias ⟨i.val / 2, by omega⟩ quarterY

/-- True whole-token inference scores pair the actual learned means with the readonly candidate token code.
Source: the ten-axis decoder; target denotes a candidate vocabulary token, not a supplied correct answer. -/
def pointerOutputScore (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (target : Fin 1024) : ℝ :=
  ∑ h, (pointerMean query key value bias h quarterX * quarterX (outputDigit target h) +
    pointerMean query key value bias h quarterY * quarterY (outputDigit target h))

omit [Nonempty J] in
/-- Every actual even output coordinate is the corresponding compact learned first-axis value mean.
Source: genuine physical decoder indexing and bounded pair arithmetic. -/
theorem pointerOutputCoordinates_even (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (h : Fin 5) :
    pointerOutputCoordinates query key value bias ⟨2 * h.val, by omega⟩ = pointerMean query key value bias h quarterX := by
  have hm : (2 * h.val) % 2 = 0 := by omega
  have hd : (2 * h.val) / 2 = h.val := by omega
  simp only [pointerOutputCoordinates, hm, hd, ite_true]

omit [Nonempty J] in
/-- Every actual odd output coordinate is the same group's compact learned second-axis value mean.
Source: the actual paired ten-axis output layout, not an ungrounded scalar readout. -/
theorem pointerOutputCoordinates_odd (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (h : Fin 5) :
    pointerOutputCoordinates query key value bias ⟨2 * h.val + 1, by omega⟩ = pointerMean query key value bias h quarterY := by
  have hm : (2 * h.val + 1) % 2 = 1 := by omega
  have hd : (2 * h.val + 1) / 2 = h.val := by omega
  simp only [pointerOutputCoordinates, hm, hd, show ¬(1 : ℕ) = 0 by decide, ite_false]

omit [Nonempty J] in
/-- The actual compact decoder score is exactly the ten-coordinate output/code dot product.
Source: all physical even/odd decoder axes and the true small-channel learned means. -/
theorem pointerOutputScore_axes (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (target : Fin 1024) :
    pointerOutputScore query key value bias target =
      ∑ i, pointerOutputCoordinates query key value bias i * outputCoordinate (outputDigit target) i := by
  unfold pointerOutputScore
  rw [Fin.sum_univ_five]
  simp only [Fin.sum_univ_succ, pointerOutputCoordinates, outputCoordinate]
  norm_num
  ring

/-- The same computed compact score equals the entire actual joint Gibbs distribution's output-code expectation.
Source: exact true matching/value marginal contraction and finite expectation linearity; no configurations are dropped. -/
theorem pointerOutputScore_joint (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (target : Fin 1024) :
    pointerOutputScore query key value bias target = weightedOutputScore
      (pointerProbability query key value bias) (fun z : SharedPointerConfiguration J => z.2.2) target := by
  unfold weightedOutputScore outputCodeScore
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  unfold pointerOutputScore
  apply Finset.sum_congr rfl
  intro h hh
  simp only [outputPairScore, mul_add, Finset.sum_add_distrib, ← mul_assoc, ← Finset.sum_mul]
  have hx : (∑ z : SharedPointerConfiguration J, pointerProbability query key value bias z * quarterX (z.2.2 h)) =
      pointerMean query key value bias h quarterX := by
    convert pointerProbability_mean query key value bias h quarterX
  have hy : (∑ z : SharedPointerConfiguration J, pointerProbability query key value bias z * quarterY (z.2.2 h)) =
      pointerMean query key value bias h quarterY := by
    convert pointerProbability_mean query key value bias h quarterY
  rw [hx, hy]

/-- Every actual output coordinate stays within the true signed-axis value-code bounds at arbitrary finite raw weights.
Source: normalized positive memory/value means, with all four real signed-axis codes checked. -/
theorem pointerOutputCoordinates_bounds (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (i : Fin 10) :
    -1 ≤ pointerOutputCoordinates query key value bias i ∧ pointerOutputCoordinates query key value bias i ≤ 1 := by
  unfold pointerOutputCoordinates
  split_ifs
  · apply pointerMean_bounds
    intro d
    fin_cases d <;> norm_num [quarterX]
  · apply pointerMean_bounds
    intro d
    fin_cases d <;> norm_num [quarterY]

/-- Every actual whole-vocabulary score remains in [-5,5], including arbitrary incorrect latent assignments.
Source: the true full distribution, normalized positive mass and the proved five-pair code bounds. -/
theorem pointerOutputScore_bounds (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (target : Fin 1024) :
    -5 ≤ pointerOutputScore query key value bias target ∧ pointerOutputScore query key value bias target ≤ 5 := by
  rw [pointerOutputScore_joint]
  have hs : (∑ z : SharedPointerConfiguration J, pointerProbability query key value bias z) = 1 := by
    convert pointerProbability_sum query key value bias
  have hlo := Finset.sum_le_sum (fun (z : SharedPointerConfiguration J) (_ : z ∈ Finset.univ) => mul_le_mul_of_nonneg_left
    (outputCodeScore_bounds (z.2.2) (outputDigit target)).1 (pointerProbability_pos query key value bias z).le)
  have hup := Finset.sum_le_sum (fun (z : SharedPointerConfiguration J) (_ : z ∈ Finset.univ) => mul_le_mul_of_nonneg_left
    (outputCodeScore_bounds (z.2.2) (outputDigit target)).2 (pointerProbability_pos query key value bias z).le)
  change -5 ≤ weightedOutputScore _ _ _ ∧ weightedOutputScore _ _ _ ≤ 5
  rw [← Finset.sum_mul, hs, one_mul] at hlo hup
  exact ⟨hlo, hup⟩

/-- True selected mass gives the actual compact learned pointer's full-vocabulary quantitative margin.
Source: full inference/expectation identity and the whole-distribution 11p-10 decoder theorem, without a correct-score premise. -/
theorem pointerOutput_margin (query : Fin 4 → Fin 4 → ℝ) (key : J → Fin 4 → Fin 4 → ℝ)
    (value : J → Fin 5 → Fin 4 → ℝ) (bias : J → ℝ) (selected : SharedPointerConfiguration J)
    (target rival : Fin 1024) (hlabel : selected.2.2 = outputDigit target) (hne : target ≠ rival) :
    11 * pointerProbability query key value bias selected - 10 ≤
      pointerOutputScore query key value bias target - pointerOutputScore query key value bias rival := by
  rw [pointerOutputScore_joint, pointerOutputScore_joint]
  have hn : ∀ z : SharedPointerConfiguration J, 0 ≤ pointerProbability query key value bias z :=
    fun z => (pointerProbability_pos query key value bias z).le
  have hs : (∑ z : SharedPointerConfiguration J, pointerProbability query key value bias z) = 1 := by
    convert pointerProbability_sum query key value bias
  exact weightedOutput_margin _ _ selected target rival hn hs hlabel hne

example : ((0 : Fin 2), ((fun _ : Fin 4 => (0 : Fin 4)), outputDigit (25 : Fin 1024))).2.2 = outputDigit 25 ∧
    (25 : Fin 1024) ≠ 17 := by decide

/-- An actual untrained compact pointer has a neutral whole-token score through its real value means.
Source: all-zero free matching/value/route logits and the four balanced signed-axis output channels. -/
example : pointerOutputScore (fun _ _ => (0 : ℝ)) (fun _ : Fin 2 => fun _ _ => 0)
    (fun _ _ _ => 0) (fun _ => 0) 25 = 0 := by
  norm_num [pointerOutputScore, pointerMean, pointerWeight, pointerPartition, channelPartition,
    channelMean, channelWeight, quarterX, quarterY, Fin.sum_univ_two, Fin.sum_univ_four,
    Fin.sum_univ_five, Fin.prod_univ_four, Fin.prod_univ_five]

end
end Transformer.GPTMini.Convex.Structured
