/-
# The exchange rate between steps and examples

arXiv:1812.06162, Appendix D, "Theory".  A full-batch step drops the loss by `ΔL_max`, a step
of batch `B` by the best `ΔL_opt(B) = ΔL_max/(1 + B_noise/B)` of eq. (2.7), so `δS` steps of
batch `B` make the progress of one full-batch step if and only if `δS = 1 + 𝓑/B`
(`mul_lossDrop_eq_iff`), at `δE = BδS` examples: the integrands of eq. (D.1), `totalSteps`
and `totalExamples` of `Section2_Tradeoff`.  The paper's full-batch step, "over which the loss
increases by an amount `δL`", is read as one over which it drops.  The exchange rate
`r = -(dδE/dB)/(dδS/dB)` is `B²/𝓑`, eq. (D.2) (`exchangeRate`).

For a noise scale `𝓑 ≥ 0` and `r > 0`, the schedule `B(s) = √(r𝓑(s))` of eq. (D.3)
(`adaptiveBatch`) takes `S_min + ∫√𝓑 ds/√r` steps and `E_min + √r ∫√𝓑 ds` examples
(`totalSteps_adaptiveBatch`, `totalExamples_adaptiveBatch`), and every positive schedule `B`
has `S(B) + E(B)/r = S(√(r𝓑)) + E(√(r𝓑))/r + ∫(B - √(r𝓑))²/(rB) ds`
(`totalSteps_add_div`): no schedule takes fewer steps on as few examples, or fewer examples in
as few steps (`adaptiveBatch_optimal`).  The converse, the paper's argument that an optimal
schedule has a constant exchange rate, is `exchangeRate_const` of `SectionD_ParetoFront`.
-/

import Transformer.NoiseScale.Section2_Tradeoff
import Transformer.NoiseScale.Section2_NoiseScale
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.Calculus.Deriv.Inv

open MeasureTheory
open scoped Matrix

namespace Transformer.NoiseScale

/-- **Appendix D**: `δS` steps of batch `B`, each dropping the loss by the best
`ΔL_opt(B) = ΔL_max/(1 + B_noise/B)` of eq. (2.7), drop it by the `ΔL_max` of one full-batch
step if and only if `δS = 1 + B_noise/B`. -/
theorem mul_lossDrop_eq_iff {ι : Type*} [Fintype ι] {G : ι → ℝ} {H S : Matrix ι ι ℝ}
    (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace) {B : ℝ} (hB : 0 < B) (δS : ℝ) :
    δS * (lossDropMax G H / (1 + noiseScale G H S / B)) = lossDropMax G H ↔
      δS = 1 + noiseScale G H S / B := by
  have hG : 0 < G ⬝ᵥ G := lt_of_le_of_ne (Finset.sum_nonneg fun a _ => mul_self_nonneg (G a))
    fun h => by simp [dotProduct_self_eq_zero.1 h.symm] at hH
  have hL : 0 < lossDropMax G H := by unfold lossDropMax; positivity
  have hn : 0 < 1 + noiseScale G H S / B := by unfold noiseScale; positivity
  rw [← mul_div_assoc, div_eq_iff hn.ne', mul_comm (lossDropMax G H), mul_left_inj' hL.ne']

/-- The hypotheses of `mul_lossDrop_eq_iff` are satisfiable: `G = H = Σ = 1`, `B = 1`. -/
example := mul_lossDrop_eq_iff (G := fun _ : Unit => 1) (H := 1) (S := 1) (by simp) (by simp)
  one_pos 2

/-- **Eq. (D.2)**: the exchange rate `r = -(dδE/dB)/(dδS/dB)` between the examples
`δE = BδS` and the steps `δS = 1 + 𝓑/B` of a full-batch step's progress is `B²/𝓑`. -/
theorem exchangeRate {b B : ℝ} (hb : b ≠ 0) (hB : B ≠ 0) :
    -deriv (fun B => B * (1 + b / B)) B / deriv (fun B => 1 + b / B) B = B ^ 2 / b := by
  have h1 : HasDerivAt (fun B => 1 + b / B) (-(b / B ^ 2)) B := by
    convert HasDerivAt.const_add (1 : ℝ) ((hasDerivAt_inv hB).const_mul b) using 1
    · simp only [div_eq_mul_inv]
    · field_simp
  rw [((hasDerivAt_id' (x := B)).fun_mul h1).deriv, h1.deriv]
  field_simp
  ring

/-- The hypotheses of `exchangeRate` are satisfiable: `𝓑 = B = 1`. -/
example := exchangeRate one_ne_zero one_ne_zero

/-- `(B - √(r𝓑))²/(rB) = 𝓑/B + B/r - 2√(r𝓑)/r`. -/
theorem sq_sub_sqrt_div {b r B : ℝ} (hb : 0 ≤ b) (hr : 0 < r) (hB : 0 < B) :
    (B - √(r * b)) ^ 2 / (r * B) = b / B + B / r - 2 * √(r * b) / r := by
  have hq := Real.sq_sqrt (mul_nonneg hr.le hb)
  field_simp
  linear_combination hq

/-- The hypotheses of `sq_sub_sqrt_div` are satisfiable: `𝓑 = r = B = 1`. -/
example := sq_sub_sqrt_div zero_le_one one_pos one_pos

variable {T : Type*} [MeasurableSpace T] {μ : Measure T} [IsFiniteMeasure μ] {𝓑 : T → ℝ}

/-- The batch schedule `B(s) = √(r𝓑(s))` of eq. (D.3), of exchange rate `B²/𝓑 = r`. -/
noncomputable def adaptiveBatch (r : ℝ) (𝓑 : T → ℝ) (s : T) : ℝ := √(r * 𝓑 s)

omit [MeasurableSpace T] [IsFiniteMeasure μ] in
theorem adaptiveBatch_eq {r : ℝ} (hr : 0 ≤ r) : adaptiveBatch r 𝓑 = fun s => √r * √(𝓑 s) :=
  funext fun _ => Real.sqrt_mul hr _

/-- The hypothesis of `adaptiveBatch_eq` is satisfiable: `r = 1`. -/
example : adaptiveBatch 1 (fun _ : Unit => 1) = fun _ => √1 * √1 := adaptiveBatch_eq zero_le_one

theorem integrable_sqrt (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) :
    Integrable (fun s => √(𝓑 s)) μ := by
  have e : (fun s => √(𝓑 s) ^ 2) = 𝓑 := funext fun s => Real.sq_sqrt (h𝓑 s)
  refine ((memLp_two_iff_integrable_sq
    (Real.continuous_sqrt.comp_aestronglyMeasurable hi.1)).2 ?_).integrable one_le_two
  rw [e]
  exact hi

/-- The hypotheses of `integrable_sqrt` are satisfiable: `𝓑 = 0`. -/
example : Integrable (fun s => √((0 : T → ℝ) s)) μ :=
  integrable_sqrt (fun _ => le_rfl) (integrable_zero _ _ _)

omit [MeasurableSpace T] [IsFiniteMeasure μ] in
theorem div_adaptiveBatch (h𝓑 : ∀ s, 0 ≤ 𝓑 s) {r : ℝ} (hr : 0 < r) :
    (fun s => 𝓑 s / adaptiveBatch r 𝓑 s) = fun s => √(𝓑 s) / √r := by
  funext s
  rw [adaptiveBatch, Real.sqrt_mul hr.le]
  rcases (h𝓑 s).eq_or_lt with h | h
  · simp [← h]
  · have := Real.sqrt_pos.2 h
    have := Real.sqrt_pos.2 hr
    field_simp
    rw [Real.sq_sqrt (h𝓑 s)]

/-- The schedule `√(r𝓑)` takes `S = S_min + ∫√𝓑 ds/√r` steps. -/
theorem totalSteps_adaptiveBatch (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) {r : ℝ}
    (hr : 0 < r) :
    totalSteps μ 𝓑 (adaptiveBatch r 𝓑) = μ.real Set.univ + (∫ s, √(𝓑 s) ∂μ) / √r := by
  rw [totalSteps_eq (by rw [div_adaptiveBatch h𝓑 hr]; exact (integrable_sqrt h𝓑 hi).div_const _),
    div_adaptiveBatch h𝓑 hr, integral_div]

/-- The schedule `√(r𝓑)` processes `E = E_min + √r ∫√𝓑 ds` examples. -/
theorem totalExamples_adaptiveBatch (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) {r : ℝ}
    (hr : 0 ≤ r) :
    totalExamples μ 𝓑 (adaptiveBatch r 𝓑) = (∫ s, 𝓑 s ∂μ) + √r * ∫ s, √(𝓑 s) ∂μ := by
  rw [adaptiveBatch_eq hr, totalExamples_eq hi ((integrable_sqrt h𝓑 hi).const_mul _),
    integral_const_mul]

/-- The hypotheses of `div_adaptiveBatch`, `totalSteps_adaptiveBatch` and
`totalExamples_adaptiveBatch` are satisfiable: `𝓑 = 0`, `r = 1`. -/
example := And.intro (div_adaptiveBatch (𝓑 := (0 : T → ℝ)) (fun _ => le_rfl) one_pos)
  <| And.intro (totalSteps_adaptiveBatch (μ := μ) (𝓑 := 0) (fun _ => le_rfl)
  (integrable_zero _ _ _) one_pos) (totalExamples_adaptiveBatch (μ := μ) (𝓑 := 0)
  (fun _ => le_rfl) (integrable_zero _ _ _) zero_le_one)

/-- Every positive schedule `B` has
`S(B) + E(B)/r = S(√(r𝓑)) + E(√(r𝓑))/r + ∫(B - √(r𝓑))²/(rB) ds`. -/
theorem totalSteps_add_div (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) {r : ℝ} (hr : 0 < r)
    {Bs : T → ℝ} (hB : ∀ s, 0 < Bs s) (hBi : Integrable Bs μ)
    (hdi : Integrable (fun s => 𝓑 s / Bs s) μ) :
    totalSteps μ 𝓑 Bs + totalExamples μ 𝓑 Bs / r =
      totalSteps μ 𝓑 (adaptiveBatch r 𝓑) + totalExamples μ 𝓑 (adaptiveBatch r 𝓑) / r +
        ∫ s, (Bs s - adaptiveBatch r 𝓑 s) ^ 2 / (r * Bs s) ∂μ := by
  have hq : Integrable (fun s => √r * √(𝓑 s)) μ := (integrable_sqrt h𝓑 hi).const_mul _
  have e : (fun s => (Bs s - adaptiveBatch r 𝓑 s) ^ 2 / (r * Bs s)) =
      fun s => 𝓑 s / Bs s + Bs s / r - 2 * (√r * √(𝓑 s)) / r := funext fun s => by
    rw [adaptiveBatch, sq_sub_sqrt_div (h𝓑 s) hr (hB s), Real.sqrt_mul hr.le]
  have h1 : Integrable (fun s => 𝓑 s / Bs s + Bs s / r) μ := hdi.add (hBi.div_const r)
  rw [e, integral_sub h1 ((hq.const_mul 2).div_const r),
    integral_add hdi (hBi.div_const r), integral_div, integral_div, integral_const_mul,
    integral_const_mul, totalSteps_adaptiveBatch h𝓑 hi hr,
    totalExamples_adaptiveBatch h𝓑 hi hr.le, totalSteps_eq hdi, totalExamples_eq hi hBi]
  obtain ⟨t, ht, rfl⟩ : ∃ t, 0 < t ∧ r = t ^ 2 :=
    ⟨√r, Real.sqrt_pos.2 hr, (Real.sq_sqrt hr.le).symm⟩
  rw [Real.sqrt_sq ht.le]
  field_simp
  ring

/-- **Eq. (D.3)**: the schedule `√(r𝓑)` is optimal: a positive schedule on no more examples
takes no fewer steps, and one in no more steps processes no fewer examples. -/
theorem adaptiveBatch_optimal (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) {r : ℝ} (hr : 0 < r)
    {Bs : T → ℝ} (hB : ∀ s, 0 < Bs s) (hBi : Integrable Bs μ)
    (hdi : Integrable (fun s => 𝓑 s / Bs s) μ) :
    (totalExamples μ 𝓑 Bs ≤ totalExamples μ 𝓑 (adaptiveBatch r 𝓑) →
      totalSteps μ 𝓑 (adaptiveBatch r 𝓑) ≤ totalSteps μ 𝓑 Bs) ∧
    (totalSteps μ 𝓑 Bs ≤ totalSteps μ 𝓑 (adaptiveBatch r 𝓑) →
      totalExamples μ 𝓑 (adaptiveBatch r 𝓑) ≤ totalExamples μ 𝓑 Bs) := by
  have h := totalSteps_add_div h𝓑 hi hr hB hBi hdi
  have hR : 0 ≤ ∫ s, (Bs s - adaptiveBatch r 𝓑 s) ^ 2 / (r * Bs s) ∂μ :=
    integral_nonneg fun s => div_nonneg (sq_nonneg _) (mul_pos hr (hB s)).le
  constructor
  · intro hE
    have := div_le_div_of_nonneg_right hE hr.le
    linarith
  · intro hS
    exact (div_le_div_iff_of_pos_right hr).1 (by linarith)

/-- The hypotheses of `totalSteps_add_div` and `adaptiveBatch_optimal` are satisfiable:
`𝓑 = 0`, `r = 1`, a unit batch. -/
example := And.intro (totalSteps_add_div (μ := μ) (𝓑 := 0) (fun _ => le_rfl)
  (integrable_zero _ _ _) one_pos (Bs := 1) (fun _ => one_pos) (integrable_const 1) (by simp))
  (adaptiveBatch_optimal (μ := μ) (𝓑 := 0) (fun _ => le_rfl) (integrable_zero _ _ _) one_pos
    (Bs := 1) (fun _ => one_pos) (integrable_const 1) (by simp))

end Transformer.NoiseScale
