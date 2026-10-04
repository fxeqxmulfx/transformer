/-
# The bias of the ratio of the estimates

arXiv:1812.06162, Appendix A.1.  After eq. (A.2) the paper notes that the ratio `𝒮/|𝒢|²` is
not an unbiased estimator for `B_noise`, with the footnote "In fact `E[x/y] ≥ E[x]/E[y]` in
general for positive variables".

The footnote is false: for `y` uniform on `{1, 2}` and `x = y²`, `E[x/y] = 3/2 < 5/3 =
E[x]/E[y]` (`not_div_integral_le_integral_div`).  It holds for `x` independent of `y > 0`
with `E[x] ≥ 0` (`div_integral_le_integral_div`): then `E[x/y] = E[x] E[1/y]` and
`E[1/y] ≥ 1/E[y]`.

The remark holds (`integral_traceEst_div_gradSqEst`): two independent per-example gradients
uniform on `{1, 3}`, those of the per-example losses `(θ + x)²/2` at `θ = 0`, so `H = 1`,
with `B_small = 1` and `B_big = 2`, give the ratio the mean `-19/7`, while
`B_noise = tr(Σ)/|G|² = 1/4`, although `𝒮` and `|𝒢|²` are unbiased
(`integral_estimates_batchGrad`).  Averaging over many batches, as the paper does instead
of correcting the bias, makes the ratio consistent: over independent steps the ratio of the
plain averages of `𝒮` and `|𝒢|²` tends almost surely to `tr(Σ)/|G|²` (`tendsto_sum_div_sum`).
The paper averages exponentially, with decays tuned until the estimates are stable.
-/

import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.ProbabilityMassFunction.Integrals
import Mathlib.Probability.StrongLaw
import Transformer.NoiseScale.Section2_NoiseScale
import Transformer.NoiseScale.SectionA_Estimators

open MeasureTheory ProbabilityTheory Filter Topology
open scoped Matrix

namespace Transformer.NoiseScale

/-- A fair coin. -/
noncomputable def coin : Measure Bool := (PMF.uniformOfFintype Bool).toMeasure

instance : IsProbabilityMeasure coin := PMF.toMeasure.isProbabilityMeasure _

theorem integral_coin (f : Bool → ℝ) : ∫ b, f b ∂coin = (f false + f true) / 2 := by
  simp [coin, PMF.integral_eq_sum, PMF.uniformOfFintype_apply]
  ring

theorem integral_coin_prod (f : Bool × Bool → ℝ) :
    ∫ ω, f ω ∂coin.prod coin =
      (f (false, false) + f (false, true) + f (true, false) + f (true, true)) / 4 := by
  rw [integral_prod f Integrable.of_finite]
  simp only [integral_coin]
  ring

/-- The footnote to the remark after eq. (A.2), "`E[x/y] ≥ E[x]/E[y]` in general for positive
variables", is false: `y` uniform on `{1, 2}` and `x = y²` give
`E[x/y] = 3/2 < 5/3 = E[x]/E[y]`. -/
theorem not_div_integral_le_integral_div :
    ¬ ∀ x y : Bool → ℝ, (∀ b, 0 < x b) → (∀ b, 0 < y b) →
      (∫ b, x b ∂coin) / ∫ b, y b ∂coin ≤ ∫ b, x b / y b ∂coin := by
  intro h
  have := h (fun b => cond b 4 1) (fun b => cond b 2 1) (fun b => by cases b <;> norm_num)
    (fun b => by cases b <;> norm_num)
  simp only [integral_coin, Bool.cond_false, Bool.cond_true] at this
  norm_num at this

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- The footnote to the remark after eq. (A.2), corrected: `E[x/y] ≥ E[x]/E[y]` for `x`
independent of `y > 0` with `E[x] ≥ 0`.  The footnote reads "in general for positive
variables"; added are the independence and the integrability of `y` and `1/y`, and the
positivity of `x` is relaxed to `E[x] ≥ 0`. -/
theorem div_integral_le_integral_div [IsProbabilityMeasure P] {x y : Ω → ℝ}
    (hind : IndepFun x y P) (hx : AEStronglyMeasurable x P) (hy : Integrable y P)
    (hy' : Integrable (fun ω => (y ω)⁻¹) P) (hpos : ∀ᵐ ω ∂P, 0 < y ω)
    (hx0 : 0 ≤ ∫ ω, x ω ∂P) :
    (∫ ω, x ω ∂P) / ∫ ω, y ω ∂P ≤ ∫ ω, x ω / y ω ∂P := by
  have hxy : ∫ ω, x ω / y ω ∂P = (∫ ω, x ω ∂P) * ∫ ω, (y ω)⁻¹ ∂P := by
    simp_rw [div_eq_mul_inv]
    exact (hind.comp measurable_id measurable_inv).integral_fun_mul_eq_mul_integral hx hy'.1
  rw [hxy]
  rcases le_or_gt (∫ ω, y ω ∂P) 0 with hm | hm
  · exact (div_nonpos_of_nonneg_of_nonpos hx0 hm).trans
      (mul_nonneg hx0 (integral_nonneg_of_ae (hpos.mono fun ω h => (inv_pos.2 h).le)))
  rw [div_eq_mul_inv]
  refine mul_le_mul_of_nonneg_left ?_ hx0
  set m := ∫ ω, y ω ∂P with hm_def
  have htan : ∀ᵐ ω ∂P, 2 / m - y ω / m ^ 2 ≤ (y ω)⁻¹ := hpos.mono fun ω h => by
    have : 0 ≤ (y ω - m) ^ 2 / (y ω * m ^ 2) := by positivity
    have key : (y ω)⁻¹ - (2 / m - y ω / m ^ 2) = (y ω - m) ^ 2 / (y ω * m ^ 2) := by
      field_simp
      ring
    linarith
  have hI : Integrable (fun ω => 2 / m - y ω / m ^ 2) P :=
    (integrable_const _).sub (hy.div_const _)
  have h := integral_mono_ae hI hy' htan
  rw [integral_sub (integrable_const _) (hy.div_const _), integral_const, integral_div,
    probReal_univ, one_smul, ← hm_def] at h
  calc m⁻¹ = 2 / m - m / m ^ 2 := by field_simp; ring
    _ ≤ _ := h

/-- The hypotheses of `div_integral_le_integral_div` are satisfiable: constants. -/
example := div_integral_le_integral_div (P := Measure.dirac ()) (x := fun _ => (1 : ℝ))
  (y := fun _ => 1) (indepFun_const_left _ _) aestronglyMeasurable_const (integrable_const _)
  (integrable_const _) (by simp) (by simp)

/-- The remark after eq. (A.2): `𝒮/|𝒢|²` is not an unbiased estimator for `B_noise`.  Two
independent per-example gradients uniform on `{1, 3}`, of mean `G = 2` and variance `Σ = 1`,
the first the batch of `B_small = 1` and both that of `B_big = 2`, and `H = 1`: the ratio has
the mean `-19/7`, and `B_noise = 1/4`. -/
theorem integral_traceEst_div_gradSqEst :
    let X : Fin 2 → Bool × Bool → Unit → ℝ :=
      ![fun ω _ => cond ω.1 3 1, fun ω _ => cond ω.2 3 1]
    let ns := fun ω => batchGrad (X ∘ Fin.castLE one_lt_two.le) ω ⬝ᵥ
      batchGrad (X ∘ Fin.castLE one_lt_two.le) ω
    let nb := fun ω => batchGrad X ω ⬝ᵥ batchGrad X ω
    (Pairwise fun i j => IndepFun (X i) (X j) (coin.prod coin)) ∧
      (∀ i a, ∫ ω, X i ω a ∂coin.prod coin = 2) ∧ (∀ i, covMatrix (X i) (coin.prod coin) = 1) ∧
      ∫ ω, traceEst 1 2 (ns ω) (nb ω) / gradSqEst 1 2 (ns ω) (nb ω) ∂coin.prod coin = -19 / 7 ∧
      noiseScale (fun _ : Unit => 2) 1 1 = 1 / 4 := by
  intro X ns nb
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · have h := indepFun_prod (μ := coin) (ν := coin) (X := fun b (_ : Unit) => cond b (3 : ℝ) 1)
      (Y := fun b (_ : Unit) => cond b (3 : ℝ) 1) .of_discrete .of_discrete
    intro i j hij
    fin_cases i <;> fin_cases j
    · exact absurd rfl hij
    · exact h
    · exact h.symm
    · exact absurd rfl hij
  · intro i a
    fin_cases i <;> simp [X, integral_coin_prod] <;> norm_num
  · intro i
    ext a b
    cases a
    cases b
    fin_cases i <;> simp [X, covMatrix, covariance, integral_coin_prod] <;> norm_num
  · simp [ns, nb, X, batchGrad, traceEst, gradSqEst, integral_coin_prod, dotProduct]
    norm_num
  · simp [noiseScale, dotProduct]
    norm_num

/-- The remark after eq. (A.2): averaging over many batches removes the bias.  For the
estimates `𝒮_k`, `|𝒢|²_k` of independent steps `k`, the ratio of their sums tends almost
surely to `E[𝒮]/E[|𝒢|²]`, by the strong law of large numbers. -/
theorem tendsto_sum_div_sum {S N : ℕ → Ω → ℝ} (hS : Integrable (S 0) P)
    (hN : Integrable (N 0) P) (hindS : Pairwise fun i j => IndepFun (S i) (S j) P)
    (hindN : Pairwise fun i j => IndepFun (N i) (N j) P) (hidS : ∀ k, IdentDistrib (S k) (S 0) P P)
    (hidN : ∀ k, IdentDistrib (N k) (N 0) P P) (hN0 : ∫ ω, N 0 ω ∂P ≠ 0) :
    ∀ᵐ ω ∂P, Tendsto (fun n => (∑ k ∈ Finset.range n, S k ω) / ∑ k ∈ Finset.range n, N k ω)
      atTop (𝓝 ((∫ ω, S 0 ω ∂P) / ∫ ω, N 0 ω ∂P)) := by
  filter_upwards [strong_law_ae S hS hindS hidS, strong_law_ae N hN hindN hidN] with ω h1 h2
  refine (h1.div h2 hN0).congr' ?_
  filter_upwards [eventually_gt_atTop 0] with n hn
  simp only [smul_eq_mul]
  exact mul_div_mul_left _ _ (inv_ne_zero (Nat.cast_ne_zero.2 hn.ne'))

/-- The hypotheses of `tendsto_sum_div_sum` are satisfiable: constant estimates. -/
example := tendsto_sum_div_sum (P := Measure.dirac ()) (S := fun _ _ => (1 : ℝ))
  (N := fun _ _ => 1) (integrable_const _) (integrable_const _)
  (fun _ _ _ => indepFun_const_left _ _) (fun _ _ _ => indepFun_const_left _ _)
  (fun _ => .refl aemeasurable_const) (fun _ => .refl aemeasurable_const) (by simp)

end Transformer.NoiseScale
