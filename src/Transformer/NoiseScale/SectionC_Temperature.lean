/-
# Temperature and the noise scale

arXiv:1812.06162, Appendix C.

The temperature `T(ε, B) = ε/ε_max(B)` of eq. (C.1), whose `ε_max(B)` is read as the best step
`ε_opt(B)` of eq. (2.6), the paper's notation elsewhere (`temperature`).  "In the case of pure
SGD it is approximated by `T ≈ ε/B` in the small batch regime", up to a factor independent of
`ε` and `B`: `T = (ε/B)(B + B_noise)/ε_max` (`temperature_eq`), `T/(ε/B) → B_noise/ε_max` as
`B → 0` (`tendsto_temperature_div`).  At the best step `T = 1` at every batch size, as the
definition intends (`temperature_stepOpt`).

In the toy model of `SectionC_ToyModel`, with `|G|²` and `GᵀHG` replaced by their means, as the
paper's `≈` does, `B_simple ≈ (B/ε) 2tr(Σ)/tr(HΣ)` and `B_noise ≈ (B/ε) 2tr(HΣ)/tr(H²Σ)`
(`toy_noiseScales`): eq. (C.2), `B_noise ∝ B_simple ∝ 1/T` at `T ∝ ε/B`.  The paper states
`(B/ε) tr(Σ)/tr(H²Σ)` and `(B/ε) tr(HΣ)/tr(H³Σ)`, the values for `M = (ε/B)Σ`, which does not
solve `MH + HM = (ε/B)Σ`; in one dimension the model's values are `2H` times these
(`toy_counterexample`).  Its `B_noise = tr(Σ)H/GᵀHG` is read as `tr(HΣ)/GᵀHG`, eq. (2.8).  The
conclusions survive: the noise scales grow as `ε` decreases or `B` increases and are unchanged
when both scale together (`toy_noiseScales_lt`), and `E[L(θ)] = (ε/B) tr(Σ)/4`: at larger `T`,
"higher up the 'walls' of the potential" (`integral_toyLoss`).  The footnote's `ε/(1 - m)` for
momentum `m` is an instance of the hypothesis on `M`.
-/

import Transformer.NoiseScale.SectionC_ToyModel
import Transformer.NoiseScale.Section2_NoiseScale
import Mathlib.Probability.Distributions.Gaussian.Real

open MeasureTheory ProbabilityTheory Filter Topology
open scoped Matrix

namespace Transformer.NoiseScale

variable {Ω ι : Type*} [Fintype ι] [MeasurableSpace Ω] {P : Measure Ω}

/-- The temperature `T(ε, B) = ε/ε_opt(B)`, eq. (C.1), whose `ε_max(B)` is the best step
`ε_opt(B)` of a batch of `B`, eq. (2.6). -/
noncomputable def temperature (G : ι → ℝ) (H S : Matrix ι ι ℝ) (ε B : ℝ) : ℝ :=
  ε / stepOpt G H S B

/-- `T = (ε/B)(B + B_noise)/ε_max`. -/
theorem temperature_eq (G : ι → ℝ) (H S : Matrix ι ι ℝ) (ε : ℝ) {B : ℝ} (hB : B ≠ 0) :
    temperature G H S ε B = ε / B * ((B + noiseScale G H S) / stepMax G H) := by
  rw [temperature, stepOpt, div_div_eq_mul_div, one_add_div hB]
  ring

/-- The hypothesis of `temperature_eq` is satisfiable: `B = 1`. -/
example (G : ι → ℝ) (H S : Matrix ι ι ℝ) (ε : ℝ) := temperature_eq G H S ε one_ne_zero

/-- **Appendix C**: "In the case of pure SGD it is approximated by `T ≈ ε/B` in the small
batch regime": `T/(ε/B) → B_noise/ε_max` as `B → 0`, a factor independent of `ε` and `B`. -/
theorem tendsto_temperature_div (G : ι → ℝ) (H S : Matrix ι ι ℝ) {ε : ℝ} (hε : ε ≠ 0) :
    Tendsto (fun B => temperature G H S ε B / (ε / B)) (𝓝[>] 0)
      (𝓝 (noiseScale G H S / stepMax G H)) := by
  have h : Tendsto (fun B : ℝ => (B + noiseScale G H S) / stepMax G H) (𝓝[>] 0)
      (𝓝 (noiseScale G H S / stepMax G H)) := by
    simpa using ((continuous_id.add continuous_const).div_const (stepMax G H)).tendsto 0
      |>.mono_left nhdsWithin_le_nhds
  refine h.congr' (eventually_nhdsWithin_of_forall fun B (hB : 0 < B) => ?_)
  simp only
  rw [temperature_eq G H S ε hB.ne', mul_div_cancel_left₀ _ (div_ne_zero hε hB.ne')]

/-- The hypothesis of `tendsto_temperature_div` is satisfiable: `ε = 1`. -/
example (G : ι → ℝ) (H S : Matrix ι ι ℝ) := tendsto_temperature_div G H S one_ne_zero

/-- **Appendix C**: at the best step `ε = ε_opt(B)`, `T = 1` at every batch size: "well-tuned
training runs should have the same temperature at different batch sizes". -/
theorem temperature_stepOpt {G : ι → ℝ} {H S : Matrix ι ι ℝ} (hH : 0 < G ⬝ᵥ H *ᵥ G)
    (hHS : 0 ≤ (H * S).trace) {B : ℝ} (hB : 0 < B) :
    temperature G H S (stepOpt G H S B) B = 1 := by
  have hG : 0 < G ⬝ᵥ G := lt_of_le_of_ne (Finset.sum_nonneg fun a _ => mul_self_nonneg (G a))
    fun h => by simp [dotProduct_self_eq_zero.1 h.symm] at hH
  rw [temperature, div_self (by rw [stepOpt_eq hH hHS hB]; positivity)]

/-- The hypotheses of `temperature_stepOpt` are satisfiable: `G = H = 1`, `Σ = 0`, `B = 1`. -/
example := temperature_stepOpt (G := fun _ : Unit => 1) (H := 1) (S := 0) (by simp) (by simp)
  one_pos

/-- **Appendix C, toy model**, corrected: for symmetric `H` and `θ` of mean zero and
covariance `M` with `MH + HM = (ε/B)Σ`, and `G = Hθ`, `tr(Σ)/E|G|² = (B/ε) 2tr(Σ)/tr(HΣ)` and
`tr(HΣ)/E[GᵀHG] = (B/ε) 2tr(HΣ)/tr(H²Σ)`.  The paper's values `(B/ε) tr(Σ)/tr(H²Σ)` and
`(B/ε) tr(HΣ)/tr(H³Σ)` are refuted by `toy_counterexample`. -/
theorem toy_noiseScales [IsProbabilityMeasure P] [DecidableEq ι] {H S : Matrix ι ι ℝ}
    (hH : Hᵀ = H) {θ : Ω → ι → ℝ} (hθ : ∀ a, MemLp (fun ω => θ ω a) 2 P)
    (h0 : ∀ a, ∫ ω, θ ω a ∂P = 0) {ε B : ℝ}
    (hM : covMatrix θ P * H + H * covMatrix θ P = (ε / B) • S) :
    S.trace / ∫ ω, H *ᵥ θ ω ⬝ᵥ H *ᵥ θ ω ∂P = B / ε * (2 * S.trace / (H * S).trace) ∧
      (H * S).trace / ∫ ω, H *ᵥ θ ω ⬝ᵥ H *ᵥ H *ᵥ θ ω ∂P =
        B / ε * (2 * (H * S).trace / (H ^ 2 * S).trace) := by
  have h1 := integral_toyGrad_quadForm hH hθ h0 0
  have h2 := integral_toyGrad_quadForm hH hθ h0 1
  have l1 := trace_lyapunov hM 1
  have l2 := trace_lyapunov hM 2
  simp only [pow_zero, Matrix.one_mulVec, pow_one, zero_add] at h1 h2 l1 l2
  rw [h1, h2, show (H ^ 2 * covMatrix θ P).trace = ε / B * (H * S).trace / 2 by linarith,
    show (H ^ 3 * covMatrix θ P).trace = ε / B * (H ^ 2 * S).trace / 2 by linarith]
  constructor <;> simp only [div_eq_mul_inv, mul_inv, inv_inv] <;> ring

/-- **Appendix C**: "at larger `T` the neural network parameters are further from the minimum
of the loss, or higher up the 'walls' of the potential": `E[L(θ)] = (ε/B) tr(Σ)/4`. -/
theorem integral_toyLoss [IsProbabilityMeasure P] {H S : Matrix ι ι ℝ} {θ : Ω → ι → ℝ}
    (hθ : ∀ a, MemLp (fun ω => θ ω a) 2 P) (h0 : ∀ a, ∫ ω, θ ω a ∂P = 0) {ε B : ℝ}
    (hM : covMatrix θ P * H + H * covMatrix θ P = (ε / B) • S) :
    ∫ ω, toyLoss H (θ ω) ∂P = ε / B * S.trace / 4 := by
  classical
  have l0 := trace_lyapunov hM 0
  simp only [zero_add, pow_one, pow_zero, Matrix.one_mul] at l0
  have hm : (fun a => ∫ ω, θ ω a ∂P) = 0 := funext h0
  simp only [toyLoss, integral_div, integral_quadForm hθ, hm, zero_dotProduct, zero_add]
  linarith

/-- The hypotheses of `toy_noiseScales` and `integral_toyLoss` are satisfiable: `θ = Σ = 0`. -/
example := And.intro
  (toy_noiseScales (P := Measure.dirac ()) (H := (1 : Matrix Unit Unit ℝ)) (S := 0)
    Matrix.transpose_one (θ := fun _ _ => 0) (fun _ => memLp_const 0) (fun _ => by simp)
    (ε := 1) (B := 1) (by ext; simp [covMatrix]))
  (integral_toyLoss (P := Measure.dirac ()) (H := (1 : Matrix Unit Unit ℝ)) (S := 0)
    (θ := fun _ _ => 0) (fun _ => memLp_const 0) (fun _ => by simp) (ε := 1) (B := 1)
    (by ext; simp [covMatrix]))

/-- **Appendix C, toy model**: the paper's `B_simple ≈ (B/ε) tr(Σ)/tr(H²Σ)` and
`B_noise ≈ (B/ε) tr(HΣ)/tr(H³Σ)` are false.  In one dimension, for `H = h > 0`,
`Σ = ε = B = 1` and `θ` Gaussian of mean zero and variance `M = 1/(2h)`, which solves
`MH + HM = (ε/B)Σ`, both noise scales are `2/h`, the paper's values `1/h²`: no constant factor
reconciles them. -/
theorem toy_counterexample {h : NNReal} (hh : 0 < h) :
    let P := gaussianReal 0 (2 * h)⁻¹
    let θ : ℝ → Unit → ℝ := fun x _ => x
    let H : Matrix Unit Unit ℝ := (h : ℝ) • 1
    let S : Matrix Unit Unit ℝ := 1
    let ε : ℝ := 1
    let B : ℝ := 1
    (∀ a, ∫ x, θ x a ∂P = 0) ∧ covMatrix θ P * H + H * covMatrix θ P = (ε / B) • S ∧
      S.trace / ∫ x, H *ᵥ θ x ⬝ᵥ H *ᵥ θ x ∂P = 2 / h ∧
      B / ε * (S.trace / (H ^ 2 * S).trace) = 1 / h ^ 2 ∧
      (H * S).trace / ∫ x, H *ᵥ θ x ⬝ᵥ H *ᵥ H *ᵥ θ x ∂P = 2 / h ∧
      B / ε * ((H * S).trace / (H ^ 3 * S).trace) = 1 / h ^ 2 := by
  intro P θ H S ε B
  have hh' : (h : ℝ) ≠ 0 := by positivity
  have hθ : ∀ a, MemLp (fun x => θ x a) 2 P := fun _ => memLp_id_gaussianReal' 2 (by simp)
  have h0 : ∀ a, ∫ x, θ x a ∂P = 0 := fun _ => integral_id_gaussianReal
  have hM : covMatrix θ P * H + H * covMatrix θ P = (ε / B) • S := by
    ext a b
    simp [covMatrix, θ, H, S, ε, B, P, covariance_self aemeasurable_id']
    field_simp
    ring
  obtain ⟨n1, n2⟩ := toy_noiseScales (by simp [H]) hθ h0 hM
  refine ⟨h0, hM, ?_, ?_, ?_, ?_⟩
  · rw [n1]
    simp [H, S, ε, B]
  · simp [H, S, ε, B, smul_pow]
  · rw [n2]
    simp [H, S, ε, B, smul_pow]
    field_simp
  · simp [H, S, ε, B, smul_pow]
    field_simp

/-- **Appendix C, toy model**: "the noise scale is expected to increase as we decrease the
learning rate or increase the batch size", and §2.5's "inflated if the learning rate is too
small": at a lower `ε/B` both noise scales are larger, for `tr(Σ)`, `tr(HΣ)`, `tr(H²Σ)` positive.
At equal `ε/B` they are equal (`toy_noiseScales`): "scaling the learning rate and batch size
together should leave the noise scale unchanged". -/
theorem toy_noiseScales_lt [IsProbabilityMeasure P] [DecidableEq ι] {H S : Matrix ι ι ℝ}
    (hH : Hᵀ = H) {θ θ' : Ω → ι → ℝ} (hθ : ∀ a, MemLp (fun ω => θ ω a) 2 P)
    (hθ' : ∀ a, MemLp (fun ω => θ' ω a) 2 P) (h0 : ∀ a, ∫ ω, θ ω a ∂P = 0)
    (h0' : ∀ a, ∫ ω, θ' ω a ∂P = 0) {ε B ε' B' : ℝ}
    (hM : covMatrix θ P * H + H * covMatrix θ P = (ε / B) • S)
    (hM' : covMatrix θ' P * H + H * covMatrix θ' P = (ε' / B') • S) (hS : 0 < S.trace)
    (hHS : 0 < (H * S).trace) (hH2S : 0 < (H ^ 2 * S).trace) (hτ' : 0 < ε' / B')
    (hτ : ε' / B' < ε / B) :
    S.trace / ∫ ω, H *ᵥ θ ω ⬝ᵥ H *ᵥ θ ω ∂P < S.trace / ∫ ω, H *ᵥ θ' ω ⬝ᵥ H *ᵥ θ' ω ∂P ∧
      (H * S).trace / ∫ ω, H *ᵥ θ ω ⬝ᵥ H *ᵥ H *ᵥ θ ω ∂P <
        (H * S).trace / ∫ ω, H *ᵥ θ' ω ⬝ᵥ H *ᵥ H *ᵥ θ' ω ∂P := by
  obtain ⟨n1, n2⟩ := toy_noiseScales hH hθ h0 hM
  obtain ⟨n1', n2'⟩ := toy_noiseScales hH hθ' h0' hM'
  rw [n1, n2, n1', n2', ← inv_div ε B, ← inv_div ε' B']
  have h := (inv_lt_inv₀ (hτ'.trans hτ) hτ').2 hτ
  constructor <;> exact mul_lt_mul_of_pos_right h (by positivity)

/-- The hypotheses of `toy_noiseScales_lt` are satisfiable: `H = Σ = 1`, `θ` Gaussian of
variance `½` at `ε/B = 1`, and `θ/2` at `ε/B = ¼`. -/
example := toy_noiseScales_lt (P := gaussianReal 0 (1 / 2)) (H := (1 : Matrix Unit Unit ℝ))
  (S := 1) Matrix.transpose_one (θ := fun x _ => x) (θ' := fun x _ => 2⁻¹ * x)
  (fun _ => memLp_id_gaussianReal' 2 (by simp))
  (fun _ => (memLp_id_gaussianReal' 2 (by simp)).const_mul _)
  (fun _ => integral_id_gaussianReal)
  (fun _ => by rw [integral_const_mul, integral_id_gaussianReal, mul_zero])
  (ε := 1) (B := 1) (ε' := 1) (B' := 4)
  (by ext; simp [covMatrix, covariance_self aemeasurable_id']; norm_num)
  (by
    have hc : cov[fun x : ℝ => 2⁻¹ * x, fun x => 2⁻¹ * x; gaussianReal 0 (1 / 2)] = 1 / 8 := by
      rw [covariance_const_mul_left, covariance_const_mul_right,
        covariance_self aemeasurable_id', variance_fun_id_gaussianReal]
      norm_num
    ext
    simp only [Matrix.mul_one, Matrix.one_mul, covMatrix, Matrix.add_apply, Matrix.of_apply, hc]
    norm_num)
  (by simp) (by simp) (by simp) (by norm_num) (by norm_num)

end Transformer.NoiseScale
