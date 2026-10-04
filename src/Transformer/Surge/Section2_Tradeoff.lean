/-
# The tradeoff between steps and examples

arXiv:2405.14578, §2.2, after Theorem 5, and Appendix F.  Eq. (18) has the form of eq. (2.7) of
arXiv:1812.06162: `δS` steps of batch `B` make the progress `ΔL_max` of one step of batch
`B → ∞` if and only if `δS = 1 + B_noise/B` (`mul_lossDropSign_eq_iff`), at `BδS` examples,
the integrands of eq. (44): `totalSteps` and `totalExamples` of `NoiseScale.Section2_Tradeoff`,
with the noise scale `B_noise(s)` along a run carried by `ν`, the paper's `ds`
(`runNoiseBatch`).  Their conclusions "still hold": a run takes at least `S_min = ∫ds` steps and
`E_min = ∫B_noise ds` examples, eq. (45) (`le_totalSteps_runNoiseBatch`), the limits as a
constant batch grows and shrinks (`tendsto_totalSteps`, `tendsto_totalExamples`), and at a
constant batch `(S/S_min - 1)(E/E_min - 1) = 1`, eq. (20) (`totalSteps_mul_totalExamples`).

At constant `μ, σ, H` the critical batch `B_crit = E_min/S_min` of eq. (21) is `B_noise`, so a
batch is the peak `B_peak` of the learning rate of eq. (13) if and only if it is this balance
point (`lrSign_signLin_eq_lrPeak_iff`); in general `B_crit` is the mean of `B_noise` over the run
(`critBatch`), the paper's "≈".  Under the fit `B_crit ≈ B_*/L^{1/α_B}` of eq. (21), taken as a
hypothesis on `B_noise`, `B_peak` grows as the loss falls (`strictAntiOn_noiseBatch`).
-/

import Transformer.NoiseScale.Section2_Tradeoff
import Transformer.Surge.Section2_LossDrop
import Transformer.Surge.Section2_PeakRate
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open MeasureTheory Real

namespace Transformer.Surge

open Transformer.NoiseScale

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Eq. (44)**: `δS` steps of batch `B`, each dropping the loss by the
`ΔL_opt(B) = ΔL_max/(1 + B_noise/B)` of eq. (18), drop it by the `ΔL_max` of one step of batch
`B → ∞` if and only if `δS = 1 + B_noise/B`, when `ΣH_ii ≥ 0`, `S > 0`, `M ≠ 0`, `B > 0`. -/
theorem mul_lossDropSign_eq_iff {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} (hD : 0 ≤ ∑ i, H i i)
    (hS : 0 < offSum μ σ H) (hM : snrSum μ σ ≠ 0) {B : ℝ} (hB : 0 < B) (δS : ℝ) :
    δS * lossDropSign (signLin μ σ B) μ H = lossDropSignMax μ σ H ↔
      δS = 1 + noiseBatch μ σ H / B := by
  have hL : 0 < lossDropSignMax μ σ H := by
    rw [lossDropSignMax_eq]
    exact div_pos (sq_pos_iff.2 hM) (by linarith)
  have hn : 0 < 1 + noiseBatch μ σ H / B := by
    have := div_nonneg (noiseBatch_nonneg hD hS) hB.le
    linarith
  rw [lossDropSign_signLin_eq hS.ne' hB, ← mul_div_assoc, div_eq_iff hn.ne',
    mul_comm (lossDropSignMax μ σ H), mul_left_inj' hL.ne']

/-- The hypotheses of `mul_lossDropSign_eq_iff` are satisfiable: `μ = σ = 1`, all `H_ij = 1`,
`B = 1`. -/
example := mul_lossDropSign_eq_iff (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])
  (by simp [snrSum]) one_pos 2

variable {T : Type*} [MeasurableSpace T] {ν : Measure T}

/-- The noise batch `B_noise(s)` of eq. (14) along a run with gradient means `μ(s)`, standard
deviations `σ(s)` and Hessians `H(s)`: the noise scale of eq. (44). -/
noncomputable def runNoiseBatch (μ σ : T → ι → ℝ) (H : T → Matrix ι ι ℝ) (s : T) : ℝ :=
  noiseBatch (μ s) (σ s) (H s)

/-- **Eq. (45)**: along a run with `ΣH_ii ≥ 0` and `S > 0`, at batches `B(s) > 0`, a run takes
at least `S_min = ∫ds` steps and processes at least `E_min = ∫B_noise ds` examples. -/
theorem le_totalSteps_runNoiseBatch [IsFiniteMeasure ν] {μ σ : T → ι → ℝ}
    {H : T → Matrix ι ι ℝ} {Bs : T → ℝ} (hD : ∀ s, 0 ≤ ∑ i, H s i i)
    (hS : ∀ s, 0 < offSum (μ s) (σ s) (H s)) (hB : ∀ s, 0 < Bs s)
    (hi : Integrable (runNoiseBatch μ σ H) ν) (hBi : Integrable Bs ν)
    (h : Integrable (fun s => runNoiseBatch μ σ H s / Bs s) ν) :
    ν.real Set.univ ≤ totalSteps ν (runNoiseBatch μ σ H) Bs ∧
      ∫ s, runNoiseBatch μ σ H s ∂ν ≤ totalExamples ν (runNoiseBatch μ σ H) Bs :=
  ⟨le_totalSteps (fun s => noiseBatch_nonneg (hD s) (hS s)) hB h, le_totalExamples hi hB hBi⟩

/-- The hypotheses of `le_totalSteps_runNoiseBatch` are satisfiable: one step at `μ = σ = 1`,
all `H_ij = 1`, `B = 1`. -/
example := le_totalSteps_runNoiseBatch (ν := Measure.dirac ()) (μ := fun _ (_ : Fin 2) => 1)
  (σ := fun _ _ => 1) (H := fun _ => Matrix.of fun _ _ => 1) (Bs := fun _ => 1) (fun _ => by simp)
  (fun _ => by simp [offSum, Fin.sum_univ_two]) (fun _ => one_pos) .of_finite .of_finite
  .of_finite

/-- **Eq. (20)**: at a constant batch `B > 0`, a run with `S_min = ∫ds > 0` and
`E_min = ∫B_noise ds > 0` has `(S/S_min - 1)(E/E_min - 1) = 1`. -/
theorem totalSteps_mul_totalExamples [IsFiniteMeasure ν] {μ σ : T → ι → ℝ}
    {H : T → Matrix ι ι ℝ} (hν : 0 < ν.real Set.univ) (hi : Integrable (runNoiseBatch μ σ H) ν)
    (hE : 0 < ∫ s, runNoiseBatch μ σ H s ∂ν) {B : ℝ} (hB : 0 < B) :
    (totalSteps ν (runNoiseBatch μ σ H) (fun _ => B) / ν.real Set.univ - 1) *
      (totalExamples ν (runNoiseBatch μ σ H) (fun _ => B) / ∫ s, runNoiseBatch μ σ H s ∂ν - 1) =
        1 := by
  rw [totalSteps_tradeoff hν hi hE hB]
  refine inv_mul_cancel₀ ?_
  rw [totalExamples_eq hi (integrable_const B), integral_const, smul_eq_mul, add_div,
    div_self hE.ne', add_sub_cancel_left]
  exact div_ne_zero (mul_pos hν hB).ne' hE.ne'

/-- The hypotheses of `totalSteps_mul_totalExamples` are satisfiable: one step at `μ = σ = 1`,
all `H_ij = 1`, `B = 1`. -/
example := totalSteps_mul_totalExamples (ν := Measure.dirac ()) (μ := fun _ (_ : Fin 2) => 1)
  (σ := fun _ _ => 1) (H := fun _ => Matrix.of fun _ _ => 1) (by simp) (integrable_const _)
  (by simp [runNoiseBatch, noiseBatch, offSum, Fin.sum_univ_two]; positivity) one_pos

variable {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} {B : ℝ}

/-- §2.2, after eq. (21): "`B_peak = B_noise` is not only the local maximum of the optimal
learning rate, but also the balance point between training speed and data efficiency".  Over a
run at constant `μ, σ, H` with `∫ds ≠ 0`, when `ΣH_ii > 0`, `S > 0` and `M > 0`, the learning
rate of eq. (13) at `B > 0` is its maximum `ε_max` if and only if `B` is the critical batch
`B_crit = E_min/S_min` of eq. (21), at which a run takes twice the least steps and twice the
least examples (`totalSteps_critBatch`). -/
theorem lrSign_signLin_eq_lrPeak_iff (hν : ν.real Set.univ ≠ 0) (hD : 0 < ∑ i, H i i)
    (hS : 0 < offSum μ σ H) (hM : 0 < snrSum μ σ) (hB : 0 < B) :
    lrSign (signLin μ σ B) μ H = lrPeak μ σ H ↔ B = critBatch ν fun _ => noiseBatch μ σ H := by
  rw [critBatch_const hν]
  refine ⟨fun h => ?_, fun h => by rw [h]; exact lrSign_signLin_noiseBatch hD hS⟩
  by_contra hne
  exact (lrSign_signLin_lt hD hS hM hB hne).ne h

/-- The hypotheses of `lrSign_signLin_eq_lrPeak_iff` are satisfiable: one step at `μ = σ = 1`,
all `H_ij = 1`, `B = 1`. -/
example := lrSign_signLin_eq_lrPeak_iff (ν := Measure.dirac ()) (μ := fun _ : Fin 2 => 1)
  (σ := fun _ => 1) (H := Matrix.of fun _ _ => 1) (by simp) (by simp)
  (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum]) one_pos

omit [DecidableEq ι] in
/-- §2.2, after eq. (21), and §2.3: "as the training progresses and the loss decreases,
according to Eq. (21), `B_peak` will gradually become larger".  Under the fit
`B_noise ≈ B_crit ≈ B_*/L^{1/α_B}` of eq. (21), after Kaplan et al., eq. (1.4), taken as a
hypothesis on the statistics `μ(L), σ(L), H(L)` at the loss `L > 0`, with `B_*, α_B > 0`, the
peak `B_peak = B_noise` (`lrSign_signLin_noiseBatch`) grows as `L` falls. -/
theorem strictAntiOn_noiseBatch [DecidableEq ι] {μ σ : ℝ → ι → ℝ} {H : ℝ → Matrix ι ι ℝ}
    {Bs αB : ℝ} (hBs : 0 < Bs) (hα : 0 < αB)
    (h : ∀ L > 0, noiseBatch (μ L) (σ L) (H L) = Bs / L ^ (1 / αB)) :
    StrictAntiOn (fun L => noiseBatch (μ L) (σ L) (H L)) (Set.Ioi 0) := by
  intro L₁ hL₁ L₂ hL₂ hL
  simp only [h L₁ hL₁, h L₂ hL₂]
  exact div_lt_div_of_pos_left hBs (Real.rpow_pos_of_pos hL₁ _)
    (Real.rpow_lt_rpow hL₁.le hL (one_div_pos.2 hα))

/-- The hypotheses of `strictAntiOn_noiseBatch` are satisfiable: `μ = σ = 1`, `H_ii = 1/L`,
`H_ij = 1` off the diagonal, `B_* = π/2`, `α_B = 1`. -/
example := strictAntiOn_noiseBatch (μ := fun _ (_ : Fin 2) => 1) (σ := fun _ _ => 1)
  (H := fun L => Matrix.of fun i j => if i = j then L⁻¹ else 1) (Bs := π / 2) (αB := 1)
  (by positivity) one_pos fun L _ => by
    simp [noiseBatch, offSum, Fin.sum_univ_two]
    ring

end Transformer.Surge
