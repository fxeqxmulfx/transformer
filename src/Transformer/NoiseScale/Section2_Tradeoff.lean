/-
# The tradeoff between steps and examples

arXiv:1812.06162, §2.3, with eq. (D.1) of Appendix D, from which the paper derives it.
A run is a trajectory of full-batch steps `s`, carried by a finite measure `μ` (the
paper's `ds`), along which the noise scale is `𝓑(s)` and the batch `B(s)`.  It takes
`S = ∫(1 + 𝓑/B) ds` steps and processes `E = ∫(𝓑 + B) ds` examples, eq. (D.1)
(`totalSteps`, `totalExamples`): at least `S_min = ∫ds` and `E_min = ∫𝓑 ds`
(`le_totalSteps`, `le_totalExamples`), the limits as a constant batch grows and shrinks
(`tendsto_totalSteps`, `tendsto_totalExamples`), the paper's "minimum possible".

At a constant batch `B`, `E = BS` (`totalExamples_const`) and `S = S_min(1 + B_crit/B)`
(`totalSteps_const`), with the critical batch `B_crit = E_min/S_min` of eq. (2.12)
(`critBatch`); hence `S/S_min - 1 = (E/E_min - 1)⁻¹`, eq. (2.11) (`totalSteps_tradeoff`),
and at `B = B_crit` a run takes twice the least steps and twice the least examples
(`totalSteps_critBatch`).  The paper fits `B_crit` to measured runs; here it is the
model's `E_min/S_min`, the mean noise scale over the run, the noise scale itself when that
is constant (`critBatch_const`).
-/

import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Topology.Algebra.Order.Field

open MeasureTheory Filter Topology

namespace Transformer.NoiseScale

variable {T : Type*} [MeasurableSpace T]

/-- The steps `S = ∫(1 + 𝓑(s)/B(s)) ds` of a run, eq. (D.1). -/
noncomputable def totalSteps (μ : Measure T) (𝓑 Bs : T → ℝ) : ℝ := ∫ s, (1 + 𝓑 s / Bs s) ∂μ

/-- The examples `E = ∫(𝓑(s) + B(s)) ds` of a run, eq. (D.1). -/
noncomputable def totalExamples (μ : Measure T) (𝓑 Bs : T → ℝ) : ℝ := ∫ s, (𝓑 s + Bs s) ∂μ

/-- The critical batch `B_crit = E_min/S_min`, eq. (2.12). -/
noncomputable def critBatch (μ : Measure T) (𝓑 : T → ℝ) : ℝ := (∫ s, 𝓑 s ∂μ) / μ.real Set.univ

variable {μ : Measure T} [IsFiniteMeasure μ] {𝓑 : T → ℝ}

theorem totalSteps_eq {Bs : T → ℝ} (h : Integrable (fun s => 𝓑 s / Bs s) μ) :
    totalSteps μ 𝓑 Bs = μ.real Set.univ + ∫ s, 𝓑 s / Bs s ∂μ := by
  rw [totalSteps, integral_add (integrable_const 1) h, integral_const, smul_eq_mul, mul_one]

/-- The hypothesis of `totalSteps_eq` is satisfiable: no noise. -/
example (Bs : T → ℝ) : totalSteps μ 0 Bs = μ.real Set.univ + ∫ s, (0 : T → ℝ) s / Bs s ∂μ :=
  totalSteps_eq (by simp)

omit [IsFiniteMeasure μ] in
theorem totalExamples_eq {Bs : T → ℝ} (h𝓑 : Integrable 𝓑 μ) (hB : Integrable Bs μ) :
    totalExamples μ 𝓑 Bs = (∫ s, 𝓑 s ∂μ) + ∫ s, Bs s ∂μ :=
  integral_add h𝓑 hB

/-- The hypotheses of `totalExamples_eq` are satisfiable: no noise, a unit batch. -/
example : totalExamples μ 0 1 = (∫ s, (0 : T → ℝ) s ∂μ) + ∫ s, (1 : T → ℝ) s ∂μ :=
  totalExamples_eq (integrable_zero _ _ _) (integrable_const 1)

/-- §2.3: a run takes at least `S_min = ∫ds` steps. -/
theorem le_totalSteps {Bs : T → ℝ} (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hB : ∀ s, 0 < Bs s)
    (h : Integrable (fun s => 𝓑 s / Bs s) μ) : μ.real Set.univ ≤ totalSteps μ 𝓑 Bs := by
  rw [totalSteps_eq h]
  exact le_add_of_nonneg_right (integral_nonneg fun s => div_nonneg (h𝓑 s) (hB s).le)

/-- The hypotheses of `le_totalSteps` are satisfiable: no noise, a unit batch. -/
example : μ.real Set.univ ≤ totalSteps μ 0 1 :=
  le_totalSteps (fun _ => le_rfl) (fun _ => one_pos) (by simp)

omit [IsFiniteMeasure μ] in
/-- §2.3: a run processes at least `E_min = ∫𝓑 ds` examples. -/
theorem le_totalExamples {Bs : T → ℝ} (h𝓑 : Integrable 𝓑 μ) (hB : ∀ s, 0 < Bs s)
    (hBi : Integrable Bs μ) : ∫ s, 𝓑 s ∂μ ≤ totalExamples μ 𝓑 Bs := by
  rw [totalExamples_eq h𝓑 hBi]
  exact le_add_of_nonneg_right (integral_nonneg fun s => (hB s).le)

/-- The hypotheses of `le_totalExamples` are satisfiable: no noise, a unit batch. -/
example : ∫ s, (0 : T → ℝ) s ∂μ ≤ totalExamples μ 0 1 :=
  le_totalExamples (integrable_zero _ _ _) (fun _ => one_pos) (integrable_const 1)

/-- §2.3: at a constant batch, `S = S_min + E_min/B`. -/
theorem totalSteps_const_eq (h𝓑 : Integrable 𝓑 μ) (B : ℝ) :
    totalSteps μ 𝓑 (fun _ => B) = μ.real Set.univ + (∫ s, 𝓑 s ∂μ) / B := by
  rw [totalSteps_eq (h𝓑.div_const B), integral_div]

/-- The hypothesis of `totalSteps_const_eq` is satisfiable: no noise. -/
example (B : ℝ) : totalSteps μ 0 (fun _ => B) = μ.real Set.univ + (∫ s, (0 : T → ℝ) s ∂μ) / B :=
  totalSteps_const_eq (integrable_zero _ _ _) B

/-- §2.3: at a constant batch `B`, `E = BS`. -/
theorem totalExamples_const (h𝓑 : Integrable 𝓑 μ) {B : ℝ} (hB : B ≠ 0) :
    totalExamples μ 𝓑 (fun _ => B) = B * totalSteps μ 𝓑 (fun _ => B) := by
  rw [totalExamples_eq h𝓑 (integrable_const B), totalSteps_const_eq h𝓑, integral_const,
    smul_eq_mul]
  field_simp
  ring

/-- The hypotheses of `totalExamples_const` are satisfiable: no noise, `B = 1`. -/
example : totalExamples μ 0 (fun _ => 1) = 1 * totalSteps μ 0 (fun _ => 1) :=
  totalExamples_const (integrable_zero _ _ _) one_ne_zero

/-- Eqs. (2.11)–(2.12) with `E = BS`: at a constant batch, `S = S_min(1 + B_crit/B)`. -/
theorem totalSteps_const (hμ : μ.real Set.univ ≠ 0) (h𝓑 : Integrable 𝓑 μ) (B : ℝ) :
    totalSteps μ 𝓑 (fun _ => B) = μ.real Set.univ * (1 + critBatch μ 𝓑 / B) := by
  rw [totalSteps_const_eq h𝓑, critBatch]
  field_simp

/-- The hypotheses of `totalSteps_const` are satisfiable: no noise over one step. -/
example (B : ℝ) : totalSteps (Measure.dirac ()) 0 (fun _ => B) =
    (Measure.dirac ()).real Set.univ * (1 + critBatch (Measure.dirac ()) 0 / B) :=
  totalSteps_const (by simp) (integrable_zero _ _ _) B

/-- **Eq. (2.11)**: at a constant batch `B > 0`, `S/S_min - 1 = (E/E_min - 1)⁻¹`. -/
theorem totalSteps_tradeoff (hμ : 0 < μ.real Set.univ) (h𝓑 : Integrable 𝓑 μ)
    (hE : 0 < ∫ s, 𝓑 s ∂μ) {B : ℝ} (hB : 0 < B) :
    totalSteps μ 𝓑 (fun _ => B) / μ.real Set.univ - 1 =
      (totalExamples μ 𝓑 (fun _ => B) / (∫ s, 𝓑 s ∂μ) - 1)⁻¹ := by
  rw [totalExamples_eq h𝓑 (integrable_const B), totalSteps_const_eq h𝓑, integral_const,
    smul_eq_mul]
  field_simp
  ring

/-- The hypotheses of `totalSteps_tradeoff` are satisfiable: unit noise over one step. -/
example : totalSteps (Measure.dirac ()) 1 (fun _ => 1) / (Measure.dirac ()).real Set.univ - 1 =
    (totalExamples (Measure.dirac ()) 1 (fun _ => 1) / (∫ s, (1 : Unit → ℝ) s ∂Measure.dirac ())
      - 1)⁻¹ :=
  totalSteps_tradeoff (by simp) (integrable_const 1) (by simp) one_pos

/-- §2.3: at `B = B_crit` both sides of eq. (2.11) are 1, so a run takes twice the least
steps and twice the least examples. -/
theorem totalSteps_critBatch (hμ : 0 < μ.real Set.univ) (h𝓑 : Integrable 𝓑 μ)
    (hE : 0 < ∫ s, 𝓑 s ∂μ) :
    totalSteps μ 𝓑 (fun _ => critBatch μ 𝓑) = 2 * μ.real Set.univ ∧
      totalExamples μ 𝓑 (fun _ => critBatch μ 𝓑) = 2 * ∫ s, 𝓑 s ∂μ := by
  rw [totalExamples_eq h𝓑 (integrable_const _), totalSteps_const_eq h𝓑, integral_const,
    smul_eq_mul, critBatch]
  constructor <;> field_simp <;> ring

/-- The hypotheses of `totalSteps_critBatch` are satisfiable: unit noise over one step. -/
example : totalSteps (Measure.dirac ()) 1 (fun _ => critBatch (Measure.dirac ()) 1) =
      2 * (Measure.dirac ()).real Set.univ ∧
    totalExamples (Measure.dirac ()) 1 (fun _ => critBatch (Measure.dirac ()) 1) =
      2 * ∫ s, (1 : Unit → ℝ) s ∂Measure.dirac () :=
  totalSteps_critBatch (by simp) (integrable_const 1) (by simp)

omit [IsFiniteMeasure μ] in
/-- §2.3: "our model predicts `B_crit ≈ B_noise`": for a constant noise scale `b`, `B_crit = b`;
in general `B_crit = ∫𝓑 ds/∫ds` is its mean over the run. -/
theorem critBatch_const (hμ : μ.real Set.univ ≠ 0) (b : ℝ) : critBatch μ (fun _ => b) = b := by
  rw [critBatch, integral_const, smul_eq_mul, mul_div_cancel_left₀ b hμ]

/-- The hypothesis of `critBatch_const` is satisfiable: one step. -/
example (b : ℝ) : critBatch (Measure.dirac ()) (fun _ => b) = b := critBatch_const (by simp) b

/-- §2.3: `S → S_min` as a constant batch grows, `B ≫ 𝓑`. -/
theorem tendsto_totalSteps (h𝓑 : Integrable 𝓑 μ) :
    Tendsto (fun B : ℝ => totalSteps μ 𝓑 fun _ => B) atTop (𝓝 (μ.real Set.univ)) := by
  simp_rw [totalSteps_const_eq h𝓑]
  have h := (tendsto_const_nhds (x := ∫ s, 𝓑 s ∂μ) (f := (atTop : Filter ℝ))).div_atTop
    tendsto_id
  simpa using (tendsto_const_nhds (x := μ.real Set.univ)).add h

/-- The hypothesis of `tendsto_totalSteps` is satisfiable: no noise. -/
example : Tendsto (fun B : ℝ => totalSteps μ 0 fun _ => B) atTop (𝓝 (μ.real Set.univ)) :=
  tendsto_totalSteps (integrable_zero _ _ _)

/-- §2.3: `E → E_min` as a constant batch shrinks, `B ≪ 𝓑`. -/
theorem tendsto_totalExamples (h𝓑 : Integrable 𝓑 μ) :
    Tendsto (fun B : ℝ => totalExamples μ 𝓑 fun _ => B) (𝓝[>] 0) (𝓝 (∫ s, 𝓑 s ∂μ)) := by
  simp_rw [totalExamples_eq h𝓑 (integrable_const _), integral_const, smul_eq_mul]
  have : Continuous fun B : ℝ => (∫ s, 𝓑 s ∂μ) + μ.real Set.univ * B := by fun_prop
  simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds

/-- The hypothesis of `tendsto_totalExamples` is satisfiable: no noise. -/
example : Tendsto (fun B : ℝ => totalExamples μ 0 fun _ => B) (𝓝[>] 0)
    (𝓝 (∫ s, (0 : T → ℝ) s ∂μ)) :=
  tendsto_totalExamples (integrable_zero _ _ _)

end Transformer.NoiseScale
