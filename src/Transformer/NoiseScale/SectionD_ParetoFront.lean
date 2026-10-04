/-
# The Pareto front of adaptive batches

arXiv:1812.06162, Appendix D, "Theory".  "If the distribution of training examples (and hence
the batch size schedule) is optimal, then transferring examples from one part of training to
another should not save any optimization steps.  This means that the exchange rate `r` should
be constant throughout training": for a noise scale `𝓑 > 0`, a positive schedule taking the
fewest steps for its examples has `B²/𝓑 = r` and `B = √(r𝓑)`, eq. (D.3), almost everywhere
(`exchangeRate_const`).  With `adaptiveBatch_optimal` of `SectionD_ExchangeRate`, the
schedules `√(r𝓑)` are the Pareto front.

Along it `S_tot/S_min - 1 = γ(E_tot/E_min - 1)⁻¹`, eq. (D.4), with
`γ = (∫√𝓑 ds)²/(S_min E_min)` of eq. (D.5) (`paretoGamma`,
`totalSteps_adaptiveBatch_tradeoff`).  At `r = E_min/S_min` both ratios are `1 + √γ`, the
footnote's choice (`adaptiveBatch_minRatio`).  `S_tot → S_min` as `r → ∞` and `E_tot → E_min`
as `r → 0`, reading the paper's "inserting `B ≫ 𝓑` and `B ≪ 𝓑` respectively into (D.3)"
(`tendsto_totalSteps_adaptiveBatch`, `tendsto_totalExamples_adaptiveBatch`); that they are "the
minimum possible" over every schedule is `le_totalSteps` and `le_totalExamples` of
`Section2_Tradeoff`.
-/

import Transformer.NoiseScale.SectionD_ExchangeRate

open MeasureTheory Filter Topology

namespace Transformer.NoiseScale

variable {T : Type*} [MeasurableSpace T] {μ : Measure T} [IsFiniteMeasure μ] {𝓑 : T → ℝ}

/-- **Appendix D**: "if the distribution of training examples (and hence the batch size
schedule) is optimal ... the exchange rate `r` should be constant throughout training.  Thus the
batch size should be varied in proportion with the square root of the noise scale": a positive
schedule taking the fewest steps among positive schedules on no more examples has
`B²/𝓑 = r` and `B = √(r𝓑)` almost everywhere, for some `r > 0`. -/
theorem exchangeRate_const (h𝓑 : ∀ s, 0 < 𝓑 s) (hi : Integrable 𝓑 μ) {Bs : T → ℝ}
    (hB : ∀ s, 0 < Bs s) (hBi : Integrable Bs μ) (hdi : Integrable (fun s => 𝓑 s / Bs s) μ)
    (hopt : ∀ B' : T → ℝ, (∀ s, 0 < B' s) → Integrable B' μ →
      Integrable (fun s => 𝓑 s / B' s) μ → totalExamples μ 𝓑 B' ≤ totalExamples μ 𝓑 Bs →
        totalSteps μ 𝓑 Bs ≤ totalSteps μ 𝓑 B') :
    ∃ r, 0 < r ∧ ∀ᵐ s ∂μ, Bs s ^ 2 / 𝓑 s = r ∧ Bs s = adaptiveBatch r 𝓑 s := by
  have h𝓑' : ∀ s, 0 ≤ 𝓑 s := fun s => (h𝓑 s).le
  rcases eq_or_ne μ 0 with rfl | hμ
  · exact ⟨1, one_pos, by simp⟩
  have hpos : ∀ f : T → ℝ, (∀ s, 0 < f s) → Integrable f μ → 0 < ∫ s, f s ∂μ := fun f hf hfi =>
    (integral_pos_iff_support_of_nonneg (fun s => (hf s).le) hfi).2 <| by
      rw [show Function.support f = Set.univ from Set.eq_univ_of_forall fun s => (hf s).ne']
      exact Measure.measure_univ_pos.2 hμ
  have hc := hpos _ (fun s => Real.sqrt_pos.2 (h𝓑 s)) (integrable_sqrt h𝓑' hi)
  obtain ⟨ρ, hρ, hρc⟩ : ∃ ρ, 0 < ρ ∧ ρ * ∫ s, √(𝓑 s) ∂μ = ∫ s, Bs s ∂μ :=
    ⟨_, div_pos (hpos _ hB hBi) hc, div_mul_cancel₀ _ hc.ne'⟩
  have hr : 0 < ρ ^ 2 := by positivity
  have hE : totalExamples μ 𝓑 (adaptiveBatch (ρ ^ 2) 𝓑) = totalExamples μ 𝓑 Bs := by
    rw [totalExamples_adaptiveBatch h𝓑' hi hr.le, Real.sqrt_sq hρ.le, hρc,
      totalExamples_eq hi hBi]
  have hqi : Integrable (adaptiveBatch (ρ ^ 2) 𝓑) μ := by
    rw [adaptiveBatch_eq hr.le]
    exact (integrable_sqrt h𝓑' hi).const_mul _
  have hS := hopt (adaptiveBatch (ρ ^ 2) 𝓑) (fun s => Real.sqrt_pos.2 (mul_pos hr (h𝓑 s))) hqi
    (by rw [div_adaptiveBatch h𝓑' hr]; exact (integrable_sqrt h𝓑' hi).div_const _) hE.le
  have h := totalSteps_add_div h𝓑' hi hr hB hBi hdi
  rw [hE] at h
  have e : (fun s => (Bs s - adaptiveBatch (ρ ^ 2) 𝓑 s) ^ 2 / (ρ ^ 2 * Bs s)) =
      fun s => 𝓑 s / Bs s + Bs s / ρ ^ 2 - 2 * adaptiveBatch (ρ ^ 2) 𝓑 s / ρ ^ 2 :=
    funext fun s => sq_sub_sqrt_div (h𝓑' s) hr (hB s)
  have hR0 : 0 ≤ fun s => (Bs s - adaptiveBatch (ρ ^ 2) 𝓑 s) ^ 2 / (ρ ^ 2 * Bs s) :=
    fun s => div_nonneg (sq_nonneg _) (mul_pos hr (hB s)).le
  have hRi : Integrable (fun s => (Bs s - adaptiveBatch (ρ ^ 2) 𝓑 s) ^ 2 / (ρ ^ 2 * Bs s)) μ := by
    rw [e]
    exact (hdi.add (hBi.div_const _)).sub ((hqi.const_mul 2).div_const _)
  have hR := (integral_eq_zero_iff_of_nonneg hR0 hRi).1
    (le_antisymm (by linarith) (integral_nonneg hR0))
  refine ⟨ρ ^ 2, hr, ?_⟩
  filter_upwards [hR] with s hs
  have hBq : Bs s = adaptiveBatch (ρ ^ 2) 𝓑 s := by
    rcases div_eq_zero_iff.1 hs with h0 | h0
    · exact sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 h0)
    · exact absurd h0 (mul_pos hr (hB s)).ne'
  refine ⟨?_, hBq⟩
  rw [hBq, adaptiveBatch, Real.sq_sqrt (mul_pos hr (h𝓑 s)).le,
    mul_div_cancel_right₀ _ (h𝓑 s).ne']

/-- The hypotheses of `exchangeRate_const` are satisfiable: `𝓑 = 1` and `B = √𝓑`, optimal by
`adaptiveBatch_optimal`. -/
example : ∃ r, 0 < r ∧ ∀ᵐ s ∂μ, adaptiveBatch 1 1 s ^ 2 / (1 : T → ℝ) s = r ∧
    adaptiveBatch 1 1 s = adaptiveBatch r 1 s := by
  have e : adaptiveBatch 1 (1 : T → ℝ) = 1 := funext fun _ => by simp [adaptiveBatch]
  refine exchangeRate_const (fun _ => one_pos) (integrable_const 1) (by simp [e])
    (by rw [e]; exact integrable_const 1) (by simp [e]) fun B' hB' hBi' hdi' hE' => ?_
  exact (adaptiveBatch_optimal (fun _ => zero_le_one) (integrable_const 1) one_pos hB' hBi'
    hdi').1 hE'

/-- `γ = (∫√𝓑 ds)²/(S_min E_min)`, eq. (D.5), which "parameterizes the amount of variation of
the noise scale over the course of training". -/
noncomputable def paretoGamma (μ : Measure T) (𝓑 : T → ℝ) : ℝ :=
  (∫ s, √(𝓑 s) ∂μ) ^ 2 / (μ.real Set.univ * ∫ s, 𝓑 s ∂μ)

/-- **Eq. (D.4)**: along the schedules `√(r𝓑)`, `S_tot/S_min - 1 = γ(E_tot/E_min - 1)⁻¹`. -/
theorem totalSteps_adaptiveBatch_tradeoff (hμ : 0 < μ.real Set.univ) (h𝓑 : ∀ s, 0 ≤ 𝓑 s)
    (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) {r : ℝ} (hr : 0 < r) :
    totalSteps μ 𝓑 (adaptiveBatch r 𝓑) / μ.real Set.univ - 1 =
      paretoGamma μ 𝓑 * (totalExamples μ 𝓑 (adaptiveBatch r 𝓑) / (∫ s, 𝓑 s ∂μ) - 1)⁻¹ := by
  rw [totalSteps_adaptiveBatch h𝓑 hi hr, totalExamples_adaptiveBatch h𝓑 hi hr.le, paretoGamma]
  have := Real.sqrt_pos.2 hr
  rcases eq_or_ne (∫ s, √(𝓑 s) ∂μ) 0 with hc | hc
  · simp [hc, hμ.ne']
  · field_simp
    ring

/-- The hypotheses of `totalSteps_adaptiveBatch_tradeoff` are satisfiable: `𝓑 = 1` over one
step, `r = 1`. -/
example := totalSteps_adaptiveBatch_tradeoff (μ := Measure.dirac ()) (𝓑 := 1) (by simp)
  (fun _ => zero_le_one) (integrable_const 1) (by simp) one_pos

/-- **Appendix D**, footnote: "the fairly natural choice `r = E_min/S_min` at which we have
`S_tot/S_min = E_tot/E_min = 1 + √γ`". -/
theorem adaptiveBatch_minRatio (hμ : 0 < μ.real Set.univ) (h𝓑 : ∀ s, 0 ≤ 𝓑 s)
    (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) :
    totalSteps μ 𝓑 (adaptiveBatch ((∫ s, 𝓑 s ∂μ) / μ.real Set.univ) 𝓑) / μ.real Set.univ =
        1 + √(paretoGamma μ 𝓑) ∧
      totalExamples μ 𝓑 (adaptiveBatch ((∫ s, 𝓑 s ∂μ) / μ.real Set.univ) 𝓑) /
        (∫ s, 𝓑 s ∂μ) = 1 + √(paretoGamma μ 𝓑) := by
  have hc : 0 ≤ ∫ s, √(𝓑 s) ∂μ := integral_nonneg fun s => Real.sqrt_nonneg _
  rw [totalSteps_adaptiveBatch h𝓑 hi (div_pos hE hμ),
    totalExamples_adaptiveBatch h𝓑 hi (div_pos hE hμ).le, paretoGamma]
  generalize ∫ s, √(𝓑 s) ∂μ = c at hc ⊢
  generalize ∫ s, 𝓑 s ∂μ = E at hE ⊢
  generalize μ.real Set.univ = S at hμ ⊢
  obtain ⟨a, ha, rfl⟩ : ∃ a, 0 < a ∧ S = a ^ 2 :=
    ⟨√S, Real.sqrt_pos.2 hμ, (Real.sq_sqrt hμ.le).symm⟩
  obtain ⟨b, hb, rfl⟩ : ∃ b, 0 < b ∧ E = b ^ 2 :=
    ⟨√E, Real.sqrt_pos.2 hE, (Real.sq_sqrt hE.le).symm⟩
  rw [show b ^ 2 / a ^ 2 = (b / a) ^ 2 by ring,
    show c ^ 2 / (a ^ 2 * b ^ 2) = (c / (a * b)) ^ 2 by ring, Real.sqrt_sq (by positivity),
    Real.sqrt_sq (by positivity)]
  constructor <;> field_simp

/-- The hypotheses of `adaptiveBatch_minRatio` are satisfiable: `𝓑 = 1` over one step. -/
example := adaptiveBatch_minRatio (μ := Measure.dirac ()) (𝓑 := 1) (by simp)
  (fun _ => zero_le_one) (integrable_const 1) (by simp)

/-- **Appendix D**: `S_tot → S_min` as `r → ∞`, the paper's `B ≫ 𝓑` in eq. (D.3). -/
theorem tendsto_totalSteps_adaptiveBatch (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) :
    Tendsto (fun r => totalSteps μ 𝓑 (adaptiveBatch r 𝓑)) atTop (𝓝 (μ.real Set.univ)) := by
  have h : Tendsto (fun r => μ.real Set.univ + (∫ s, √(𝓑 s) ∂μ) / √r) atTop
      (𝓝 (μ.real Set.univ)) := by
    simpa using tendsto_const_nhds.add (tendsto_const_nhds.div_atTop Real.tendsto_sqrt_atTop)
  exact h.congr' ((eventually_gt_atTop 0).mono fun r hr =>
    (totalSteps_adaptiveBatch h𝓑 hi hr).symm)

/-- **Appendix D**: `E_tot → E_min` as `r → 0`, the paper's `B ≪ 𝓑` in eq. (D.3). -/
theorem tendsto_totalExamples_adaptiveBatch (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) :
    Tendsto (fun r => totalExamples μ 𝓑 (adaptiveBatch r 𝓑)) (𝓝[>] 0)
      (𝓝 (∫ s, 𝓑 s ∂μ)) := by
  have h : Continuous fun r => (∫ s, 𝓑 s ∂μ) + √r * ∫ s, √(𝓑 s) ∂μ := by fun_prop
  refine ((h.tendsto 0).mono_left nhdsWithin_le_nhds).congr' ?_ |>.trans (by simp)
  exact eventually_nhdsWithin_of_forall fun r (hr : 0 < r) =>
    (totalExamples_adaptiveBatch h𝓑 hi hr.le).symm

/-- The hypotheses of `tendsto_totalSteps_adaptiveBatch` and
`tendsto_totalExamples_adaptiveBatch` are satisfiable: `𝓑 = 0`. -/
example := And.intro (tendsto_totalSteps_adaptiveBatch (μ := μ) (𝓑 := 0) (fun _ => le_rfl)
  (integrable_zero _ _ _)) (tendsto_totalExamples_adaptiveBatch (μ := μ) (𝓑 := 0)
  (fun _ => le_rfl) (integrable_zero _ _ _))

end Transformer.NoiseScale
