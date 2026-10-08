import Transformer.Modes.Section3_CharacteristicGap

/-!
# Exponential damping at large frequencies

Section 3 `thm:br` of arXiv:2412.09080v3 assumes an integrable norm
power of the characteristic function. Decay of a sufficiently large
actual normalized sum implies decay of the single-summand function.
Together with its proved strict modulus at nonzero frequencies,
continuity and compactness give a common bound `ε < 1` outside any
positive radius. The normalized-sum formula then gives `ε^n` outside
that radius times `sqrt n`, as required in §5.4 `eq:big-z-exp`.

The law and radius are fixed here. Dependence of `ε` on the bandwidth
and translation for a varying `lawY` is retained; an integral error bound
uniform in such laws is not claimed by these statements.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped Topology ENNReal
namespace Transformer.Modes

/-- An integrable characteristic norm power implies decay of the actual
single-summand characteristic function. Source: arXiv:2412.09080v3,
§3 `thm:br` and §5.4, the decay needed to define `ε`. -/
theorem tendsto_characteristic2_zero_of_integrable_power
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ) :
    Tendsto (characteristic2 μ) (cocompact (ℝ × ℝ)) (𝓝 0) := by
  obtain ⟨n, hIn, hn⟩ :=
    ((eventually_integrable_characteristic_scaledSum μ hcf).and (eventually_ge_atTop 1)).exists
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let ν : Measure (ℝ × ℝ) := (Measure.pi fun _ : Fin n => μ).map (scaledSum n)
  have hI : Integrable (charFun (complexLaw ν)) := integrable_charFun_complexLaw ν hIn
  have he : Tendsto Complex.equivRealProdCLM.symm (cocompact (ℝ × ℝ)) (cocompact ℂ) :=
    Complex.equivRealProdCLM.symm.toHomeomorph.toCocompactMap.cocompact_tendsto'
  have hscale : Tendsto (fun w : ℂ => (Real.sqrt n : ℝ) • w) (cocompact ℂ) (cocompact ℂ) := by
    apply Filter.tendsto_cocompact_cocompact_of_norm
    intro ε
    refine ⟨ε / Real.sqrt n, ?_⟩
    intro w hw
    rw [norm_smul, Real.norm_of_nonneg hs.le]
    have h := (div_lt_iff₀ hs).mp hw
    simpa only [mul_comm] using h
  have hpow (ξ : ℝ × ℝ) :
      charFun (complexLaw ν) ((Real.sqrt n : ℝ) • Complex.equivRealProdCLM.symm ξ) =
        characteristic2 μ ξ ^ n := by
    rw [charFun_complexLaw]
    have hp : (((Real.sqrt n : ℝ) • Complex.equivRealProdCLM.symm ξ).re,
        ((Real.sqrt n : ℝ) • Complex.equivRealProdCLM.symm ξ).im) = (Real.sqrt n : ℝ) • ξ := by
      ext <;> simp [Complex.real_smul]
    rw [hp]
    dsimp only [ν]
    rw [characteristic_scaledSum, smul_smul, inv_mul_cancel₀ hs.ne', one_smul]
  have ht := (tendsto_charFun_zero_of_integrable (complexLaw ν) hI).comp (hscale.comp he)
  have htN : Tendsto (fun ξ : ℝ × ℝ => ‖characteristic2 μ ξ‖ ^ n)
      (cocompact (ℝ × ℝ)) (𝓝 (0 : ℝ)) := by
    convert ht.norm using 1
    · funext ξ
      dsimp only [Function.comp_def]
      rw [hpow, norm_pow]
    · simp
  apply Metric.tendsto_nhds.mpr
  intro ε hε
  have hb := htN.eventually_lt_const (pow_pos hε n)
  filter_upwards [hb] with ξ hξ
  rw [dist_zero_right]
  by_contra h
  have hle := pow_le_pow_left₀ hε.le (not_lt.mp h) n
  linarith

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2⟩

/-- Outside a positive radius, a fixed law satisfying the source's
integrable-power condition has a single modulus bound below one.
Source: arXiv:2412.09080v3, §5.4, the strict bound defining `ε`. -/
theorem characteristic2_gap_away_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ ε : ℝ, 0 < ε ∧ ε < 1 ∧ ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε := by
  have htN : Tendsto (fun ξ => ‖characteristic2 μ ξ‖)
      (cocompact (ℝ × ℝ)) (𝓝 (0 : ℝ)) := by
    simpa only [norm_zero] using (tendsto_characteristic2_zero_of_integrable_power μ hcf).norm
  have ht := htN.eventually_lt_const (by norm_num : (0 : ℝ) < 1 / 2)
  obtain ⟨R, hR⟩ := Metric.closedBall_compl_subset_of_mem_cocompact ht (0 : ℝ × ℝ)
  let S : Set (ℝ × ℝ) := {ξ | a ≤ ‖ξ‖} ∩ Metric.closedBall 0 (max R a)
  have hS : IsCompact S :=
    (isCompact_closedBall (0 : ℝ × ℝ) (max R a)).inter_left (isClosed_le continuous_const continuous_norm)
  have hab : ‖((a, 0) : ℝ × ℝ)‖ = a := by
    simp [Prod.norm_def, Real.norm_eq_abs, abs_of_pos ha, ha.le]
  have hne : S.Nonempty := ⟨(a, 0), by
    simp only [S, Set.mem_inter_iff, Set.mem_ofPred_eq, Metric.mem_closedBall, dist_zero_right, hab]
    exact ⟨le_rfl, le_max_right _ _⟩⟩
  obtain ⟨η, hη, hmax⟩ := hS.exists_isMaxOn hne (continuous_characteristic2 μ).norm.continuousOn
  have hηpos : 0 < ‖η‖ := ha.trans_le hη.1
  have hηlt : ‖characteristic2 μ η‖ < 1 := norm_characteristic2_lt_one μ hcf (norm_pos_iff.mp hηpos)
  refine ⟨max ‖characteristic2 μ η‖ (1 / 2), lt_max_of_lt_right (by norm_num),
    max_lt hηlt (by norm_num), ?_⟩
  intro ξ hξ
  by_cases hsmall : ‖ξ‖ ≤ max R a
  · have hmem : ξ ∈ S := ⟨hξ, by simpa only [Metric.mem_closedBall, dist_zero_right] using hsmall⟩
    exact (hmax hmem).trans (le_max_left _ _)
  · have hlarge : R < ‖ξ‖ := (le_max_left R a).trans_lt (not_le.mp hsmall)
    have hmem : ξ ∈ (Metric.closedBall (0 : ℝ × ℝ) R)ᶜ := by
      simpa only [Set.mem_compl_iff, Metric.mem_closedBall, dist_zero_right, not_le] using hlarge
    exact (hR hmem).le.trans (le_max_right _ _)

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- The exact normalized-sum formula raises the single-summand modulus
bound to the `n`th power outside `a sqrt n`. Source: arXiv:2412.09080v3,
§5.4 `eq:big-z-exp`, with the actual frequency normalization retained. -/
theorem norm_characteristic_scaledSum_le_of_gap
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {a ε : ℝ}
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε)
    {n : ℕ} (hn : 1 ≤ n) (ξ : ℝ × ℝ) (hξ : a * Real.sqrt n ≤ ‖ξ‖) :
    ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ ≤ ε ^ n := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hη : a ≤ ‖(Real.sqrt (n : ℝ))⁻¹ • ξ‖ := by
    rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hs.le)]
    exact (le_inv_mul_iff₀ hs).mpr (by simpa only [mul_comm] using hξ)
  rw [characteristic_scaledSum, norm_pow]
  exact pow_le_pow_left₀ (norm_nonneg _) (hgap _ hη) n

example : ∃ ε : ℝ, (∀ ξ : ℝ × ℝ, 1 ≤ ‖ξ‖ → ‖characteristic2 stdGauss2 ξ‖ ≤ ε) ∧
    1 ≤ (1 : ℕ) ∧ 1 * Real.sqrt (1 : ℕ) ≤ ‖((1, 0) : ℝ × ℝ)‖ := by
  obtain ⟨ε, _, _, hgap⟩ := characteristic2_gap_away_zero stdGauss2
    hasIntegrableCharFun_stdGauss2 (a := 1) one_pos
  exact ⟨ε, hgap, le_rfl, by norm_num⟩

/-- For a fixed law with an integrable characteristic norm power, the
actual normalized sums have exponential damping outside any positive
frequency radius times `sqrt n`. Source: arXiv:2412.09080v3, §3 `thm:br`
and §5.4 `eq:big-z-exp`; dependence of `ε` on the law is retained. -/
theorem exists_characteristic_scaledSum_large_frequency_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ ε : ℝ, 0 < ε ∧ ε < 1 ∧ ∀ n : ℕ, 1 ≤ n → ∀ ξ : ℝ × ℝ,
      a * Real.sqrt n ≤ ‖ξ‖ →
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ ≤ ε ^ n := by
  obtain ⟨ε, hε0, hε1, hgap⟩ := characteristic2_gap_away_zero μ hcf ha
  exact ⟨ε, hε0, hε1, fun n hn ξ hξ => norm_characteristic_scaledSum_le_of_gap μ hgap hn ξ hξ⟩

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
