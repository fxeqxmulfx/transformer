module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound

/-! The transfer theorem and its handoff to the recovery theorem. The
finite system and the limiting particles share their N independent
initial draws and every minibatch gradient `A.grad t`. The expected
maximum over the updates B ≤ t ≤ H of the supremum over the open segment
of the positive part of the residual gap is at most `widthError`. The
maximum sits inside the expectation, as the recovery theorem needs one
error allowance per run. At a fixed horizon the error vanishes as the
width grows. A bound r̄ on the limiting residual over B ≤ t ≤ H, together
with an allowance for the updates after H, yields an excess that is
measurable, integrable and nonnegative, bounds the finite residual at
every update t ≥ B and every fraction, and has expectation at most
max(r̄, 0) + widthError + E[tail]: the hypotheses on the excess in
`permanent_of_observed_drift`. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

variable {P K J : Type*}

/-- The transfer theorem: E[max_{B ≤ t ≤ H} sup_{0<u<1} (R_t(u) - R̄_t(u))₊] ≤ widthError. -/
theorem expected_residual_gap_le [Fintype P] [Fintype K] [Fintype J] (A : Training P K)
    (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ}
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) (B H : ℕ) {n : ℕ} (hn : 0 < n) :
    ∫⁻ z, ⨆ t ∈ Icc B H, ⨆ u ∈ Set.Ioo (0:ℝ) 1,
        ENNReal.ofReal (max (residual A Γ χ z t u - limitResidual A μ Γ χ t u) 0)
        ∂(Measure.pi fun _ : Fin N => μ) ≤
      ENNReal.ofReal (widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n) := by
  obtain ⟨hi, hle⟩ := integral_majorant_le A μ h hN B H n
  have hpt (z : Fin N → P → ℝ) : (⨆ t ∈ Icc B H, ⨆ u ∈ Set.Ioo (0:ℝ) 1,
      ENNReal.ofReal (max (residual A Γ χ z t u - limitResidual A μ Γ χ t u) 0)) ≤
        ENNReal.ofReal (majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n) := by
    refine iSup₂_le fun t ht => iSup₂_le fun u hu => ?_
    exact ENNReal.ofReal_le_ofReal
      (max_le (residual_sub_le_majorant A μ h hN z hn ht hu) (majorant_nonneg A μ h z B H n))
  calc _ ≤ ∫⁻ z, ENNReal.ofReal (majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n)
          ∂(Measure.pi fun _ : Fin N => μ) := lintegral_mono hpt
    _ = ENNReal.ofReal (∫ z, majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n ∂(Measure.pi fun _ : Fin N => μ)) :=
        (ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ fun z => majorant_nonneg A μ h z B H n)).symm
    _ ≤ _ := ENNReal.ofReal_le_ofReal hle

/-- At a fixed horizon the error vanishes in width: first choose the grid,
then the width. -/
theorem widthError_vanishes {eps η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ} (k j B H : ℕ) {δ : ℝ} (hδ : 0 < δ) :
    ∃ n > 0, ∃ N₀ : ℕ, ∀ N ≥ N₀, widthError eps η LG Lφ Cφ LΓ Lχ Mχ Cχ k j N B H n < δ := by
  obtain ⟨n, hn, hb⟩ : ∃ n : ℕ, 0 < n ∧ LΓ * (2 * Mχ / n) < δ / 2 := by
    refine ⟨⌈4 * |LΓ * Mχ| / δ⌉₊ + 1, Nat.succ_pos _, ?_⟩
    have hm : 4 * |LΓ * Mχ| / δ < ((⌈4 * |LΓ * Mχ| / δ⌉₊ + 1 : ℕ) : ℝ) := by
      push_cast
      linarith [Nat.le_ceil (4 * |LΓ * Mχ| / δ)]
    have hm0 : (0:ℝ) < ((⌈4 * |LΓ * Mχ| / δ⌉₊ + 1 : ℕ) : ℝ) := by positivity
    rw [show LΓ * (2 * Mχ / ((⌈4 * |LΓ * Mχ| / δ⌉₊ + 1 : ℕ) : ℝ)) =
      2 * (LΓ * Mχ) / ((⌈4 * |LΓ * Mχ| / δ⌉₊ + 1 : ℕ) : ℝ) by ring, div_lt_iff₀ hm0]
    have := (div_lt_iff₀ hδ).1 hm
    linarith [le_abs_self (LΓ * Mχ)]
  set a := LΓ * (2 * Lχ * (growth eps η LG Lφ ^ (H+1) * forcing eps η LG * ((H+1) * (k * Cφ))) +
    (H + 1 - B : ℕ) * ((n+1) * (j * Cχ)))
  refine ⟨n, hn, ⌈(2 * |a| / δ)^2⌉₊ + 1, fun N hN => ?_⟩
  have hN' : (2 * |a| / δ)^2 < N := by
    have h1 := Nat.le_ceil ((2 * |a| / δ)^2)
    have h2 : ((⌈(2 * |a| / δ)^2⌉₊ + 1 : ℕ) : ℝ) ≤ N := by exact_mod_cast hN
    push_cast at h2
    linarith
  have hs : 2 * |a| / δ < Real.sqrt N := (Real.lt_sqrt (by positivity)).2 hN'
  have hs0 : 0 < Real.sqrt N := lt_of_le_of_lt (by positivity) hs
  have e : widthError eps η LG Lφ Cφ LΓ Lχ Mχ Cχ k j N B H n = a / Real.sqrt N + LΓ * (2 * Mχ / n) := by
    unfold widthError
    ring
  have ha : a / Real.sqrt N < δ / 2 := by
    rw [div_lt_iff₀ hs0]
    have := (div_lt_iff₀ hδ).1 hs
    linarith [le_abs_self a]
  rw [e]
  linarith

theorem continuous_l1_meanField_sub {α : Type*} [TopologicalSpace α] [Fintype J] {N : ℕ}
    {Ξ : α → J → ℝ} (hΞ : ∀ j, Continuous fun x => Ξ x j) (c : J → ℝ) :
    Continuous fun z : Fin N → α => l1 (meanField Ξ z - c) := by
  unfold l1
  refine continuous_finsetSum _ fun j _ => ?_
  change Continuous fun z : Fin N → α => |(∑ i, Ξ (z i) j) / N - c j|
  exact (((continuous_finsetSum _ fun i _ => (hΞ j).comp (continuous_apply i)).div_const _).sub
    continuous_const).abs

section Measurable

variable [Fintype P] [Fintype K] [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
  [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ}
  {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}

omit [IsProbabilityMeasure μ] in
theorem continuous_fluctuation (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N s : ℕ) :
    Continuous fun z : Fin N → P → ℝ => fluctuation A μ z s := by
  have hφ := continuous_of_l1 h.Lφ_nonneg h.field
  have hw := (continuous_flow A μ h.eps_pos h.continuous_grad s).1
  exact continuous_l1_meanField_sub (Ξ := fun x => A.field (flow A μ s x).weights)
    (fun k => (continuous_apply k).comp (hφ.comp hw)) _

theorem continuous_segmentError (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N t : ℕ) (u : ℝ) :
    Continuous fun z : Fin N → P → ℝ => segmentError A μ χ z t u :=
  continuous_l1_meanField_sub (fun j => (integrable_features A μ h t u j).1) _

theorem measurable_majorant (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N B H n : ℕ) :
    Measurable fun z : Fin N → P → ℝ => majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n := by
  refine Continuous.measurable ?_
  unfold majorant
  have h1 := continuous_finsetSum (range (H+1)) fun s _ => continuous_fluctuation A μ h N s
  have h2 := continuous_finsetSum (Icc B H) fun t _ =>
    continuous_finsetSum (range (n+1)) fun k _ => continuous_segmentError A μ h N t ((k:ℝ)/n)
  exact continuous_const.mul (((continuous_const.mul (continuous_const.mul h1)).add h2).add
    continuous_const)

/-- The excess handed to the recovery theorem: the limiting bound, the
width majorant over B ≤ t ≤ H, and the allowance for the tail. -/
theorem drift_excess (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N)
    {B H n : ℕ} (hn : 0 < n) {r : ℝ}
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ Γ χ t u ≤ r)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A Γ χ z t u ≤ tail z) :
    Measurable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z) ∧
      Integrable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z)
        (Measure.pi fun _ : Fin N => μ) ∧
      (∀ z, 0 ≤ max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z) ∧
      ∫ z, (max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z)
          ∂(Measure.pi fun _ : Fin N => μ) ≤
        max r 0 + widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n +
          ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) ∧
      ∀ z t, B ≤ t → ∀ u ∈ Set.Ioo (0:ℝ) 1,
        residual A Γ χ z t u ≤ max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z := by
  obtain ⟨hi, hle⟩ := integral_majorant_le A μ h hN B H n
  have hm := measurable_majorant A μ h N B H n
  have hc : Integrable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n)
      (Measure.pi fun _ : Fin N => μ) := (integrable_const _).add hi
  refine ⟨(measurable_const.add hm).add htm, hc.add hti, fun z => ?_, ?_, fun z t ht u hu => ?_⟩
  · have := majorant_nonneg A μ h z B H n
    have := htn z
    have := le_max_right r 0
    linarith
  · rw [integral_add hc hti, integral_add (integrable_const _) hi, integral_const]
    simp only [probReal_univ, smul_eq_mul, one_mul]
    linarith
  · have h0 := majorant_nonneg A μ h z B H n
    have hr := le_max_left r 0
    have hr0 := le_max_right r 0
    rcases le_or_gt t H with htH | htH
    · have hmem : t ∈ Icc B H := mem_Icc.2 ⟨ht, htH⟩
      have := residual_sub_le_majorant A μ h hN z hn hmem hu
      have := hlim t hmem u hu
      have := htn z
      linarith
    · have := htail z t htH u hu
      linarith

end Measurable

section Examples

example := integral_l1_meanField_sub_le (Measure.dirac (0:ℝ)) (N := 2) two_pos
  (Ξ := fun _ (_ : Fin 1) => (0:ℝ)) (fun _ => measurable_const) (C := 0) fun _ _ => by simp
example := identityRegular.gronwall_nonneg identityTraining 3
example := residual_sub_le_majorant identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (fun _ => 0) (B := 1) (H := 3) (n := 4) (by norm_num) (t := 2) (by simp) (u := 1/2)
  ⟨by norm_num, by norm_num⟩
example := majorant_nonneg identityTraining (Measure.dirac 0) identityRegular (N := 2) (fun _ => 0) 1 3 4
example := integral_majorant_le identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos 1 3 4
example := expected_residual_gap_le identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  1 3 (n := 4) (by norm_num)
example := widthError_vanishes (eps := 1) (η := 1) (LG := 1) (Lφ := 0) (Cφ := 1) (LΓ := 1) (Lχ := 1)
  (Mχ := 1) (Cχ := 1) 1 1 0 3 one_pos

example := continuous_l1_meanField_sub (N := 2) (Ξ := fun (x : ℝ) (_ : Fin 1) => x)
  (fun _ => continuous_id) 0
example := continuous_fluctuation identityTraining (Measure.dirac 0) identityRegular 2 3
example := continuous_segmentError identityTraining (Measure.dirac 0) identityRegular 2 3 (1/2)
example := measurable_majorant identityTraining (Measure.dirac 0) identityRegular 2 1 3 4
example := drift_excess identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (B := 1) (H := 3) (n := 4) (by norm_num) (r := 0) (fun _ _ _ _ => by simp [limitResidual])
  (fun _ => 0) measurable_const (integrable_const _) (fun _ => le_rfl)
  (fun _ _ _ _ _ => by simp [residual])

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
