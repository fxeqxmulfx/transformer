module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound

/-! The transfer theorem. The finite system and the limiting particles share
their N independent initial draws and every minibatch gradient `A.grad t`.
The expected maximum over the updates B ≤ t ≤ H of the supremum over the
open segment of the positive part of the residual gap is at most
`widthError`. The maximum sits inside the expectation, as the recovery
theorem needs one error allowance per run. At a fixed horizon the error
vanishes as the width grows. -/

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

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
