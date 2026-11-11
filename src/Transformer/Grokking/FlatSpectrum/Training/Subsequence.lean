module

public import Transformer.Grokking.FlatSpectrum.Training.AlmostEvery
public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun

/-!
# Trained fits have a subsequence whose factors converge

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD7,
statement 1.
Statement 1 of item WD7 is about trained fits that converge. They always do along a
subsequence:

* factors with a bounded weight decay `‖U_j‖² + ‖V_j‖² ≤ b` have entries bounded by `√b`, so a
  subsequence of them converges (`exists_subseq_tendsto_of_frobeniusSq_le`);
* a trained fit of the squared loss on `Ω` has `½ (‖U‖² + ‖V‖²) = ‖U Vᴴ‖_S1 ≤ ‖K‖_S1`
  (`IsDecayedMin.weightDecay_le_nuclearNorm`), by items WD2 and WD7;
* so every sequence of trained fits of one width with `λ_j → 0⁺` has a subsequence whose
  factors converge, and the limit is an optimal fit (`exists_subseq_tendsto_isOptimalFit`);
* trained fits exist at every width `r ≥ min(|m|, |n|)` (item WD10), so an optimal fit of
  every such width exists (`exists_isOptimalFit`).

The witness is the shrinking family of the `1 × 1` kernel `1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {Ω : Set (m × n)} {K : Matrix m n ℂ} {lam : ℝ}

omit [DecidableEq m] [DecidableEq n] in
/-- **Factors with a bounded weight decay have a convergent subsequence.** -/
theorem exists_subseq_tendsto_of_frobeniusSq_le {Us : ℕ → Matrix m k ℂ}
    {Vs : ℕ → Matrix n k ℂ} {b : ℝ} (hb : ∀ j, frobeniusSq (Us j) + frobeniusSq (Vs j) ≤ b) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ) (φ : ℕ → ℕ), StrictMono φ ∧
      Tendsto (fun j => Us (φ j)) atTop (𝓝 U) ∧ Tendsto (fun j => Vs (φ j)) atTop (𝓝 V) := by
  have : FirstCountableTopology (Matrix m k ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → k → ℂ))
  have : FirstCountableTopology (Matrix n k ℂ) :=
    inferInstanceAs (FirstCountableTopology (n → k → ℂ))
  obtain ⟨UV, -, φ, hφ, hlim⟩ := ((isCompact_setOf_norm_apply_le (m := m) (n := k) (√b)).prod
    (isCompact_setOf_norm_apply_le (m := n) (n := k) (√b))).tendsto_subseq
    (x := fun j => (Us j, Vs j)) fun j =>
      ⟨fun i l => (norm_apply_le_sqrt_frobeniusSq _ i l).trans
        (Real.sqrt_le_sqrt (by linarith [hb j, frobeniusSq_nonneg (Vs j)])),
      fun i l => (norm_apply_le_sqrt_frobeniusSq _ i l).trans
        (Real.sqrt_le_sqrt (by linarith [hb j, frobeniusSq_nonneg (Us j)]))⟩
  exact ⟨UV.1, UV.2, φ, hφ, hlim.fst_nhds, hlim.snd_nhds⟩

omit [DecidableEq m] in
/-- A trained fit of the squared loss has the weight decay `½ (‖U‖² + ‖V‖²) ≤ ‖K‖_S1`. -/
lemma IsDecayedMin.weightDecay_le_nuclearNorm {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (hlam : 0 < lam) (h : IsDecayedMin (fitLoss Ω K) lam U V) :
    (frobeniusSq U + frobeniusSq V) / 2 ≤ nuclearNorm K := by
  have h1 := h.weightDecay_eq_nuclearNorm hlam
  have h2 := h.nuclearNorm_le hlam (X' := K) fun _ _ => rfl
  linarith

omit [DecidableEq m] in
/-- **Trained fits have a subsequence whose factors converge, to an optimal fit**
(`review/flat_spectrum.md`, item WD7, statement 1): for the squared loss on `Ω`, weight decays
`λ_j → 0⁺` and trained fits of one width. Nothing is assumed about their convergence. -/
theorem exists_subseq_tendsto_isOptimalFit {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) {r : ℕ} {Us : ℕ → Matrix m (Fin r) ℂ}
    {Vs : ℕ → Matrix n (Fin r) ℂ}
    (hmin : ∀ j, IsDecayedMin (fitLoss Ω K) (lams j) (Us j) (Vs j)) :
    ∃ (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ) (φ : ℕ → ℕ), StrictMono φ ∧
      Tendsto (fun j => Us (φ j)) atTop (𝓝 U) ∧ Tendsto (fun j => Vs (φ j)) atTop (𝓝 V) ∧
        IsOptimalFit {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} U V := by
  obtain ⟨U, V, φ, hφ, hU, hV⟩ := exists_subseq_tendsto_of_frobeniusSq_le (Us := Us) (Vs := Vs)
    (b := 2 * nuclearNorm K) fun j => by linarith [(hmin j).weightDecay_le_nuclearNorm (hpos j)]
  exact ⟨U, V, φ, hφ, hU, hV, isOptimalFit_of_tendsto (lams := fun j => lams (φ j))
    (fun j => hpos _) (hlam.comp hφ.tendsto_atTop) (fun j => hmin _) hU hV⟩

/-- **An optimal fit of every width `r ≥ min(|m|, |n|)` exists**
(`review/flat_spectrum.md`, items WD7 and WD10): a limit of trained fits of that width. -/
theorem exists_isOptimalFit (Ω : Set (m × n)) (K : Matrix m n ℂ) {r : ℕ}
    (hr : Fintype.card m ≤ r ∨ Fintype.card n ≤ r) :
    ∃ (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
      IsOptimalFit {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} U V := by
  choose Us Vs h using fun j : ℕ => exists_isDecayedMin_of_card_le (k := Fin r)
    (fitLoss_first_order Ω K) (continuous_mask_sub Ω K) (analyticAt_fitLoss Ω K)
    (by positivity : (0 : ℝ) < 1 / ((j : ℝ) + 1)) (fitLoss_nonneg Ω K) (by simpa using hr)
  obtain ⟨U, V, -, -, -, -, hopt⟩ := exists_subseq_tendsto_isOptimalFit
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h
  exact ⟨U, V, hopt⟩

/-! The witness: the shrinking family `(1 − s_j) · (1 − s_j)` of the `1 × 1` kernel `1`, with
`‖U_j‖² + ‖V_j‖² ≤ 2`. -/

example := exists_subseq_tendsto_of_frobeniusSq_le
  (Us := fun j : ℕ => (1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (Vs := fun j : ℕ => (1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (b := 2 * nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ))
  fun j => by linarith [(isDecayedMin_shrink j).weightDecay_le_nuclearNorm (shrink_pos j)]

example := (isDecayedMin_one_sub (s := 1 / 2) (by norm_num)
  (by norm_num)).weightDecay_le_nuclearNorm (by norm_num)

/-- A subsequence of the shrinking family converges to an optimal fit. -/
example := exists_subseq_tendsto_isOptimalFit shrink_pos tendsto_shrink isDecayedMin_shrink

/-- The kernel of `a³ + a b` for `p = 3` with the cell `(0, 0)` seen has an optimal fit of
width `3`. -/
example := exists_isOptimalFit ({((0 : ZMod 3), (0 : ZMod 3))}) (cubicKernel 3) (r := 3)
  (Or.inl (by simp))

end Transformer.Grokking.FlatSpectrum.Training
