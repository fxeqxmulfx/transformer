module

public import Transformer.Grokking.FlatSpectrum.Training.LeastNorm
public import Transformer.Grokking.FlatSpectrum.Training.Scope.MarginContinuity
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhaseStep

/-!
# One run with a decreasing weight decay

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE18; the item
"No weight decay" of XE13, last sentence.

The limits of the items XE6, XE9 and XE10 run over trained fits, one for each weight decay.
Here one run of gradient descent passes through the weight decays `λ_0, λ_1, …`, any positive
sequence that tends to `0`: phase `j` makes `T j` steps with the weight decay `λ_j` from the
factors at which phase `j − 1` ended, with the step that XE3 admits for the ball of those
factors (`phaseRun`, with the step `phaseStep` and the product `phaseProduct`). Only the
lengths `T j` of the phases are left free.

On samples that some product classifies with a positive margin, from stacked factors of full
row rank, there are lengths `T j` for which, at the phase ends, the nuclear norm of the product
tends to infinity and its margin for the nuclear norm tends to the largest margin `μ*` of the
unit ball (`exists_tendsto_margin_div_phaseProduct`).

* A phase from factors of full row rank converges to a trained fit for its weight decay and
  keeps full row rank (`exists_tendsto_decayRun_phaseStep`).
* The nuclear norm is continuous, and the margin for the nuclear norm is continuous at a
  trained fit with a nonzero product (`MarginContinuity`). So phase `j` is stopped where the
  norm is within `1` and the margin for the norm within `λ_j` of their values at the trained
  fit. There the norms tend to infinity and the margins for the norm to `μ*` (XE6), so they
  do at the phase ends.
* The phases are chosen one after another (`exists_seq_of_forall_exists`).

The lengths of the phases are not explicit: they come from the convergence of each phase.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C}
  {X₀ : Matrix m n ℂ} {a : ℝ} {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}

/-- **The run in phases** (`review/flat_spectrum.md`, item XE18): phase `j` makes `T j` steps of
gradient descent on the cross-entropy with the weight decay `lam j`, from the factors at which
phase `j − 1` ended and with the step `phaseStep` of XE3 for the ball of those factors.
`phaseRun A y a lam T U₀ V₀ j` is the pair of factors at the start of phase `j`. -/
noncomputable def phaseRun (A : S → C → Matrix m n ℂ) (y : S → C) (a : ℝ) (lam : ℕ → ℝ)
    (T : ℕ → ℕ) (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) : ℕ → Matrix m k ℂ × Matrix n k ℂ
  | 0 => (U₀, V₀)
  | j + 1 => decayRun (ceGrad A y) (phaseStep S C a (lam j) (phaseRun A y a lam T U₀ V₀ j))
      (lam j) (phaseRun A y a lam T U₀ V₀ j).1 (phaseRun A y a lam T U₀ V₀ j).2 (T j)

/-- The product `U Vᴴ` of the factors at the start of phase `j` of the run in phases. -/
noncomputable def phaseProduct (A : S → C → Matrix m n ℂ) (y : S → C) (a : ℝ) (lam : ℕ → ℝ)
    (T : ℕ → ℕ) (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) (j : ℕ) : Matrix m n ℂ :=
  (phaseRun A y a lam T U₀ V₀ j).1 * (phaseRun A y a lam T U₀ V₀ j).2ᴴ

/-- **One run with a decreasing weight decay attains the largest margin**
(`review/flat_spectrum.md`, item XE18). Let some product have a positive margin, let the
probes have norm at most `a`, and let `λ_j > 0` tend to `0`. From stacked factors of full row
rank there are lengths `T j` such that the run that makes `T j` steps with the weight decay
`λ_j` and the step of XE3, one phase after another, has at the phase ends `‖X_j‖_S1 → ∞` and
`μ(X_j) / ‖X_j‖_S1 → μ* = max {μ(X) : ‖X‖_S1 ≤ 1}`. -/
theorem exists_tendsto_margin_div_phaseProduct [Nonempty S] [Nontrivial C] (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hsep : 0 < margin A y X₀) {lam : ℕ → ℝ}
    (hpos : ∀ j, 0 < lam j) (hlam : Tendsto lam atTop (𝓝 0))
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ T : ℕ → ℕ,
      Tendsto (fun j => nuclearNorm (phaseProduct A y a lam T U₀ V₀ j)) atTop atTop ∧
      Tendsto (fun j => margin A y (phaseProduct A y a lam T U₀ V₀ j) /
        nuclearNorm (phaseProduct A y a lam T U₀ V₀ j)) atTop (𝓝 (maxMargin A y)) := by
  obtain ⟨F, hF⟩ : ∃ F : Matrix m k ℂ × Matrix n k ℂ → ℝ,
      ∀ W, F W = margin A y (W.1 * W.2ᴴ) / nuclearNorm (W.1 * W.2ᴴ) := ⟨_, fun _ => rfl⟩
  obtain ⟨N, hN, hNc⟩ : ∃ N : Matrix m k ℂ × Matrix n k ℂ → ℝ,
      (∀ W, N W = nuclearNorm (W.1 * W.2ᴴ)) ∧ Continuous N :=
    ⟨_, fun _ => rfl, continuous_nuclearNorm.comp continuous_mul_conjTranspose⟩
  obtain ⟨W, hW0, hW⟩ := exists_seq_of_forall_exists
    (P := fun W : Matrix m k ℂ × Matrix n k ℂ => Function.Surjective (stack W.1 W.2).mulVec)
    (R := fun (j : ℕ) (W W' : Matrix m k ℂ × Matrix n k ℂ) =>
      ∃ (T : ℕ) (Us : Matrix m k ℂ) (Vs : Matrix n k ℂ),
        W' = decayRun (ceGrad A y) (phaseStep S C a (lam j) W) (lam j) W.1 W.2 T ∧
        IsDecayedMin (ceLoss A y) (lam j) Us Vs ∧ |N W' - N (Us, Vs)| < 1 ∧
        (nuclearNorm (Us * Vsᴴ) ≠ 0 → |F W' - F (Us, Vs)| < lam j))
    (x₀ := (U₀, V₀)) hfull (fun j W hP => by
      obtain ⟨Us, Vs, hlim, hmin, hrank⟩ :=
        exists_tendsto_decayRun_phaseStep (A := A) (y := y) ha0 ha (hpos j) hP
      obtain ⟨R, hR⟩ : ∃ R : ℕ → Matrix m k ℂ × Matrix n k ℂ,
          R = decayRun (ceGrad A y) (phaseStep S C a (lam j) W) (lam j) W.1 W.2 := ⟨_, rfl⟩
      rw [← hR] at hlim hrank
      have h1 : ∀ᶠ t in atTop, |N (R t) - N (Us, Vs)| < 1 := by
        have h := Metric.tendsto_nhds.1 ((hNc.tendsto (Us, Vs)).comp hlim) 1 one_pos
        simpa [Real.dist_eq] using h
      by_cases h0 : nuclearNorm (Us * Vsᴴ) = 0
      · obtain ⟨T, hT⟩ := h1.exists
        exact ⟨R T, ⟨T, Us, Vs, by rw [hR], hmin, hT, fun h => absurd h0 h⟩, hrank T⟩
      · have hc : ContinuousAt F (Us, Vs) := by
          rw [funext hF]
          exact ContinuousAt.comp (g := fun X => margin A y X / nuclearNorm X)
            (f := fun W : Matrix m k ℂ × Matrix n k ℂ => W.1 * W.2ᴴ) (x := (Us, Vs))
            (continuousAt_margin_div_nuclearNorm A y h0)
            continuous_mul_conjTranspose.continuousAt
        have h2 : ∀ᶠ t in atTop, |F (R t) - F (Us, Vs)| < lam j := by
          have h := Metric.tendsto_nhds.1 (hc.tendsto.comp hlim) (lam j) (hpos j)
          simpa [Real.dist_eq] using h
        obtain ⟨T, hT1, hT2⟩ := (h1.and h2).exists
        exact ⟨R T, ⟨T, Us, Vs, by rw [hR], hmin, hT1, fun _ => hT2⟩, hrank T⟩)
  choose T Us Vs hrun hmin hnorm hclose using hW
  have hWj : ∀ j, phaseRun A y a lam T U₀ V₀ j = W j := fun j => by
    induction j with
    | zero => exact hW0.symm
    | succ j ih => rw [phaseRun, ih, ← hrun j]
  refine ⟨T, ?_, ?_⟩
  · obtain ⟨Z, hZ1, hZ2⟩ := exists_margin_eq_maxMargin A y
    have hgrow := tendsto_nuclearNorm_atTop hZ1 (maxMargin_pos hsep)
      (fun s c hc => hZ2.ge.trans (margin_le_gap Z hc)) hpos hlam hmin
    have h1 : Tendsto (fun j => N (W (j + 1))) atTop atTop :=
      tendsto_atTop_mono (fun j => by
        have h := (abs_lt.1 (hnorm j)).1
        rw [hN (Us j, Vs j)] at h
        linarith) (tendsto_atTop_add_const_right atTop (-1) hgrow)
    refine ((tendsto_add_atTop_iff_nat (f := fun j => N (W j)) 1).1 h1).congr fun j => ?_
    rw [hN, phaseProduct, hWj]
  · have hlimF : Tendsto (fun j => F (Us j, Vs j)) atTop (𝓝 (maxMargin A y)) := by
      simp only [hF]
      exact tendsto_margin_div_nuclearNorm hsep hpos hlam hmin
    have hne : ∀ᶠ j in atTop, nuclearNorm (Us j * (Vs j)ᴴ) ≠ 0 := by
      filter_upwards [hlimF.eventually (lt_mem_nhds (maxMargin_pos hsep))] with j hj h0
      simp [hF, h0] at hj
    have hdist : Tendsto (fun j => dist (F (Us j, Vs j)) (F (W (j + 1)))) atTop (𝓝 0) := by
      refine squeeze_zero' (Eventually.of_forall fun _ => dist_nonneg) ?_ hlam
      filter_upwards [hne] with j hj
      rw [dist_comm, Real.dist_eq]
      exact (hclose j hj).le
    have h1 : Tendsto (fun j => F (W (j + 1))) atTop (𝓝 (maxMargin A y)) :=
      hlimF.congr_dist hdist
    refine ((tendsto_add_atTop_iff_nat (f := fun j => F (W j)) 1).1 h1).congr fun j => ?_
    rw [hF, phaseProduct, hWj]

/-! The witness: one sample of a `1 × 1` product read with the code `±1`, where the product
`1` has the margin `2`, from the start `U₀ = [1 0]`, `V₀ = [0 1]` of full row rank, along the
weight decays `1 / (j + 1)`. -/

example := exists_tendsto_margin_div_phaseProduct (y := fun _ => true) zero_le_one
  frobeniusSq_signProbe_le margin_signProbe_one_pos (lam := fun j => 1 / ((j : ℝ) + 1))
  (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
  (U₀ := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (V₀ := fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

/-- The first phase of the witness starts at the factors given. -/
example : phaseRun signProbe (fun _ => true) 1 (fun j => 1 / ((j : ℝ) + 1)) (fun _ => 1)
    (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
    (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ)) 0 =
    (fromCols 1 0, fromCols 0 1) := rfl

end Transformer.Grokking.FlatSpectrum.Training.Scope
