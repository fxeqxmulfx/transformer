module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun

/-!
# The mean of the losses: a rescaling of the step and of the weight decay

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE3, the mean.
The models of the note are trained on the mean of the losses of the samples, `L / N`, with the
gradient `G / N`. For a real `c ≠ 0`:

* a step of item WD1 on `c G` with the step `η` and the weight decay `λ` is a step on `G` with
  the step `c η` and the weight decay `λ / c` (`decayStep_smul`), and so is the run
  (`decayRun_smul`);
* the weight-decayed loss of `c L` at `λ` is `c` times that of `L` at `λ / c`
  (`decayedLoss_const_mul`), so for `c > 0` the decayed minima are the same
  (`isDecayedMin_const_mul_iff`).

With `c = 1 / N` the bounds of item XE3 become those of one sample, `4 log |C| ≤ λ ρ`,
`2 η (λ² + (2 a)²) ≤ λ` and `η (λ + 2 a + 2 a² ρ) ≤ 1`: the run on the mean cross-entropy
converges under bounds that do not depend on the number of samples
(`exists_tendsto_decayRun_mean_ceLoss_isDecayedMin`).

The witness is the sample of item XE3 taken twice.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k]
  {G : Matrix m n ℂ → Matrix m n ℂ} {L : Matrix m n ℂ → ℝ} {c η lam : ℝ}

/-- **A step on `c G` is a step on `G` with the step `c η` and the weight decay `λ / c`**
(`review/flat_spectrum.md`, item XE3, the mean). -/
theorem decayStep_smul (hc : c ≠ 0) (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (UV : Matrix m k ℂ × Matrix n k ℂ) :
    decayStep (fun X => c • G X) η lam UV = decayStep G (c * η) (lam / c) UV := by
  have e : c * η * (lam / c) = η * lam := by field_simp
  refine Prod.ext ?_ ?_
  · simp only [decayStep, Matrix.smul_mul, smul_add, smul_smul, e, mul_comm η c]
  · simp only [decayStep, conjTranspose_smul, star_trivial, Matrix.smul_mul, smul_add,
      smul_smul, e, mul_comm η c]

/-- The run on `c G` is the run on `G` with the step `c η` and the weight decay `λ / c`. -/
theorem decayRun_smul (hc : c ≠ 0) (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) :
    decayRun (fun X => c • G X) η lam U₀ V₀ = decayRun G (c * η) (lam / c) U₀ V₀ := by
  funext t
  rw [decayRun, decayRun, funext (decayStep_smul hc G η lam)]

/-- The weight-decayed loss of `c L` at `λ` is `c` times that of `L` at `λ / c`. -/
lemma decayedLoss_const_mul (hc : c ≠ 0) (L : Matrix m n ℂ → ℝ) (lam : ℝ) (U : Matrix m k ℂ)
    (V : Matrix n k ℂ) :
    decayedLoss (fun X => c * L X) lam U V = c * decayedLoss L (lam / c) U V := by
  unfold decayedLoss
  field_simp

/-- **The decayed minima of `c L` at `λ` are those of `L` at `λ / c`**
(`review/flat_spectrum.md`, item XE3, the mean), for `c > 0`. -/
theorem isDecayedMin_const_mul_iff (hc : 0 < c) {U : Matrix m k ℂ} {V : Matrix n k ℂ} :
    IsDecayedMin (fun X => c * L X) lam U V ↔ IsDecayedMin L (lam / c) U V := by
  unfold IsDecayedMin
  simp only [decayedLoss_const_mul hc.ne', mul_le_mul_iff_right₀ hc]

variable [Fintype S] [Fintype C] [DecidableEq m] [DecidableEq n] {A : S → C → Matrix m n ℂ}
  {y : S → C} {a ρ : ℝ}

/-- **The run on the mean cross-entropy converges, under bounds without the number of
samples** (`review/flat_spectrum.md`, item XE3, the mean): for probes of norm at most `a`, if
`4 log |C| ≤ λ ρ`, `2 η (λ² + (2 a)²) ≤ λ` and `η (λ + 2 a + 2 a² ρ) ≤ 1`, the run from a
start of full row rank in the ball converges to a decayed minimum of the mean. -/
theorem exists_tendsto_decayRun_mean_ceLoss_isDecayedMin [Nonempty S] (hη : 0 < η)
    (hlam : 0 < lam) (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hρ : 4 * Real.log (Fintype.card C) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * a) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * a + 2 * a ^ 2 * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (fun X => (Fintype.card S : ℝ)⁻¹ • ceGrad A y X) η lam U₀ V₀) atTop
          (𝓝 (U, V)) ∧
        IsDecayedMin (fun X => (Fintype.card S : ℝ)⁻¹ * ceLoss A y X) lam U V := by
  have hN : (0 : ℝ) < Fintype.card S := Nat.cast_pos.2 Fintype.card_pos
  have e : lam / (Fintype.card S : ℝ)⁻¹ = Fintype.card S * lam := by
    rw [div_inv_eq_mul, mul_comm]
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin (A := A) (y := y)
    (η := (Fintype.card S : ℝ)⁻¹ * η) (lam := lam / (Fintype.card S : ℝ)⁻¹) (ρ := ρ)
    (mul_pos (inv_pos.2 hN) hη) (div_pos hlam (inv_pos.2 hN)) ha0 ha
    (by rw [e]; nlinarith [mul_le_mul_of_nonneg_left hρ hN.le])
    (by
      rw [e, show 2 * ((Fintype.card S : ℝ)⁻¹ * η) * ((Fintype.card S * lam) ^ 2
          + (2 * a * Fintype.card S) ^ 2)
          = Fintype.card S * (2 * η * (lam ^ 2 + (2 * a) ^ 2)) by field_simp]
      exact mul_le_mul_of_nonneg_left hstep hN.le)
    (by
      rw [e, show (Fintype.card S : ℝ)⁻¹ * η * (Fintype.card S * lam + 2 * a * Fintype.card S
          + 2 * (Fintype.card S * a ^ 2) * ρ) = η * (lam + 2 * a + 2 * a ^ 2 * ρ) by
        field_simp]
      exact hdesc) hw hfull
  exact ⟨U, V, by rwa [decayRun_smul (inv_pos.2 hN).ne'],
    (isDecayedMin_const_mul_iff (inv_pos.2 hN)).2 hmin⟩

/-! The witness: the sample of item XE3 taken twice, `S = Fin 2`, with `a = 1`, `λ = 1`,
`ρ = 3` and `η = 1/10`. -/

example := decayStep_smul (c := 2) two_ne_zero (fun X : Matrix (Fin 1) (Fin 1) ℂ => X) 1 1
  ((1 : Matrix (Fin 1) (Fin 1) ℂ), (1 : Matrix (Fin 1) (Fin 1) ℂ))

example := decayRun_smul (c := 2) two_ne_zero (fun X : Matrix (Fin 1) (Fin 1) ℂ => X) 1 1
  (1 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ)

example := decayedLoss_const_mul (c := 2) two_ne_zero
  (fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing X X) 1 (1 : Matrix (Fin 1) (Fin 1) ℂ) 1

/-- The trained fits of twice the cross-entropy at `λ = 2` are those of the cross-entropy at
`λ = 1`. -/
example {U V : Matrix (Fin 1) (Fin 1) ℂ} :=
  isDecayedMin_const_mul_iff (c := 2) (L := ceLoss signProbe fun _ => true) (lam := 2)
    (U := U) (V := V) two_pos

/-- The run on the mean of two copies of the sample of XE3 converges to a trained fit. -/
example := exists_tendsto_decayRun_mean_ceLoss_isDecayedMin
  (A := fun _ : Fin 2 => signProbe 0) (y := fun _ => true) (η := 1 / 10) (lam := 1) (ρ := 3)
  (by norm_num) one_pos zero_le_one (fun _ c => frobeniusSq_signProbe_le 0 c)
  (by
    simp only [Fintype.card_bool, Nat.cast_ofNat]
    linarith [Real.log_two_lt_d9])
  (by norm_num) (by norm_num) (stepWitness_ball.trans (by norm_num))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training
