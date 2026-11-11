module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogit
public import Transformer.Grokking.FlatSpectrum.Training.StepSize

/-!
# Logits that are quadratic in the product: a ball that the steps contract

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE14.
For the logits `±(Re x)²` of `SquaredLogit` the gradient is small near the zero
factorization: in the ball `‖U‖² + ‖V‖² ≤ ρ` it has `‖G‖ ≤ ρ`
(`frobeniusSq_sqLogitGrad_mul_le`), as `‖G X‖ ≤ 2 |Re x|` and `‖U Vᴴ‖ ≤ ρ/2`. So for
`ρ = λ/4` and a step with `η λ ≤ 1/2` the weight decay wins:

    ‖U'‖² + ‖V'‖² ≤ (1 − η λ / 4) (‖U‖² + ‖V‖²)

after one step (`frobeniusSq_decayStep_sqLogit_le`), and a run from the ball shrinks
geometrically (`frobeniusSq_decayRun_sqLogit_le`). The proof expands `‖(1 − η M) W‖²` as in
WD6, with `⟨G X, X⟩ ≥ −2 (Re x)²` and `‖M W‖² ≤ 2 (λ² + ‖G‖²) ‖W‖²`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {η lam : ℝ}

/-- Matrices whose squared norm tends to `0` tend to `0`. -/
lemma tendsto_zero_of_tendsto_frobeniusSq {α : Type*} {l : Filter α} {A : α → Matrix m n ℂ}
    (h : Tendsto (fun a => frobeniusSq (A a)) l (𝓝 0)) : Tendsto A l (𝓝 0) := by
  refine tendsto_pi_nhds.2 fun i => tendsto_pi_nhds.2 fun j => ?_
  refine squeeze_zero_norm (fun a => norm_apply_le_sqrt_frobeniusSq (A a) i j) ?_
  have h0 := (Real.continuous_sqrt.tendsto 0).comp h
  rw [Real.sqrt_zero] at h0
  exact h0

/-- `(Re x)² ≤ ‖X‖²` for a `1 × 1` matrix. -/
lemma re_sq_le_frobeniusSq (X : Matrix (Fin 1) (Fin 1) ℂ) : (X 0 0).re ^ 2 ≤ frobeniusSq X := by
  simp only [frobeniusSq, Finset.univ_unique, Fin.default_eq_zero, Finset.sum_singleton]
  rw [Complex.sq_norm, Complex.normSq_apply]
  nlinarith [sq_nonneg (X 0 0).im]

variable {U V U₀ V₀ : Matrix (Fin 1) k ℂ}

/-- In the ball `‖U‖² + ‖V‖² ≤ ρ` the gradient has `‖G‖ ≤ ρ`. -/
lemma frobeniusSq_sqLogitGrad_mul_le {ρ : ℝ} (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    frobeniusSq (sqLogitGrad (U * Vᴴ)) ≤ ρ ^ 2 := by
  have h1 := frobeniusSq_sqLogitGrad_le (U * Vᴴ)
  have h2 := re_sq_le_frobeniusSq (U * Vᴴ)
  have h3 := frobeniusSq_mul_conjTranspose_le hw
  nlinarith

/-- **A step contracts the ball `‖U‖² + ‖V‖² ≤ λ/4`** by the factor `1 − η λ / 4`. -/
theorem frobeniusSq_decayStep_sqLogit_le (hη : 0 ≤ η) (hlam : 0 < lam)
    (hstep : η * lam ≤ 1 / 2) (hw : frobeniusSq U + frobeniusSq V ≤ lam / 4) :
    frobeniusSq (decayStep sqLogitGrad η lam (U, V)).1 +
      frobeniusSq (decayStep sqLogitGrad η lam (U, V)).2 ≤
        (1 - η * lam / 4) * (frobeniusSq U + frobeniusSq V) := by
  have hw0 := add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V)
  have hG := frobeniusSq_sqLogitGrad_mul_le (U := U) (V := V) le_rfl
  have h1 := frobeniusSq_lift_mul_stack_le hG lam U V
  have h2 := neg_le_pairing_sqLogitGrad (U * Vᴴ)
  have h3 := re_sq_le_frobeniusSq (U * Vᴴ)
  have h4 := frobeniusSq_mul_conjTranspose_le (U := U) (V := V) le_rfl
  rw [← frobeniusSq_stack, stack_decayStep, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul,
    frobeniusSq_sub, pairing_smul_right, frobeniusSq_smul, pairing_stack_lift, frobeniusSq_stack]
  set w := frobeniusSq U + frobeniusSq V with hwdef
  have hηl := mul_nonneg hη hlam.le
  have a0 : η * (-(4 * pairing (sqLogitGrad (U * Vᴴ)) (U * Vᴴ))) ≤ η * (2 * w ^ 2) :=
    mul_le_mul_of_nonneg_left (by nlinarith) hη
  have a1 : η * w ^ 2 ≤ η * lam / 4 * w := by nlinarith [mul_nonneg hη hw0]
  have a2 : η ^ 2 * lam ^ 2 * w ≤ η * lam / 2 * w := by
    nlinarith [mul_nonneg hηl hw0, mul_nonneg (mul_nonneg hηl hw0) (sub_nonneg.2 hstep)]
  have a3 : η ^ 2 * w ^ 2 * w ≤ η * lam / 32 * w := by
    have b1 : w ^ 2 ≤ (lam / 4) ^ 2 := pow_le_pow_left₀ hw0 hw 2
    have b2 : η ^ 2 * w ^ 2 * w ≤ η ^ 2 * (lam / 4) ^ 2 * w :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left b1 (sq_nonneg η)) hw0
    nlinarith [mul_nonneg hηl hw0, mul_nonneg (mul_nonneg hηl hw0) (sub_nonneg.2 hstep)]
  have a4 := mul_le_mul_of_nonneg_left h1 (sq_nonneg η)
  nlinarith

/-- A run from the ball `‖U₀‖² + ‖V₀‖² ≤ λ/4` shrinks geometrically. -/
theorem frobeniusSq_decayRun_sqLogit_le (hη : 0 ≤ η) (hlam : 0 < lam) (hstep : η * lam ≤ 1 / 2)
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ lam / 4) (t : ℕ) :
    frobeniusSq (decayRun sqLogitGrad η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun sqLogitGrad η lam U₀ V₀ t).2 ≤
        (1 - η * lam / 4) ^ t * (frobeniusSq U₀ + frobeniusSq V₀) := by
  have hw0 := add_nonneg (frobeniusSq_nonneg U₀) (frobeniusSq_nonneg V₀)
  have hq0 : 0 ≤ 1 - η * lam / 4 := by linarith
  have hq1 : 1 - η * lam / 4 ≤ 1 := by linarith [mul_nonneg hη hlam.le]
  induction t with
  | zero => simp [decayRun]
  | succ t ih =>
    have hball := ih.trans ((mul_le_of_le_one_left hw0 (pow_le_one₀ hq0 hq1)).trans hw)
    rw [decayRun, Function.iterate_succ_apply', ← decayRun, pow_succ]
    calc _ ≤ (1 - η * lam / 4) * (frobeniusSq (decayRun sqLogitGrad η lam U₀ V₀ t).1 +
          frobeniusSq (decayRun sqLogitGrad η lam U₀ V₀ t).2) :=
          frobeniusSq_decayStep_sqLogit_le hη hlam hstep hball
      _ ≤ (1 - η * lam / 4) * ((1 - η * lam / 4) ^ t * (frobeniusSq U₀ + frobeniusSq V₀)) :=
          mul_le_mul_of_nonneg_left ih hq0
      _ = _ := by ring

/-! The witnesses: `λ = 1/4`, `η = 1` and the start `U₀ = [1/8 0]`, `V₀ = [1/8 1/8]`, in the
ball as `3/64 ≤ 1/16`. -/

example := frobeniusSq_decayRun_sqLogit_le (η := 1) (lam := 1 / 4) zero_le_one (by norm_num)
  (by norm_num) (U₀ := (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ)) (V₀ := !![1 / 8, 1 / 8])
  (by norm_num [frobeniusSq, Fin.sum_univ_two]) 3

example := frobeniusSq_decayStep_sqLogit_le (η := 1) (lam := 1 / 4) zero_le_one (by norm_num)
  (by norm_num) (U := (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ)) (V := !![1 / 8, 1 / 8])
  (by norm_num [frobeniusSq, Fin.sum_univ_two])

example := frobeniusSq_sqLogitGrad_mul_le (U := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (V := 1)
  (ρ := 2) (by norm_num [frobeniusSq])

example := tendsto_zero_of_tendsto_frobeniusSq (l := atTop)
  (A := fun _ : ℕ => (0 : Matrix (Fin 1) (Fin 1) ℂ)) (by simp [frobeniusSq])

end Transformer.Grokking.FlatSpectrum.Training.Scope
