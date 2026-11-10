module

public import Transformer.Grokking.FlatSpectrum.Training.Convergence
public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss

/-!
# An explicit admissible step for the squared loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD6.
For the squared loss on `Ω` with `‖K‖ ≤ κ`, weight decay `λ > 0` and a radius `ρ` with
`κ² ≤ λ ρ`, put `γ = ρ/2 + κ`. In the ball `‖U‖² + ‖V‖² ≤ ρ` the gradient
`G = P_Ω (U Vᴴ − K)` has `‖G‖ ≤ γ` (`frobeniusSq_mask_sub_le`). If `2 η (λ² + γ²) ≤ λ`, then

* a step keeps the ball (`frobeniusSq_decayStep_le`): with `w = ‖W‖²`,
  `‖(1 − η M) W‖² = w − 2η ⟨W, M W⟩ + η² ‖M W‖² ≤ (1 − η λ) w + η κ²`, from
  `⟨G, U Vᴴ⟩ ≥ −κ²/4` and `‖M W‖² ≤ 2 (λ² + γ²) w`; so a run from the ball stays in it
  (`frobeniusSq_decayRun_le`);
* every step matrix in the ball is positive definite (`one_sub_smul_lift_posDef_of_ball`), as
  `η (λ + γ) < 1`.

So for the squared loss the step condition of `isDecayedMin_of_tendsto_decayRun` follows from
this bound on `η` (`isDecayedMin_fitLoss_of_tendsto_decayRun`). With no cell seen the run is
`(1 − η λ)^t (U₀, V₀)` (`decayRun_mask_empty`), which gives a convergent witness.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {Ω : Set (m × n)}
  {K : Matrix m n ℂ} {η lam κ ρ : ℝ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}

/-- In the ball `‖U‖² + ‖V‖² ≤ ρ` the gradient has `‖P_Ω (U Vᴴ − K)‖ ≤ ρ/2 + κ`. -/
lemma frobeniusSq_mask_sub_le (hκ0 : 0 ≤ κ) (hκ : frobeniusSq K ≤ κ ^ 2)
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    frobeniusSq (mask Ω (U * Vᴴ - K)) ≤ (ρ / 2 + κ) ^ 2 := by
  have ha := frobeniusSq_nonneg U
  have hb := frobeniusSq_nonneg V
  have hX : frobeniusSq (U * Vᴴ) ≤ (ρ / 2) ^ 2 := by
    have h := frobeniusSq_mul_le U Vᴴ
    rw [frobeniusSq_conjTranspose] at h
    nlinarith [sq_nonneg (frobeniusSq U - frobeniusSq V), mul_nonneg (sub_nonneg.mpr hw)
      (by linarith : 0 ≤ ρ + (frobeniusSq U + frobeniusSq V))]
  have hXK : |pairing (U * Vᴴ) K| ≤ ρ / 2 * κ :=
    abs_pairing_le (mul_nonneg (by linarith) hκ0) (by
      calc frobeniusSq (U * Vᴴ) * frobeniusSq K ≤ (ρ / 2) ^ 2 * κ ^ 2 :=
            mul_le_mul hX hκ (frobeniusSq_nonneg _) (sq_nonneg _)
        _ = (ρ / 2 * κ) ^ 2 := by ring)
  have h1 := frobeniusSq_sub (U * Vᴴ) K
  have h2 := neg_abs_le (pairing (U * Vᴴ) K)
  have h3 := frobeniusSq_mask_le Ω (U * Vᴴ - K)
  nlinarith

variable [DecidableEq m] [DecidableEq n]

/-- **A step keeps the ball** (`review/flat_spectrum.md`, item WD6): for the squared loss with
`‖K‖ ≤ κ`, `κ² ≤ λ ρ` and `2 η (λ² + (ρ/2 + κ)²) ≤ λ`, a step from `‖U‖² + ‖V‖² ≤ ρ` ends in
the same ball. -/
theorem frobeniusSq_decayStep_le (hη : 0 ≤ η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    frobeniusSq (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).1 +
      frobeniusSq (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).2 ≤ ρ := by
  rw [← frobeniusSq_stack, stack_decayStep, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul,
    frobeniusSq_sub, pairing_smul_right, frobeniusSq_smul, pairing_stack_lift, frobeniusSq_stack]
  have hw0 := add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V)
  have h1 := frobeniusSq_lift_mul_stack_le (frobeniusSq_mask_sub_le (Ω := Ω) hκ0 hκ hw) lam U V
  have h2 := neg_le_pairing_mask_sub Ω K (U * Vᴴ)
  have hηl : 2 * (η * lam) ≤ 1 := le_of_mul_le_mul_left (by
    nlinarith [mul_nonneg hη (sq_nonneg (ρ / 2 + κ))] : lam * (2 * (η * lam)) ≤ lam * 1) hlam
  have e1 : 0 ≤ η * (pairing (mask Ω (U * Vᴴ - K)) (U * Vᴴ) + κ ^ 2 / 4) :=
    mul_nonneg hη (by linarith)
  have e2 := mul_le_mul_of_nonneg_left h1 (sq_nonneg η)
  have e3 := mul_le_mul_of_nonneg_left hstep (mul_nonneg hη hw0)
  have e4 := mul_nonneg (by linarith : 0 ≤ 1 - η * lam) (sub_nonneg.mpr hw)
  have e5 := mul_nonneg hη (sub_nonneg.mpr hρ)
  nlinarith

/-- **A run from the ball stays in it** (`review/flat_spectrum.md`, item WD6). -/
theorem frobeniusSq_decayRun_le (hη : 0 ≤ η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    frobeniusSq (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).2 ≤ ρ := by
  induction t with
  | zero => exact hw
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun]
    exact frobeniusSq_decayStep_le hη hlam hκ0 hκ hρ hstep ih

/-- **Steps in the ball are admissible** (`review/flat_spectrum.md`, item WD6): for the squared
loss with `‖K‖ ≤ κ` and `2 η (λ² + (ρ/2 + κ)²) ≤ λ`, the step matrix `1 − η M` is positive
definite wherever `‖U‖² + ‖V‖² ≤ ρ`. -/
theorem one_sub_smul_lift_posDef_of_ball (hη : 0 ≤ η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    (1 - η • lift lam (mask Ω (U * Vᴴ - K))).PosDef := by
  have hγ : 0 ≤ ρ / 2 + κ := by
    linarith [frobeniusSq_nonneg U, frobeniusSq_nonneg V]
  refine one_sub_smul_lift_posDef hη hγ (frobeniusSq_mask_sub_le hκ0 hκ hw) ?_
  rw [abs_of_pos hlam]
  by_contra hcon
  have h1 := mul_le_mul_of_nonneg_left hstep (add_nonneg hlam.le hγ)
  have h2 := mul_le_mul_of_nonneg_right (not_lt.mp hcon)
    (by positivity : 0 ≤ 2 * (lam ^ 2 + (ρ / 2 + κ) ^ 2))
  nlinarith [sq_nonneg (lam - (ρ / 2 + κ)), sq_nonneg (ρ / 2 + κ), mul_pos hlam hlam]

/-- **Convergent runs on the squared loss, with an explicit step**
(`review/flat_spectrum.md`, items WD5 and WD6): for the squared loss on `Ω` with `‖K‖ ≤ κ`,
weight decay `λ > 0`, a start of full row rank in the ball `‖U₀‖² + ‖V₀‖² ≤ ρ` with
`κ² ≤ λ ρ`, and a step `0 < η` with `2 η (λ² + (ρ/2 + κ)²) ≤ λ`, the limit of a convergent run
has the least weight-decayed loss among the factorizations of every width. -/
theorem isDecayedMin_fitLoss_of_tendsto_decayRun (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hlim : Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V))) :
    IsDecayedMin (fitLoss Ω K) lam U V :=
  isDecayedMin_of_tendsto_decayRun (fitLoss_first_order Ω K) (continuous_mask_sub Ω K) hη hfull
    (fun t => one_sub_smul_lift_posDef_of_ball hη.le hlam hκ0 hκ hstep
      (frobeniusSq_decayRun_le hη.le hlam hκ0 hκ hρ hstep hw t)) hlim

omit [DecidableEq m] [DecidableEq n] in
/-- With no cell seen the gradient vanishes and the run only decays:
`(U_t, V_t) = (1 − η λ)^t (U₀, V₀)`. -/
theorem decayRun_mask_empty (K : Matrix m n ℂ) (η lam : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (t : ℕ) :
    decayRun (fun X => mask (∅ : Set (m × n)) (X - K)) η lam U₀ V₀ t =
      ((1 - η * lam) ^ t • U₀, (1 - η * lam) ^ t • V₀) := by
  have h0 (X : Matrix m n ℂ) : mask (∅ : Set (m × n)) X = 0 := by
    ext i j
    simp [mask]
  induction t with
  | zero => simp [decayRun]
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun, ih, decayStep]
    simp only [h0, conjTranspose_zero, Matrix.zero_mul, zero_add, pow_succ]
    ext : 1 <;> module

/-! The witness: the `1 × 1` kernel `1`, `λ = 1`, `κ = 1`, `ρ = 2`, `η = 1/10`, so that
`2 η (λ² + (ρ/2 + κ)²) = 1 = λ`, from `U₀ = [1 0]`, `V₀ = [0 1]` with `‖U₀‖² + ‖V₀‖² = 2`. -/

lemma stepWitness_ball : frobeniusSq (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ)
    (0 : Matrix (Fin 1) (Fin 1) ℂ)) + frobeniusSq (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ)
      (1 : Matrix (Fin 1) (Fin 1) ℂ)) ≤ 2 := by
  rw [← frobeniusSq_stack, stack_fromCols_one]
  norm_num [frobeniusSq, Matrix.one_apply]

lemma stepWitness_kernel : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ (1 : ℝ) ^ 2 := by
  simp [frobeniusSq]

example := frobeniusSq_mask_sub_le (Ω := Set.univ) zero_le_one stepWitness_kernel stepWitness_ball

example := frobeniusSq_decayStep_le (Ω := Set.univ) (η := 1 / 10) (by norm_num) one_pos
  zero_le_one stepWitness_kernel (by norm_num) (by norm_num) stepWitness_ball

example := frobeniusSq_decayRun_le (Ω := Set.univ) (η := 1 / 10) (by norm_num) one_pos
  zero_le_one stepWitness_kernel (by norm_num) (by norm_num) stepWitness_ball 3

example := one_sub_smul_lift_posDef_of_ball (Ω := Set.univ) (η := 1 / 10) (by norm_num) one_pos
  zero_le_one stepWitness_kernel (by norm_num) stepWitness_ball

/-- With no cell seen, the run from `U₀ = [1 0]`, `V₀ = [0 1]` is `(9/10)^t (U₀, V₀) → 0`. -/
example : IsDecayedMin (fitLoss (∅ : Set (Fin 1 × Fin 1)) (1 : Matrix (Fin 1) (Fin 1) ℂ)) 1
    (0 : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ) := by
  refine isDecayedMin_fitLoss_of_tendsto_decayRun (η := 1 / 10) (κ := 1) (ρ := 2) (by norm_num)
    one_pos zero_le_one stepWitness_kernel (by norm_num) (by norm_num) stepWitness_ball
    (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) ?_
  rw [funext (decayRun_mask_empty 1 (1 / 10) 1 _ _)]
  have hp := tendsto_pow_atTop_nhds_zero_of_abs_lt_one (r := 1 - 1 / 10 * (1 : ℝ))
    (by rw [abs_lt]; constructor <;> norm_num)
  have hU := hp.smul_const (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  have hV := hp.smul_const (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  rw [zero_smul] at hU hV
  exact hU.prodMk_nhds hV

end Transformer.Grokking.FlatSpectrum.Training
