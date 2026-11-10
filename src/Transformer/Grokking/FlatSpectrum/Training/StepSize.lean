module

public import Transformer.Grokking.FlatSpectrum.Training.Convergence
public import Transformer.Grokking.FlatSpectrum.Training.SquaredBounds

/-!
# An explicit admissible step

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD6.
Fix a weight decay `λ > 0` and a radius `ρ`. The gradient field `G` of the loss enters through
two bounds at the matrices with `‖X‖ ≤ ρ/2`: `‖G X‖ ≤ γ` and `⟨G X, X⟩ ≥ −b`. In the ball
`‖U‖² + ‖V‖² ≤ ρ` the product has `‖U Vᴴ‖ ≤ ρ/2` (`frobeniusSq_mul_conjTranspose_le`). If
`4 b ≤ λ ρ` and `2 η (λ² + γ²) ≤ λ`, then

* a step keeps the ball (`frobeniusSq_decayStep_le`): with `w = ‖W‖²`,
  `‖(1 − η M) W‖² = w − 2η ⟨W, M W⟩ + η² ‖M W‖² ≤ (1 − η λ) w + 4 η b`, from
  `⟨W, M W⟩ = λ w + 2 ⟨G, U Vᴴ⟩ ≥ λ w − 2 b` and `‖M W‖² ≤ 2 (λ² + γ²) w`; so a run from the
  ball stays in it (`frobeniusSq_decayRun_le`);
* every step matrix in the ball is positive definite (`one_sub_smul_lift_posDef_of_ball`), as
  `η (λ + γ) < 1`.

So the step condition of `isDecayedMin_of_tendsto_decayRun` follows from these bounds
(`isDecayedMin_of_tendsto_decayRun_of_ball`). The examples are the squared loss, where
`γ = ρ/2 + κ` and `b = κ²/4` for `‖K‖ ≤ κ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ}
  {G : Matrix m n ℂ → Matrix m n ℂ} {η lam b γ ρ : ℝ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}

/-- In the ball `‖U‖² + ‖V‖² ≤ ρ` the product has `‖U Vᴴ‖ ≤ ρ/2`. -/
lemma frobeniusSq_mul_conjTranspose_le (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    frobeniusSq (U * Vᴴ) ≤ (ρ / 2) ^ 2 := by
  have ha := frobeniusSq_nonneg U
  have hb := frobeniusSq_nonneg V
  have h := frobeniusSq_mul_le U Vᴴ
  rw [frobeniusSq_conjTranspose] at h
  nlinarith [sq_nonneg (frobeniusSq U - frobeniusSq V), mul_nonneg (sub_nonneg.mpr hw)
    (by linarith : 0 ≤ ρ + (frobeniusSq U + frobeniusSq V))]

example := frobeniusSq_mul_conjTranspose_le stepWitness_ball

variable [DecidableEq m] [DecidableEq n]

/-- **A step keeps the ball** (`review/flat_spectrum.md`, item WD6): if `‖G X‖ ≤ γ` and
`⟨G X, X⟩ ≥ −b` for `‖X‖ ≤ ρ/2`, `4 b ≤ λ ρ` and `2 η (λ² + γ²) ≤ λ`, then a step from
`‖U‖² + ‖V‖² ≤ ρ` ends in the same ball. -/
theorem frobeniusSq_decayStep_le (hη : 0 ≤ η) (hlam : 0 < lam)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X) (hρ : 4 * b ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam) (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    frobeniusSq (decayStep G η lam (U, V)).1 + frobeniusSq (decayStep G η lam (U, V)).2 ≤ ρ := by
  have hX := frobeniusSq_mul_conjTranspose_le hw
  have h1 := frobeniusSq_lift_mul_stack_le (hγ _ hX) lam U V
  have h2 := hb _ hX
  rw [← frobeniusSq_stack, stack_decayStep, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul,
    frobeniusSq_sub, pairing_smul_right, frobeniusSq_smul, pairing_stack_lift, frobeniusSq_stack]
  have hw0 := add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V)
  have hηl : 2 * (η * lam) ≤ 1 := le_of_mul_le_mul_left (by
    nlinarith [mul_nonneg hη (sq_nonneg γ)] : lam * (2 * (η * lam)) ≤ lam * 1) hlam
  have e1 : 0 ≤ η * (pairing (G (U * Vᴴ)) (U * Vᴴ) + b) := mul_nonneg hη (by linarith)
  have e2 := mul_le_mul_of_nonneg_left h1 (sq_nonneg η)
  have e3 := mul_le_mul_of_nonneg_left hstep (mul_nonneg hη hw0)
  have e4 := mul_nonneg (by linarith : 0 ≤ 1 - η * lam) (sub_nonneg.mpr hw)
  have e5 := mul_nonneg hη (sub_nonneg.mpr hρ)
  nlinarith

/-- **A run from the ball stays in it** (`review/flat_spectrum.md`, item WD6), under the
bounds of `frobeniusSq_decayStep_le`. -/
theorem frobeniusSq_decayRun_le (hη : 0 ≤ η) (hlam : 0 < lam)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X) (hρ : 4 * b ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    frobeniusSq (decayRun G η lam U₀ V₀ t).1 + frobeniusSq (decayRun G η lam U₀ V₀ t).2 ≤ ρ := by
  induction t with
  | zero => exact hw
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun]
    exact frobeniusSq_decayStep_le hη hlam hγ hb hρ hstep ih

/-- **Steps in the ball are admissible** (`review/flat_spectrum.md`, item WD6): if `‖G X‖ ≤ γ`
for `‖X‖ ≤ ρ/2` and `2 η (λ² + γ²) ≤ λ`, the step matrix `1 − η M` is positive definite
wherever `‖U‖² + ‖V‖² ≤ ρ`. -/
theorem one_sub_smul_lift_posDef_of_ball (hη : 0 ≤ η) (hlam : 0 < lam) (hγ0 : 0 ≤ γ)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam) (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    (1 - η • lift lam (G (U * Vᴴ))).PosDef := by
  refine one_sub_smul_lift_posDef hη hγ0 (hγ _ (frobeniusSq_mul_conjTranspose_le hw)) ?_
  rw [abs_of_pos hlam]
  by_contra hcon
  have h1 := mul_le_mul_of_nonneg_left hstep (add_nonneg hlam.le hγ0)
  have h2 := mul_le_mul_of_nonneg_right (not_lt.mp hcon)
    (by positivity : 0 ≤ 2 * (lam ^ 2 + γ ^ 2))
  nlinarith [sq_nonneg (lam - γ), sq_nonneg γ, mul_pos hlam hlam]

/-- **Convergent runs, with an explicit step** (`review/flat_spectrum.md`, items WD5 and
WD6): let `L` satisfy the first-order inequality with a continuous `G` such that `‖G X‖ ≤ γ`
and `⟨G X, X⟩ ≥ −b` for `‖X‖ ≤ ρ/2`. For a weight decay `λ > 0`, a start of full row rank in
the ball `‖U₀‖² + ‖V₀‖² ≤ ρ` with `4 b ≤ λ ρ`, and a step `0 < η` with `2 η (λ² + γ²) ≤ λ`,
the limit of a convergent run has the least weight-decayed loss among the factorizations of
every width. -/
theorem isDecayedMin_of_tendsto_decayRun_of_ball
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G) (hη : 0 < η)
    (hlam : 0 < lam) (hγ0 : 0 ≤ γ)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X) (hρ : 4 * b ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) : IsDecayedMin L lam U V :=
  isDecayedMin_of_tendsto_decayRun hconv hG hη hfull
    (fun t => one_sub_smul_lift_posDef_of_ball hη.le hlam hγ0 hγ hstep
      (frobeniusSq_decayRun_le hη.le hlam hγ hb hρ hstep hw t)) hlim

variable {Ω : Set (m × n)} {K : Matrix m n ℂ} {κ : ℝ}

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
    IsDecayedMin (fitLoss Ω K) lam U V := by
  have hρ0 : 0 ≤ ρ := (add_nonneg (frobeniusSq_nonneg U₀) (frobeniusSq_nonneg V₀)).trans hw
  exact isDecayedMin_of_tendsto_decayRun_of_ball (fitLoss_first_order Ω K)
    (continuous_mask_sub Ω K) hη hlam (by linarith) (frobeniusSq_mask_sub_le hρ0 hκ0 hκ)
    (fun X _ => neg_le_pairing_mask_sub_of_le hκ X) (by linarith) hstep hw hfull hlim

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

/-! The witness: the squared loss of the `1 × 1` kernel `1`, `λ = 1`, `γ = 2`, `b = 1/4`,
`ρ = 2`, `η = 1/10`, so that `4 b = 1 ≤ λ ρ` and `2 η (λ² + γ²) = 1 = λ`. -/

example := frobeniusSq_decayStep_le (η := 1 / 10) (by norm_num) one_pos stepWitness_bound
  (fun X _ => stepWitness_low X) (by norm_num) (by norm_num) stepWitness_ball

example := frobeniusSq_decayRun_le (η := 1 / 10) (by norm_num) one_pos stepWitness_bound
  (fun X _ => stepWitness_low X) (by norm_num) (by norm_num) stepWitness_ball 3

example := one_sub_smul_lift_posDef_of_ball (η := 1 / 10) (by norm_num) one_pos (by norm_num)
  stepWitness_bound (by norm_num) stepWitness_ball

/-- With no cell seen, the run from `U₀ = [1 0]`, `V₀ = [0 1]` is `(9/10)^t (U₀, V₀) → 0`. -/
private lemma stepWitness_tendsto :
    Tendsto (decayRun (fun X => mask (∅ : Set (Fin 1 × Fin 1)) (X - 1)) (1 / 10) 1
      (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
      (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))) atTop
      (𝓝 (0, 0)) := by
  rw [funext (decayRun_mask_empty 1 (1 / 10) 1 _ _)]
  have hp := tendsto_pow_atTop_nhds_zero_of_abs_lt_one (r := 1 - 1 / 10 * (1 : ℝ))
    (by rw [abs_lt]; constructor <;> norm_num)
  have hU := hp.smul_const (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  have hV := hp.smul_const (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  rw [zero_smul] at hU hV
  exact hU.prodMk_nhds hV

example := isDecayedMin_of_tendsto_decayRun_of_ball (fitLoss_first_order ∅ 1)
  (continuous_mask_sub ∅ 1) (η := 1 / 10) (by norm_num) one_pos (by norm_num)
  (frobeniusSq_mask_sub_le zero_le_two zero_le_one stepWitness_kernel)
  (fun X _ => neg_le_pairing_mask_sub_of_le stepWitness_kernel X) (by norm_num) (by norm_num)
  stepWitness_ball (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) stepWitness_tendsto

example := isDecayedMin_fitLoss_of_tendsto_decayRun (η := 1 / 10) (κ := 1) (ρ := 2)
  (by norm_num) one_pos zero_le_one stepWitness_kernel (by norm_num) (by norm_num)
  stepWitness_ball (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) stepWitness_tendsto

end Transformer.Grokking.FlatSpectrum.Training
