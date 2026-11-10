module

public import Transformer.Grokking.FlatSpectrum.Training.StepSize

/-!
# A step lowers the weight-decayed squared loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
For the squared loss on `Ω` write `G = P_Ω (U Vᴴ − K)`, `M = [[λ, G], [Gᴴ, λ]]`, `W = [U; V]`
and `D = M W`, the gradient of the weight-decayed loss `F`. The loss is quadratic in the
product, so with `G` frozen (`decayedLoss_fitLoss_sub`)

`F(W') − F(W) = ½ (⟨W', M W'⟩ − ⟨W, M W⟩) + ½ ‖P_Ω (U' V'ᴴ − U Vᴴ)‖²`.

For the step `W' = W − η D` the first term is `−η ‖D‖² + (η²/2) ⟨D, M D⟩`
(`pairing_one_sub_smul`) with `⟨D, M D⟩ ≤ (λ + ‖G‖) ‖D‖²`, and
`U' V'ᴴ − U Vᴴ = −η (D_U V'ᴴ + U D_Vᴴ)` has squared norm at most `2 η² ρ ‖D‖²` when
`‖U‖², ‖V'‖² ≤ ρ` (`frobeniusSq_step_mul_sub_le`). So `η (λ + ‖G‖ + 2ρ) ≤ 1` gives
`F(W') + (η/2) ‖D‖² ≤ F(W)` (`decayedLoss_decayStep_add_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k p : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype p] {Ω : Set (m × n)}
  {K : Matrix m n ℂ}

/-- The squared loss is quadratic: its expansion at `X` is exact. -/
lemma fitLoss_eq_add (Ω : Set (m × n)) (K X Y : Matrix m n ℂ) :
    fitLoss Ω K Y = fitLoss Ω K X + pairing (mask Ω (X - K)) (Y - X) +
      frobeniusSq (mask Ω (Y - X)) / 2 := by
  have e : mask Ω (Y - K) = mask Ω (X - K) + mask Ω (Y - X) := by
    rw [mask_sub Ω Y X, mask_sub Ω Y K, mask_sub Ω X K]
    abel
  rw [fitLoss, fitLoss, e, frobeniusSq_add, ← pairing_mask_left]
  ring

/-- With the gradient `G = P_Ω (U Vᴴ − K)` frozen, the weight-decayed squared loss changes by
half the change of the quadratic form of `M = [[λ, G], [Gᴴ, λ]]` plus half the squared change
of the seen entries. -/
lemma decayedLoss_fitLoss_sub [DecidableEq m] [DecidableEq n] (Ω : Set (m × n))
    (K : Matrix m n ℂ) (lam : ℝ) (U U' : Matrix m k ℂ) (V V' : Matrix n k ℂ) :
    decayedLoss (fitLoss Ω K) lam U' V' - decayedLoss (fitLoss Ω K) lam U V =
      (pairing (stack U' V') (lift lam (mask Ω (U * Vᴴ - K)) * stack U' V') -
        pairing (stack U V) (lift lam (mask Ω (U * Vᴴ - K)) * stack U V)) / 2 +
      frobeniusSq (mask Ω (U' * V'ᴴ - U * Vᴴ)) / 2 := by
  rw [decayedLoss, decayedLoss, fitLoss_eq_add Ω K (U * Vᴴ) (U' * V'ᴴ), pairing_stack_lift,
    pairing_stack_lift, pairing_sub_right]
  ring

/-- The quadratic form of a Hermitian `M` after the step `W ← (1 − η M) W`. -/
lemma pairing_one_sub_smul [DecidableEq p] {M : Matrix p p ℂ} (hM : M.IsHermitian) (η : ℝ)
    (W : Matrix p k ℂ) :
    pairing ((1 - η • M) * W) (M * ((1 - η • M) * W)) =
      pairing W (M * W) - 2 * η * frobeniusSq (M * W) +
        η ^ 2 * pairing (M * W) (M * (M * W)) := by
  have hsym : pairing W (M * (M * W)) = frobeniusSq (M * W) := by
    rw [← pairing_self, pairing_mul_left M W, hM.eq]
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, Matrix.mul_sub, Matrix.mul_smul,
    pairing_sub_left, pairing_sub_right, pairing_sub_right, pairing_smul_right,
    pairing_smul_left, pairing_smul_left, pairing_smul_right, pairing_self, hsym]
  ring

example := pairing_one_sub_smul (M := (1 : Matrix (Fin 1) (Fin 1) ℂ)) isHermitian_one 1
  (1 : Matrix (Fin 1) (Fin 1) ℂ)

/-- The product of the factors moves by `−η (A V'ᴴ + U Bᴴ)` when the factors move by `−η A`
and `−η B`; in squared norm by at most `2 η² (‖A‖² ‖V'‖² + ‖U‖² ‖B‖²)`. -/
lemma frobeniusSq_step_mul_sub_le (η : ℝ) (U A : Matrix m k ℂ) (V B : Matrix n k ℂ) :
    frobeniusSq ((U - η • A) * (V - η • B)ᴴ - U * Vᴴ) ≤
      2 * η ^ 2 * (frobeniusSq A * frobeniusSq (V - η • B) + frobeniusSq U * frobeniusSq B) := by
  have e : (U - η • A) * (V - η • B)ᴴ - U * Vᴴ = (-η) • (A * (V - η • B)ᴴ + U * Bᴴ) := by
    rw [conjTranspose_sub, conjTranspose_smul, star_trivial]
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul]
    module
  rw [e, frobeniusSq_smul]
  have h1 := frobeniusSq_add_le (A * (V - η • B)ᴴ) (U * Bᴴ)
  have h2 := frobeniusSq_mul_le A (V - η • B)ᴴ
  have h3 := frobeniusSq_mul_le U Bᴴ
  rw [frobeniusSq_conjTranspose] at h2 h3
  nlinarith [sq_nonneg η]

variable [DecidableEq m] [DecidableEq n]

/-- **A step lowers the weight-decayed squared loss** (`review/flat_spectrum.md`, item WD8):
with `G = P_Ω (U Vᴴ − K)` of norm at most `g`, `‖U‖² ≤ ρ` before the step, `‖V'‖² ≤ ρ` after
it and `η (λ + g + 2ρ) ≤ 1`, the loss falls by at least `(η/2) ‖M W‖²`, half the step times
the squared gradient. -/
theorem decayedLoss_decayStep_add_le {η lam g ρ : ℝ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (hη : 0 ≤ η) (hg : 0 ≤ g) (hG : frobeniusSq (mask Ω (U * Vᴴ - K)) ≤ g ^ 2)
    (hU : frobeniusSq U ≤ ρ)
    (hV : frobeniusSq (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).2 ≤ ρ)
    (hstep : η * (lam + g + 2 * ρ) ≤ 1) :
    decayedLoss (fitLoss Ω K) lam (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).1
        (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).2 +
      η / 2 * frobeniusSq (lift lam (mask Ω (U * Vᴴ - K)) * stack U V) ≤
      decayedLoss (fitLoss Ω K) lam U V := by
  set G := mask Ω (U * Vᴴ - K) with hGdef
  set DU := G * V + lam • U with hDU
  set DV := Gᴴ * U + lam • V with hDV
  have hstack : stack (U - η • DU) (V - η • DV) = (1 - η • lift lam G) * stack U V :=
    stack_decayStep (fun X => mask Ω (X - K)) η lam U V
  have hD : lift lam G * stack U V = stack DU DV := lift_mul_stack lam G U V
  have h0 := decayedLoss_fitLoss_sub Ω K lam U (U - η • DU) V (V - η • DV)
  rw [← hGdef, hstack, pairing_one_sub_smul (lift_isHermitian lam G), hD, pairing_stack_lift,
    frobeniusSq_stack] at h0
  have hq := two_mul_abs_pairing_mul_le hg hG DU DV
  have hq' := neg_abs_le (pairing G (DU * DVᴴ))
  have hq'' := le_abs_self (pairing G (DU * DVᴴ))
  have hE := (frobeniusSq_mask_le Ω _).trans (frobeniusSq_step_mul_sub_le η U DU V DV)
  have ha := frobeniusSq_nonneg DU
  have hc := frobeniusSq_nonneg DV
  have hS : 0 ≤ frobeniusSq DU + frobeniusSq DV := add_nonneg ha hc
  have e1 : frobeniusSq DU * frobeniusSq (V - η • DV) + frobeniusSq U * frobeniusSq DV ≤
      ρ * (frobeniusSq DU + frobeniusSq DV) := by
    have hV' : frobeniusSq (V - η • DV) ≤ ρ := hV
    have := mul_le_mul_of_nonneg_left hV' ha
    have := mul_le_mul_of_nonneg_right hU hc
    linarith
  have e2 := mul_le_mul_of_nonneg_left e1 (by positivity : (0 : ℝ) ≤ 2 * η ^ 2)
  have e3 := mul_le_mul_of_nonneg_left hstep (mul_nonneg hη hS)
  have e4 := mul_le_mul_of_nonneg_left hq (sq_nonneg η)
  have e5 := mul_le_mul_of_nonneg_left hq'' (sq_nonneg η)
  have key : decayedLoss (fitLoss Ω K) lam (U - η • DU) (V - η • DV) +
      η / 2 * frobeniusSq (lift lam G * stack U V) ≤ decayedLoss (fitLoss Ω K) lam U V := by
    rw [hD, frobeniusSq_stack]
    nlinarith
  exact key

/-- At the origin, a saddle of the `1 × 1` kernel `1` for `λ = 1`, the step `η = 1/2` is `0`. -/
example := decayedLoss_decayStep_add_le (Ω := (Set.univ : Set (Fin 1 × Fin 1))) (K := 1)
  (η := 1 / 2) (lam := 1) (g := 1) (ρ := 0) (U := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (V := 0)
  (by norm_num) zero_le_one (by simp [frobeniusSq, mask]) (by simp [frobeniusSq])
  (by simp [decayStep, frobeniusSq]) (by norm_num)

variable {η lam κ ρ : ℝ}

/-- **A step from the ball lowers the loss** (`review/flat_spectrum.md`, items WD6 and WD8):
under the bounds of WD6 and `η (λ + ρ/2 + κ + 2ρ) ≤ 1`, a step from `‖U‖² + ‖V‖² ≤ ρ` lowers
the weight-decayed squared loss by at least `(η/2) ‖M W‖²`. -/
theorem decayedLoss_decayStep_add_le_of_ball {U : Matrix m k ℂ} {V : Matrix n k ℂ} (hη : 0 ≤ η)
    (hlam : 0 < lam) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    decayedLoss (fitLoss Ω K) lam (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).1
        (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).2 +
      η / 2 * frobeniusSq (lift lam (mask Ω (U * Vᴴ - K)) * stack U V) ≤
      decayedLoss (fitLoss Ω K) lam U V := by
  have h := frobeniusSq_decayStep_le (Ω := Ω) hη hlam hκ0 hκ hρ hstep hw
  have hU := frobeniusSq_nonneg U
  have hV := frobeniusSq_nonneg V
  have h1 := frobeniusSq_nonneg (decayStep (fun X => mask Ω (X - K)) η lam (U, V)).1
  exact decayedLoss_decayStep_add_le hη (by linarith) (frobeniusSq_mask_sub_le hκ0 hκ hw)
    (by linarith) (by linarith) hdesc

/-- **Every step of a run from the ball lowers the loss** (`review/flat_spectrum.md`, item
WD8), by at least `(η/2) ‖M_t W_t‖²`. -/
theorem decayedLoss_decayRun_succ_add_le (hη : 0 ≤ η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    decayedLoss (fitLoss Ω K) lam (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ (t + 1)).1
        (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ (t + 1)).2 +
      η / 2 * frobeniusSq (lift lam (mask Ω
        ((decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).1 *
          (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).2ᴴ - K)) *
        stack (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).1
          (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).2) ≤
      decayedLoss (fitLoss Ω K) lam (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).1
        (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t).2 := by
  rw [decayRun, Function.iterate_succ_apply', ← decayRun]
  exact decayedLoss_decayStep_add_le_of_ball hη hlam hκ0 hκ hρ hstep hdesc
    (frobeniusSq_decayRun_le hη hlam hκ0 hκ hρ hstep hw t)

/-! The witness: the `1 × 1` kernel `1` seen on its cell, `λ = 1`, `κ = 1`, `ρ = 2`,
`η = 1/10`, from `U₀ = [1 0]`, `V₀ = [0 1]`: `2 η (λ² + 2²) = 1` and `η (λ + 2 + 2ρ) = 7/10`. -/

example := decayedLoss_decayStep_add_le_of_ball (Ω := Set.univ) (η := 1 / 10) (by norm_num)
  one_pos zero_le_one stepWitness_kernel (by norm_num) (by norm_num) (by norm_num)
  stepWitness_ball

example := decayedLoss_decayRun_succ_add_le (Ω := Set.univ) (η := 1 / 10) (by norm_num)
  one_pos zero_le_one stepWitness_kernel (by norm_num) (by norm_num) (by norm_num)
  stepWitness_ball 3

end Transformer.Grokking.FlatSpectrum.Training
