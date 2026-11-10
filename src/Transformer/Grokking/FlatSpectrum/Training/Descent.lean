module

public import Transformer.Grokking.FlatSpectrum.Training.StepSize

/-!
# A step lowers the weight-decayed loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
Write `G = G (U Vᴴ)`, `M = [[λ, G], [Gᴴ, λ]]`, `W = [U; V]` and `D = M W`, the gradient of the
weight-decayed loss `F`. Let the loss lie below its quadratic model with constant `s` at the
matrices with `‖X‖, ‖Y‖ ≤ ρ/2`: `L Y ≤ L X + ⟨G X, Y − X⟩ + (s/2) ‖Y − X‖²`. With `G` frozen
(`decayedLoss_sub_le`)

`F(W') − F(W) ≤ ½ (⟨W', M W'⟩ − ⟨W, M W⟩) + (s/2) ‖U' V'ᴴ − U Vᴴ‖²`.

For the step `W' = W − η D` the first term is `−η ‖D‖² + (η²/2) ⟨D, M D⟩`
(`pairing_one_sub_smul`) with `⟨D, M D⟩ ≤ (λ + ‖G‖) ‖D‖²`, and
`U' V'ᴴ − U Vᴴ = −η (D_U V'ᴴ + U D_Vᴴ)` has squared norm at most `2 η² ρ ‖D‖²` when
`‖U‖², ‖V'‖² ≤ ρ` (`frobeniusSq_step_mul_sub_le`). So `η (λ + ‖G‖ + 2 s ρ) ≤ 1` gives
`F(W') + (η/2) ‖D‖² ≤ F(W)` (`decayedLoss_decayStep_add_le`). The examples are the squared
loss, where `s = 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k p : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype p]
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {η lam b γ s ρ : ℝ}

/-- With the gradient `G = G (U Vᴴ)` frozen, a loss below its quadratic model with constant `s`
changes the weight-decayed loss by at most half the change of the quadratic form of
`M = [[λ, G], [Gᴴ, λ]]` plus `s/2` times the squared change of the product. -/
lemma decayedLoss_sub_le [DecidableEq m] [DecidableEq n] (lam : ℝ) {U U' : Matrix m k ℂ}
    {V V' : Matrix n k ℂ}
    (hs : L (U' * V'ᴴ) ≤ L (U * Vᴴ) + pairing (G (U * Vᴴ)) (U' * V'ᴴ - U * Vᴴ) +
      s / 2 * frobeniusSq (U' * V'ᴴ - U * Vᴴ)) :
    decayedLoss L lam U' V' - decayedLoss L lam U V ≤
      (pairing (stack U' V') (lift lam (G (U * Vᴴ)) * stack U' V') -
        pairing (stack U V) (lift lam (G (U * Vᴴ)) * stack U V)) / 2 +
      s / 2 * frobeniusSq (U' * V'ᴴ - U * Vᴴ) := by
  rw [decayedLoss, decayedLoss, pairing_stack_lift, pairing_stack_lift]
  rw [pairing_sub_right] at hs
  linarith

example := decayedLoss_sub_le (L := fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1)
  (G := fun X => mask Set.univ (X - 1)) (s := 1) 1 (U := (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (U' := 1) (V := 0) (V' := 1) (fitLoss_le_add Set.univ 1 _ _)

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

variable [DecidableEq m] [DecidableEq n] {U : Matrix m k ℂ} {V : Matrix n k ℂ}

/-- **A step lowers the weight-decayed loss** (`review/flat_spectrum.md`, item WD8): let `L`
lie below its quadratic model with constant `s ≥ 0` for `‖X‖, ‖Y‖ ≤ ρ/2`. If the step goes
from the ball `‖U‖² + ‖V‖² ≤ ρ` into it, `G = G (U Vᴴ)` has norm at most `g`, and
`η (λ + g + 2 s ρ) ≤ 1`, the loss falls by at least `(η/2) ‖M W‖²`, half the step times the
squared gradient. -/
theorem decayedLoss_decayStep_add_le {g : ℝ} (hη : 0 ≤ η) (hg : 0 ≤ g) (hs0 : 0 ≤ s)
    (hG : frobeniusSq (G (U * Vᴴ)) ≤ g ^ 2)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ)
    (hw' : frobeniusSq (decayStep G η lam (U, V)).1 +
      frobeniusSq (decayStep G η lam (U, V)).2 ≤ ρ)
    (hstep : η * (lam + g + 2 * s * ρ) ≤ 1) :
    decayedLoss L lam (decayStep G η lam (U, V)).1 (decayStep G η lam (U, V)).2 +
      η / 2 * frobeniusSq (lift lam (G (U * Vᴴ)) * stack U V) ≤ decayedLoss L lam U V := by
  set Gm := G (U * Vᴴ) with hGdef
  set DU := Gm * V + lam • U with hDU
  set DV := Gmᴴ * U + lam • V with hDV
  have hstack : stack (U - η • DU) (V - η • DV) = (1 - η • lift lam Gm) * stack U V :=
    stack_decayStep G η lam U V
  have hD : lift lam Gm * stack U V = stack DU DV := lift_mul_stack lam Gm U V
  have hw'' : frobeniusSq (U - η • DU) + frobeniusSq (V - η • DV) ≤ ρ := hw'
  have h0 := decayedLoss_sub_le (L := L) (G := G) lam (hs (U * Vᴴ)
    ((U - η • DU) * (V - η • DV)ᴴ) (frobeniusSq_mul_conjTranspose_le hw)
    (frobeniusSq_mul_conjTranspose_le hw''))
  rw [← hGdef, hstack, pairing_one_sub_smul (lift_isHermitian lam Gm), hD, pairing_stack_lift,
    frobeniusSq_stack] at h0
  have hq := two_mul_abs_pairing_mul_le hg hG DU DV
  have hq' := le_abs_self (pairing Gm (DU * DVᴴ))
  have ha := frobeniusSq_nonneg DU
  have hc := frobeniusSq_nonneg DV
  have e1 : frobeniusSq DU * frobeniusSq (V - η • DV) + frobeniusSq U * frobeniusSq DV ≤
      ρ * (frobeniusSq DU + frobeniusSq DV) := by
    have hV' : frobeniusSq (V - η • DV) ≤ ρ := by linarith [frobeniusSq_nonneg (U - η • DU)]
    have hU' : frobeniusSq U ≤ ρ := by linarith [frobeniusSq_nonneg V]
    have := mul_le_mul_of_nonneg_left hV' ha
    have := mul_le_mul_of_nonneg_right hU' hc
    linarith
  have e2 := (frobeniusSq_step_mul_sub_le η U DU V DV).trans
    (mul_le_mul_of_nonneg_left e1 (by positivity : (0 : ℝ) ≤ 2 * η ^ 2))
  have e3 := mul_le_mul_of_nonneg_left e2 (by linarith : (0 : ℝ) ≤ s / 2)
  have e4 := mul_le_mul_of_nonneg_left hstep (mul_nonneg hη (add_nonneg ha hc))
  have e5 := mul_le_mul_of_nonneg_left hq (sq_nonneg η)
  have e6 := mul_le_mul_of_nonneg_left hq' (sq_nonneg η)
  have key : decayedLoss L lam (U - η • DU) (V - η • DV) +
      η / 2 * frobeniusSq (lift lam Gm * stack U V) ≤ decayedLoss L lam U V := by
    rw [hD, frobeniusSq_stack]
    nlinarith
  exact key

/-- At the origin, a saddle of the `1 × 1` kernel `1` for `λ = 1`, the step `η = 1/2` is `0`. -/
example := decayedLoss_decayStep_add_le (L := fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1)
  (η := 1 / 2) (lam := 1) (g := 1) (ρ := 0) (U := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (V := 0)
  (by norm_num) zero_le_one zero_le_one (by simp [frobeniusSq, mask])
  (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by simp [frobeniusSq])
  (by simp [decayStep, frobeniusSq]) (by norm_num)

/-- **A step from the ball lowers the loss** (`review/flat_spectrum.md`, items WD6 and WD8):
under the bounds of WD6, with `L` below its quadratic model with constant `s` for
`‖X‖, ‖Y‖ ≤ ρ/2` and `η (λ + γ + 2 s ρ) ≤ 1`, a step from `‖U‖² + ‖V‖² ≤ ρ` lowers the
weight-decayed loss by at least `(η/2) ‖M W‖²`. -/
theorem decayedLoss_decayStep_add_le_of_ball (hη : 0 ≤ η) (hlam : 0 < lam) (hγ0 : 0 ≤ γ)
    (hs0 : 0 ≤ s) (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hρ : 4 * b ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam)
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1) (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    decayedLoss L lam (decayStep G η lam (U, V)).1 (decayStep G η lam (U, V)).2 +
      η / 2 * frobeniusSq (lift lam (G (U * Vᴴ)) * stack U V) ≤ decayedLoss L lam U V :=
  decayedLoss_decayStep_add_le hη hγ0 hs0 (hγ _ (frobeniusSq_mul_conjTranspose_le hw)) hs hw
    (frobeniusSq_decayStep_le hη hlam hγ hb hρ hstep hw) hdesc

/-- **Every step of a run from the ball lowers the loss** (`review/flat_spectrum.md`, item
WD8), by at least `(η/2) ‖M_t W_t‖²`, under the bounds of
`decayedLoss_decayStep_add_le_of_ball`. -/
theorem decayedLoss_decayRun_succ_add_le (hη : 0 ≤ η) (hlam : 0 < lam) (hγ0 : 0 ≤ γ)
    (hs0 : 0 ≤ s) (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hρ : 4 * b ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam)
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    decayedLoss L lam (decayRun G η lam U₀ V₀ (t + 1)).1 (decayRun G η lam U₀ V₀ (t + 1)).2 +
      η / 2 * frobeniusSq (lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
        (decayRun G η lam U₀ V₀ t).2ᴴ)) *
          stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2) ≤
      decayedLoss L lam (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2 := by
  rw [decayRun, Function.iterate_succ_apply', ← decayRun]
  exact decayedLoss_decayStep_add_le_of_ball hη hlam hγ0 hs0 hγ hb hs hρ hstep hdesc
    (frobeniusSq_decayRun_le hη hlam hγ hb hρ hstep hw t)

/-! The witness: the squared loss of the `1 × 1` kernel `1`, `λ = 1`, `γ = 2`, `b = 1/4`,
`s = 1`, `ρ = 2`, `η = 1/10`: `2 η (λ² + γ²) = 1` and `η (λ + γ + 2 s ρ) = 7/10`. -/

example := decayedLoss_decayStep_add_le_of_ball (L := fitLoss Set.univ 1) (η := 1 / 10)
  (by norm_num) one_pos (by norm_num) zero_le_one stepWitness_bound
  (fun X _ => stepWitness_low X) (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num)
  (by norm_num) (by norm_num) stepWitness_ball

example := decayedLoss_decayRun_succ_add_le (L := fitLoss Set.univ 1) (η := 1 / 10)
  (by norm_num) one_pos (by norm_num) zero_le_one stepWitness_bound
  (fun X _ => stepWitness_low X) (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num)
  (by norm_num) (by norm_num) stepWitness_ball 3

end Transformer.Grokking.FlatSpectrum.Training
