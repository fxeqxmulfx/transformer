module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitBall

/-!
# Logits that are quadratic in the product: runs that do not end at a decayed minimum

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"Logits that are not linear in the product", and item XE14. For logits that are real-linear
in the product, XE3 says: the run converges, and "from a start of full row rank the limit is a
decayed minimum". For the logits `±(Re x)²` of `SquaredLogit` the second statement fails.

Let `0 < λ ≤ 1/4` and `0 < η` with `η λ ≤ 1/2`. In the ball `‖U‖² + ‖V‖² ≤ λ/4` the gradient
is small, `‖G‖ ≤ ‖U‖² + ‖V‖²`, so the weight decay wins: a step multiplies `‖U‖² + ‖V‖²` by
at most `1 − η λ / 4` (`frobeniusSq_decayStep_sqLogit_le`). Hence every run from the ball, of
any width and rank, converges to the zero factorization (`tendsto_decayRun_sqLogit_zero`), and
its step matrices are positive definite (`one_sub_smul_lift_posDef_sqLogit`), as WD5 asks. The
zero factorization is not a decayed minimum: its weight-decayed loss is `log 2`, and the
factorization `(3/2) · (4/3)` of the product `2` has less
(`not_isDecayedMin_sqLogit_zero`). So

    every run from the ball converges, and its limit is not a decayed minimum

(`exists_tendsto_decayRun_sqLogit_not_isDecayedMin`). The ball contains starts of full row
rank, with a product that is not zero (the last example). The loss meets every hypothesis of
WD5, WD9 and XE3 but one: it is not convex (`not_first_order_sqLogit`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {k : Type*} [Fintype k] {η lam : ℝ} {U V U₀ V₀ : Matrix (Fin 1) k ℂ}

/-- **Every run from the ball converges to the zero factorization**, whatever its width and
rank. -/
theorem tendsto_decayRun_sqLogit_zero (hη : 0 < η) (hlam : 0 < lam) (hstep : η * lam ≤ 1 / 2)
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ lam / 4) :
    Tendsto (decayRun sqLogitGrad η lam U₀ V₀) atTop (𝓝 (0, 0)) := by
  have hq : |1 - η * lam / 4| < 1 := abs_lt.2 ⟨by linarith, by linarith [mul_pos hη hlam]⟩
  have hpow : Tendsto (fun t : ℕ => (1 - η * lam / 4) ^ t * (frobeniusSq U₀ + frobeniusSq V₀))
      atTop (𝓝 0) := by
    simpa using (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hq).mul_const
      (frobeniusSq U₀ + frobeniusSq V₀)
  have hsum := squeeze_zero
    (fun t => add_nonneg (frobeniusSq_nonneg (decayRun sqLogitGrad η lam U₀ V₀ t).1)
      (frobeniusSq_nonneg (decayRun sqLogitGrad η lam U₀ V₀ t).2))
    (frobeniusSq_decayRun_sqLogit_le hη.le hlam hstep hw) hpow
  have h1 := squeeze_zero (fun t => frobeniusSq_nonneg (decayRun sqLogitGrad η lam U₀ V₀ t).1)
    (fun t => le_add_of_nonneg_right (frobeniusSq_nonneg _)) hsum
  have h2 := squeeze_zero (fun t => frobeniusSq_nonneg (decayRun sqLogitGrad η lam U₀ V₀ t).2)
    (fun t => le_add_of_nonneg_left (frobeniusSq_nonneg _)) hsum
  exact (tendsto_zero_of_tendsto_frobeniusSq h1).prodMk_nhds
    (tendsto_zero_of_tendsto_frobeniusSq h2)

/-- In the ball every step matrix is positive definite: the steps are those of WD5. -/
theorem one_sub_smul_lift_posDef_sqLogit (hη : 0 ≤ η) (hlam : 0 < lam)
    (hstep : η * lam ≤ 1 / 2) (hw : frobeniusSq U + frobeniusSq V ≤ lam / 4) :
    (1 - η • lift lam (sqLogitGrad (U * Vᴴ))).PosDef := by
  refine one_sub_smul_lift_posDef hη (by positivity) (frobeniusSq_sqLogitGrad_mul_le hw) ?_
  rw [abs_of_pos hlam]
  nlinarith

/-- **The zero factorization is not a decayed minimum** for a weight decay `λ ≤ 1/4`: the
factorization `(3/2) · (4/3)` of the product `2` has less weight-decayed loss than `log 2`. -/
theorem not_isDecayedMin_sqLogit_zero (hlam : lam ≤ 1 / 4) :
    ¬ IsDecayedMin sqLogitLoss lam (0 : Matrix (Fin 1) k ℂ) (0 : Matrix (Fin 1) k ℂ) :=
  fun hmin => by
    have h := hmin 1 ((3 / 2 : ℝ) • 1) ((4 / 3 : ℝ) • 1)
    have hre : ((((3 / 2 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) *
        ((4 / 3 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ))ᴴ) 0 0).re = 2 := by
      simp
      norm_num
    have hone : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 := by simp [frobeniusSq]
    have h0 : decayedLoss sqLogitLoss lam (0 : Matrix (Fin 1) k ℂ) 0 = Real.log 2 := by
      simp [decayedLoss, sqLogitLoss_zero, frobeniusSq]
    rw [h0, decayedLoss, sqLogitLoss, hre, frobeniusSq_smul, frobeniusSq_smul, hone] at h
    have he : Real.exp (-(2 * (2 : ℝ) ^ 2)) ≤ 1 / 9 := by
      rw [Real.exp_neg, inv_eq_one_div]
      exact one_div_le_one_div_of_le (by norm_num)
        (by linarith [Real.add_one_le_exp (2 * (2 : ℝ) ^ 2)])
    have hlog := Real.log_le_sub_one_of_pos
      (show (0 : ℝ) < 1 + Real.exp (-(2 * (2 : ℝ) ^ 2)) by positivity)
    have h2 := Real.log_two_gt_d9
    have hl := mul_le_mul_of_nonneg_right hlam
      (show (0 : ℝ) ≤ ((3 / 2) ^ 2 * 1 + (4 / 3) ^ 2 * 1) / 2 by norm_num)
    norm_num at h hl he hlog h2
    linarith

/-- **Without linear logits the limit of a run need not be a decayed minimum**
(`review/flat_spectrum.md`, item XE14): for the logits `±(Re x)²`, a weight decay
`0 < λ ≤ 1/4` and a step `0 < η` with `η λ ≤ 1/2`, every run from the ball
`‖U₀‖² + ‖V₀‖² ≤ λ/4` converges, and its limit is not a decayed minimum. -/
theorem exists_tendsto_decayRun_sqLogit_not_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (hlam' : lam ≤ 1 / 4) (hstep : η * lam ≤ 1 / 2)
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ lam / 4) :
    ∃ U V : Matrix (Fin 1) k ℂ, Tendsto (decayRun sqLogitGrad η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
      ¬ IsDecayedMin sqLogitLoss lam U V :=
  ⟨0, 0, tendsto_decayRun_sqLogit_zero hη hlam hstep hw, not_isDecayedMin_sqLogit_zero hlam'⟩

/-! The witnesses: `λ = 1/4`, `η = 1`, and the start `U₀ = [1/8 0]`, `V₀ = [1/8 1/8]` of width
two. It lies in the ball, `3/64 ≤ 1/16`, it has full row rank, and its product is `1/64`. -/

/-- The start of the examples. -/
example : frobeniusSq (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ) +
    frobeniusSq (!![1 / 8, 1 / 8] : Matrix (Fin 1) (Fin 2) ℂ) ≤ (1 / 4 : ℝ) / 4 := by
  norm_num [frobeniusSq, Fin.sum_univ_two]

example : ∃ U V : Matrix (Fin 1) (Fin 2) ℂ, Tendsto (decayRun sqLogitGrad 1 (1 / 4)
    (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ) !![1 / 8, 1 / 8]) atTop (𝓝 (U, V)) ∧
    ¬ IsDecayedMin sqLogitLoss (1 / 4) U V :=
  exists_tendsto_decayRun_sqLogit_not_isDecayedMin one_pos (by norm_num) le_rfl (by norm_num)
    (by norm_num [frobeniusSq, Fin.sum_univ_two])

/-- The start has full row rank and a product that is not zero. -/
example : Function.Surjective (stack (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ)
    !![1 / 8, 1 / 8]).mulVec ∧
    (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ) * (!![1 / 8, 1 / 8] : Matrix (Fin 1) (Fin 2) ℂ)ᴴ
      ≠ 0 := by
  refine ⟨fun v => ⟨![8 * v (Sum.inl 0), 8 * v (Sum.inr 0) - 8 * v (Sum.inl 0)], ?_⟩, fun h => ?_⟩
  · ext (i | i) <;> fin_cases i
    · simp [stack, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    · simp [stack, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
      ring
  · have h0 := congrFun (congrFun h 0) 0
    simp [Matrix.mul_apply, Fin.sum_univ_two] at h0

example := tendsto_decayRun_sqLogit_zero (η := 1) (lam := 1 / 4) one_pos (by norm_num)
  (by norm_num) (U₀ := (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ)) (V₀ := !![1 / 8, 1 / 8])
  (by norm_num [frobeniusSq, Fin.sum_univ_two])

example := one_sub_smul_lift_posDef_sqLogit (η := 1) (lam := 1 / 4) zero_le_one (by norm_num)
  (by norm_num) (U := (!![1 / 8, 0] : Matrix (Fin 1) (Fin 2) ℂ)) (V := !![1 / 8, 1 / 8])
  (by norm_num [frobeniusSq, Fin.sum_univ_two])

example := not_isDecayedMin_sqLogit_zero (k := Fin 2) (lam := 1 / 4) le_rfl

end Transformer.Grokking.FlatSpectrum.Training.Scope
