module

public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
public import Transformer.KERPLE.Section4_Attention

/-!
# §4: the bias kernels of KERPLE are not PD

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §4, below Eq. 2: "The second equality defines a
*bias kernel* which is positive definite using Lemma 1: `k_{r_1,…,r_ℓ} = c + k̃_{r_1,…,r_ℓ}`"
(Eq. 3). Lemma 1 is false (`not_cpd_shift_lemma`), and so is this claim for both variants of §4.

* A PD kernel `c + k̃` with `k̃(x, x) = 0` has `k̃ ≥ −2c` (`IsPDKernel.neg_two_mul_le`), by its
  `2 × 2` minors. Both variants are unbounded below on the positions `ℕ`, so no real `c` makes
  either bias kernel PD there (`not_isPDKernel_power_bias`, `not_isPDKernel_log_bias`).
* On the positions `0, …, L` a PD bias kernel needs `c ≥ r₁ L^{r₂} / 2`, respectively
  `c ≥ r₁ log(1 + r₂ L) / 2` (`le_of_isPDKernel_power_bias`, `le_of_isPDKernel_log_bias`): no
  single `c` serves every length, which length extrapolation would need.
* At `r₂ = 2` no real `c` makes the power variant's bias kernel PD on `L ≥ 3` positions
  (`not_isPDKernel_power_bias_two`): there `−r₁(m − n)²` is CPD but not strictly CPD.
* The attention never uses `c` (`attention_eq_composite`): the claim concerns the kernel reading of
  Eq. 2, not the model trained in §5.
-/

@[expose] public section

namespace Transformer.KERPLE

open Filter

variable {X : Type*}

/-- **A PD kernel `c + k̃` bounds `k̃` below**: if `c + k̃` is PD and `k̃` vanishes on the diagonal,
then `k̃ ≥ −2c`, since the `2 × 2` minors give `(c + k̃(x, y))² ≤ c²`. -/
theorem IsPDKernel.neg_two_mul_le {k : X → X → ℝ} {c : ℝ} (h : IsPDKernel (biasKernel c k))
    (hdiag : ∀ x, k x x = 0) (x y : X) : -(2 * c) ≤ k x y := by
  have h1 := h.sq_le_mul x y
  have h0 := h.diag_nonneg x
  simp only [biasKernel, hdiag, add_zero] at h1 h0
  nlinarith

/-- On the two positions `0, 1` the bias kernel `1 − |m − n|` is PD: it is the identity. -/
theorem isPDKernel_bias_two_positions :
    IsPDKernel fun m n : Fin 2 => biasKernel 1 (powerKernel 1 1) (m : ℕ) (n : ℕ) := by
  have h : Matrix.of (fun m n : Fin 2 => biasKernel 1 (powerKernel 1 1) (m : ℕ) (n : ℕ)) = 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [biasKernel, powerKernel]
  rw [isPDKernel_iff_posSemidef, h]
  exact Matrix.PosSemidef.one

/-- The hypotheses of `IsPDKernel.neg_two_mul_le` hold for `1 − |m − n|` on two positions. -/
example (m n : Fin 2) : -(2 * 1) ≤ powerKernel 1 1 (m : ℕ) (n : ℕ) :=
  IsPDKernel.neg_two_mul_le (k := fun m n : Fin 2 => powerKernel 1 1 (m : ℕ) (n : ℕ))
    isPDKernel_bias_two_positions (fun x => by simp [powerKernel]) m n

/-- **The power variant's bias kernel is PD for no `c`** on the positions `ℕ`, for `r₁, r₂ > 0`,
against §4's claim that the bias kernel `c + k̃` "is positive definite using Lemma 1". -/
theorem not_isPDKernel_power_bias {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (c : ℝ) :
    ¬IsPDKernel fun m n : ℕ => biasKernel c (powerKernel r₁ r₂) m n := by
  intro h
  obtain ⟨N, hN⟩ := ((((tendsto_rpow_atTop hr₂).comp tendsto_natCast_atTop_atTop).const_mul_atTop
    hr₁).eventually_gt_atTop (2 * c)).exists
  have := IsPDKernel.neg_two_mul_le (k := fun m n : ℕ => powerKernel r₁ r₂ m n) h
    (fun x => by simp [powerKernel, hr₂.ne']) N 0
  simp only [powerKernel, Nat.cast_zero, sub_zero, Nat.abs_cast] at this
  simp only [Function.comp_apply] at hN
  linarith

/-- The hypotheses of `not_isPDKernel_power_bias` hold for ALiBi's `r₂ = 1`. -/
example : ¬IsPDKernel fun m n : ℕ => biasKernel 5 (powerKernel 1 1) m n :=
  not_isPDKernel_power_bias one_pos one_pos 5

/-- **The logarithmic variant's bias kernel is PD for no `c`** on the positions `ℕ`, for
`r₁, r₂ > 0`, against §4's claim that the bias kernel `c + k̃` "is positive definite using
Lemma 1". -/
theorem not_isPDKernel_log_bias {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (c : ℝ) :
    ¬IsPDKernel fun m n : ℕ => biasKernel c (logKernel r₁ r₂) m n := by
  intro h
  have hlim : Tendsto (fun N : ℕ => r₁ * Real.log (1 + r₂ * N)) atTop atTop :=
    (Real.tendsto_log_atTop.comp (tendsto_atTop_add_const_left _ 1
      (tendsto_natCast_atTop_atTop.const_mul_atTop hr₂))).const_mul_atTop hr₁
  obtain ⟨N, hN⟩ := (hlim.eventually_gt_atTop (2 * c)).exists
  have := IsPDKernel.neg_two_mul_le (k := fun m n : ℕ => logKernel r₁ r₂ m n) h
    (fun x => by simp [logKernel]) N 0
  simp only [logKernel, Nat.cast_zero, sub_zero, Nat.abs_cast] at this
  linarith

/-- The hypotheses of `not_isPDKernel_log_bias` hold for `r₁ = r₂ = 1`. -/
example : ¬IsPDKernel fun m n : ℕ => biasKernel 5 (logKernel 1 1) m n :=
  not_isPDKernel_log_bias one_pos one_pos 5

/-- **The constant grows with the length** (power variant): if the bias kernel `c + k̃` is PD on
the positions `0, …, L`, then `c ≥ r₁ L^{r₂} / 2`. -/
theorem le_of_isPDKernel_power_bias {r₁ r₂ c : ℝ} (hr₂ : 0 < r₂) {L : ℕ}
    (h : IsPDKernel fun m n : Fin (L + 1) => biasKernel c (powerKernel r₁ r₂) (m : ℕ) (n : ℕ)) :
    r₁ * (L : ℝ) ^ r₂ / 2 ≤ c := by
  have := IsPDKernel.neg_two_mul_le (k := fun m n : Fin (L + 1) => powerKernel r₁ r₂ (m : ℕ) n) h
    (fun x => by simp [powerKernel, hr₂.ne']) (Fin.last L) 0
  simp only [powerKernel, Fin.val_last, Fin.val_zero, Nat.cast_zero, sub_zero,
    Nat.abs_cast] at this
  linarith

/-- The hypotheses of `le_of_isPDKernel_power_bias` hold on two positions for `c = r₁ = r₂ = 1`. -/
example : (1 : ℝ) * ((1 : ℕ) : ℝ) ^ (1 : ℝ) / 2 ≤ 1 :=
  le_of_isPDKernel_power_bias (L := 1) one_pos isPDKernel_bias_two_positions

/-- **The constant grows with the length** (logarithmic variant): if the bias kernel `c + k̃` is PD
on the positions `0, …, L`, then `c ≥ r₁ log(1 + r₂ L) / 2`. -/
theorem le_of_isPDKernel_log_bias {r₁ r₂ c : ℝ} {L : ℕ}
    (h : IsPDKernel fun m n : Fin (L + 1) => biasKernel c (logKernel r₁ r₂) (m : ℕ) (n : ℕ)) :
    r₁ * Real.log (1 + r₂ * L) / 2 ≤ c := by
  have := IsPDKernel.neg_two_mul_le (k := fun m n : Fin (L + 1) => logKernel r₁ r₂ (m : ℕ) n) h
    (fun x => by simp [logKernel]) (Fin.last L) 0
  simp only [logKernel, Fin.val_last, Fin.val_zero, Nat.cast_zero, sub_zero,
    Nat.abs_cast] at this
  linarith

/-- The hypotheses of `le_of_isPDKernel_log_bias` hold on two positions for `c = r₂ = 1` and
`r₁ = 1 / log 2`, where the bias kernel is the identity. -/
example : (Real.log 2)⁻¹ * Real.log (1 + 1 * ((1 : ℕ) : ℝ)) / 2 ≤ 1 := by
  refine le_of_isPDKernel_log_bias (L := 1) ?_
  refine isPDKernel_bias_two_positions.congr fun m n => ?_
  have h2 : Real.log 2 ≠ 0 := (Real.log_pos one_lt_two).ne'
  fin_cases m <;> fin_cases n <;> norm_num [biasKernel, powerKernel, logKernel, h2]

/-- **At `r₂ = 2` the bias kernel is PD for no `c` on three positions**: for `r₁ > 0` and every
real `c`, `c − r₁(m − n)²` is not PD on `L ≥ 3` positions. At `0, 1, 2` the coefficients
`(r₁ + e, −2e, e)` give the form `r₁² (c − 4e)`, negative for `e = |c| + 1`. -/
theorem not_isPDKernel_power_bias_two {r₁ : ℝ} (hr₁ : 0 < r₁) (c : ℝ) {L : ℕ} (hL : 3 ≤ L) :
    ¬IsPDKernel fun m n : Fin L => biasKernel c (powerKernel r₁ 2) (m : ℕ) (n : ℕ) := by
  intro h
  have := h.2 3 (fun i => Fin.castLE hL i) ![r₁ + (|c| + 1), -2 * (|c| + 1), |c| + 1]
  simp only [biasKernel, powerKernel, Real.rpow_two, sq_abs, Fin.val_castLE, Fin.sum_univ_three,
    Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
    Matrix.head_cons, Fin.val_zero, Fin.val_one, Fin.val_two, Nat.cast_zero, Nat.cast_one,
    Nat.cast_ofNat] at this
  have hneg : r₁ ^ 2 * (c - 4 * (|c| + 1)) < 0 :=
    mul_neg_of_pos_of_neg (by positivity) (by linarith [le_abs_self c, abs_nonneg c])
  nlinarith

/-- The hypotheses of `not_isPDKernel_power_bias_two` hold on the `512` training positions of §5. -/
example : ¬IsPDKernel fun m n : Fin 512 => biasKernel 100 (powerKernel 1 2) (m : ℕ) (n : ℕ) :=
  not_isPDKernel_power_bias_two one_pos 100 (by norm_num)

end Transformer.KERPLE
