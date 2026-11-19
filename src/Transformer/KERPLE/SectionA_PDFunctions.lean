module

public import Mathlib.Analysis.SpecificLimits.Basic
public import Transformer.KERPLE.Section3_Powers

/-!
# Appendix A.2: positive definite functions and Fact 5

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.2 (Shift-invariant Kernels with
Bounded and Unbounded Ranges).

* Definition 3 (`IsPDFunction`): `f` is a PD function when every matrix `[f(x_i − x_j)]` is
  positive semidefinite, that is, when the shift-invariant kernel `f(x − y)` is PD (Definition 1):
  "they are equivalent by definition".
* Fact 5 (Corollary 3 of Schoenberg 1938): `exp(−|x|^p)` is PD if `0 < p ≤ 2`
  (`isPDFunction_exp_neg_rpow`, from Corollary 1(a) and Schoenberg's theorem) and not PD if
  `p > 2` (`not_isPDFunction_exp_neg_rpow`); both hold for `exp(−a|x|^p)` at every scale `a > 0`.
  For `p > 2` the coefficients `(1, −2, 1)` at `0, h, 2h` give `g(2h) ≤ 4 g(h)` for
  `g(t) = 1 − exp(−a t^p)`, hence `0 < g(1) ≤ a (4 · 2^{−p})^n → 0`.
* The footnote to the proof of Proposition 2, `exp(−λ|x|^p) = exp(−|λ^{1/p} x|^p)`
  (`exp_neg_mul_abs_rpow`), and the bounded range `(0, 1]` of these kernels
  (`exp_neg_mul_rpow_mem_Ioc`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Filter Topology

/-- **Definition 3** (positive definite function): every matrix `[f(x_i − x_j)]_{i,j=1}^N` is
positive semidefinite, that is, the shift-invariant kernel `(x, y) ↦ f(x − y)` is PD. -/
def IsPDFunction {G : Type*} [Sub G] (f : G → ℝ) : Prop := IsPDKernel fun x y => f (x - y)

/-- **Fact 5 for `0 < p ≤ 2`**, at every scale `a ≥ 0`: `exp(−a|x|^p)` is a PD function on `ℝ`,
by Schoenberg's theorem for the CPD kernel `−a|x − y|^p` of Corollary 1(a). -/
theorem isPDFunction_exp_neg_mul_rpow {a p : ℝ} (ha : 0 ≤ a) (hp : 0 < p) (hp2 : p ≤ 2) :
    IsPDFunction fun x : ℝ => Real.exp (-a * |x| ^ p) :=
  IsPDKernel.congr ((isCPDKernel_neg_mul_norm_rpow (E := ℝ) ha hp hp2).isPDKernel_exp_mul
    zero_le_one) fun x y => by rw [one_mul, Real.norm_eq_abs]

/-- **Fact 5, `0 < p ≤ 2`**: `exp(−|x|^p)` is a PD function. -/
theorem isPDFunction_exp_neg_rpow {p : ℝ} (hp : 0 < p) (hp2 : p ≤ 2) :
    IsPDFunction fun x : ℝ => Real.exp (-|x| ^ p) :=
  IsPDKernel.congr (isPDFunction_exp_neg_mul_rpow zero_le_one hp hp2) fun x y => by
    simp only [neg_one_mul]

/-- The hypotheses of `isPDFunction_exp_neg_rpow` hold for the Gaussian, `p = 2`. -/
example : IsPDFunction fun x : ℝ => Real.exp (-|x| ^ (2 : ℝ)) :=
  isPDFunction_exp_neg_rpow two_pos le_rfl

/-- **Fact 5 for `p > 2`**, at every scale `a > 0`: `exp(−a|x|^p)` is not a PD function on `ℝ`.
The coefficients `(1, −2, 1)` at `0, t, 2t` give `g(2t) ≤ 4 g(t)` for `g(t) = 1 − exp(−a t^p)`;
iterating from `t = 2^{−n}` and `g(t) ≤ a t^p` gives `0 < g(1) ≤ a (4 · 2^{−p})^n → 0`. -/
theorem not_isPDFunction_exp_neg_mul_rpow {a p : ℝ} (ha : 0 < a) (hp : 2 < p) :
    ¬IsPDFunction fun x : ℝ => Real.exp (-a * |x| ^ p) := by
  intro h
  have hp0 : p ≠ 0 := by linarith
  set g : ℝ → ℝ := fun t => 1 - Real.exp (-a * t ^ p) with hg
  have h3 : ∀ t, 0 ≤ t → g (2 * t) ≤ 4 * g t := fun t ht => by
    have := h.2 3 ![0, t, 2 * t] ![1, -2, 1]
    simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.tail_cons, Matrix.head_cons] at this
    rw [show t - 2 * t = -t by ring, show 2 * t - t = t by ring] at this
    simp only [sub_self, sub_zero, zero_sub, abs_neg, abs_zero, abs_of_nonneg ht,
      abs_of_nonneg (by linarith : (0 : ℝ) ≤ 2 * t), Real.zero_rpow hp0, mul_zero,
      Real.exp_zero] at this
    simp only [hg]
    linarith
  have hind : ∀ n : ℕ, g 1 ≤ 4 ^ n * g ((1 / 2) ^ n) := fun n => by
    induction n with
    | zero => simp
    | succ n ih =>
      have := h3 ((1 / 2) ^ (n + 1)) (by positivity)
      rw [show 2 * (1 / 2 : ℝ) ^ (n + 1) = (1 / 2) ^ n by rw [pow_succ]; ring] at this
      calc g 1 ≤ 4 ^ n * g ((1 / 2) ^ n) := ih
        _ ≤ 4 ^ n * (4 * g ((1 / 2) ^ (n + 1))) := by gcongr
        _ = 4 ^ (n + 1) * g ((1 / 2) ^ (n + 1)) := by ring
  have hgle : ∀ t, g t ≤ a * t ^ p := fun t => by
    have := Real.add_one_le_exp (-a * t ^ p)
    simp only [hg]
    linarith
  set q : ℝ := 4 * (1 / 2) ^ p
  have hq0 : 0 ≤ q := by positivity
  have hq1 : q < 1 := by
    have := Real.rpow_lt_rpow_of_exponent_gt (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num) hp
    rw [Real.rpow_two] at this
    linarith
  have hbound : ∀ n : ℕ, g 1 ≤ a * q ^ n := fun n => by
    have h1 := hgle ((1 / 2) ^ n)
    rw [← Real.rpow_pow_comm (by norm_num) p n] at h1
    calc g 1 ≤ 4 ^ n * g ((1 / 2) ^ n) := hind n
      _ ≤ 4 ^ n * (a * ((1 / 2) ^ p) ^ n) := by gcongr
      _ = a * q ^ n := by rw [mul_pow]; ring
  have hlim : Tendsto (fun n : ℕ => a * q ^ n) atTop (𝓝 (a * 0)) :=
    (tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1).const_mul a
  have hg1 : g 1 ≤ a * 0 := ge_of_tendsto' hlim hbound
  have hexp : Real.exp (-a) < 1 := Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr ha)
  simp only [hg, Real.one_rpow, mul_one, mul_zero] at hg1
  linarith

/-- **Fact 5, `p > 2`**: `exp(−|x|^p)` is not a PD function. -/
theorem not_isPDFunction_exp_neg_rpow {p : ℝ} (hp : 2 < p) :
    ¬IsPDFunction fun x : ℝ => Real.exp (-|x| ^ p) := fun h =>
  not_isPDFunction_exp_neg_mul_rpow one_pos hp
    (IsPDKernel.congr h fun x y => by simp only [neg_one_mul])

/-- The hypothesis of `not_isPDFunction_exp_neg_rpow` holds for `p = 3`. -/
example : ¬IsPDFunction fun x : ℝ => Real.exp (-|x| ^ (3 : ℝ)) :=
  not_isPDFunction_exp_neg_rpow (by norm_num)

/-- The footnote to the proof of Proposition 2: for `λ > 0` and `p ≠ 0`,
`exp(−λ|x|^p) = exp(−|λ^{1/p} x|^p)`, a rescaling of `exp(−|x|^p)`. -/
theorem exp_neg_mul_abs_rpow {l p : ℝ} (hl : 0 < l) (hp : p ≠ 0) (x : ℝ) :
    Real.exp (-l * |x| ^ p) = Real.exp (-|l ^ (1 / p) * x| ^ p) := by
  rw [abs_mul, abs_of_pos (Real.rpow_pos_of_pos hl _),
    Real.mul_rpow (Real.rpow_nonneg hl.le _) (abs_nonneg x), one_div, Real.rpow_inv_rpow hl.le hp,
    neg_mul]

/-- The hypotheses of `exp_neg_mul_abs_rpow` hold for `λ = 4`, `p = 2`. -/
example (x : ℝ) : Real.exp (-4 * |x| ^ (2 : ℝ)) = Real.exp (-|4 ^ (1 / (2 : ℝ)) * x| ^ (2 : ℝ)) :=
  exp_neg_mul_abs_rpow (by norm_num) two_ne_zero x

/-- The kernels of Fact 5 are bounded: `exp(−a|x|^p) ∈ (0, 1]` for `a ≥ 0`. -/
theorem exp_neg_mul_rpow_mem_Ioc {a : ℝ} (ha : 0 ≤ a) (p x : ℝ) :
    Real.exp (-a * |x| ^ p) ∈ Set.Ioc 0 1 :=
  ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr (by nlinarith [Real.rpow_nonneg (abs_nonneg x) p])⟩

/-- The hypothesis of `exp_neg_mul_rpow_mem_Ioc` holds for the Gaussian kernel, `a = 1`. -/
example (x : ℝ) : Real.exp (-1 * |x| ^ (2 : ℝ)) ∈ Set.Ioc 0 1 :=
  exp_neg_mul_rpow_mem_Ioc zero_le_one 2 x

end Transformer.KERPLE
