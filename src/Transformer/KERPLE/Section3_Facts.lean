module

public import Mathlib.Algebra.QuadraticDiscriminant
public import Transformer.KERPLE.Section3_Kernels

/-!
# §3.2 Facts 2 and 4: scaling, sums and the negative squared distance

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.2.

* Fact 2 (scaling and summation): if `k̃₁` and `k̃₂` are CPD, so are `a k̃₁` for `a > 0` and
  `k̃₁ + k̃₂` (`IsCPDKernel.smul`, which allows `a = 0`, and `IsCPDKernel.add`). The same holds
  for PD kernels (`IsPDKernel.smul`, `IsPDKernel.add`).
* Fact 4 (page 3 of Schölkopf 2000): the negative squared distance `−‖x − x'‖²` is CPD, on every
  real inner product space (`isCPDKernel_neg_norm_sub_sq`). Its form is `2 ‖Σ_i c_i x_i‖²`.
* A PD kernel is nonnegative on the diagonal and has `k(x, y)² ≤ k(x, x) k(y, y)`
  (`IsPDKernel.diag_nonneg`, `IsPDKernel.sq_le_mul`): the two-point case of Definition 1.
* A CPD kernel vanishing on the diagonal is `≤ 0` (`IsCPDKernel.nonpos_of_diag`): Facts 1 and 3
  ask for the range `(−∞, 0]` beside `k̃(x, x) = 0`, and the first follows from the second.
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset

variable {X : Type*}

/-- A sum `Σ_i Σ_j c_i c_j f_i` factors as `(Σ_i c_i f_i)(Σ_j c_j)`. -/
theorem sum_sum_mul_left {ι : Type*} [Fintype ι] (f c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * f i = (∑ i, c i * f i) * ∑ j, c j := by
  rw [sum_mul_sum]
  exact sum_congr rfl fun i _ => sum_congr rfl fun j _ => by ring

/-- A sum `Σ_i Σ_j c_i c_j f_j` factors as `(Σ_i c_i)(Σ_j c_j f_j)`. -/
theorem sum_sum_mul_right {ι : Type*} [Fintype ι] (f c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * f j = (∑ i, c i) * ∑ j, c j * f j := by
  rw [sum_mul_sum]
  exact sum_congr rfl fun i _ => sum_congr rfl fun j _ => by ring

/-- The form of `a k` is `a` times the form of `k`. -/
theorem sum_sum_smul {ι : Type*} [Fintype ι] (k : X → X → ℝ) (a : ℝ) (x : ι → X) (c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * (a * k (x i) (x j)) = a * ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  rw [mul_sum]
  exact sum_congr rfl fun i _ => by rw [mul_sum]; exact sum_congr rfl fun j _ => by ring

/-- The form of `k₁ + k₂` is the sum of their forms. -/
theorem sum_sum_add {ι : Type*} [Fintype ι] (k₁ k₂ : X → X → ℝ) (x : ι → X) (c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * (k₁ (x i) (x j) + k₂ (x i) (x j)) =
      ∑ i, ∑ j, c i * c j * k₁ (x i) (x j) + ∑ i, ∑ j, c i * c j * k₂ (x i) (x j) := by
  simp only [mul_add, sum_add_distrib]

/-- **Fact 2, scaling.** If `k̃` is CPD, so is `a k̃` for `a ≥ 0` (the paper asks `a > 0`). -/
theorem IsCPDKernel.smul {k : X → X → ℝ} (hk : IsCPDKernel k) {a : ℝ} (ha : 0 ≤ a) :
    IsCPDKernel fun x y => a * k x y :=
  ⟨fun x y => congrArg (a * ·) (hk.1 x y), fun N x c hc => by
    rw [sum_sum_smul]
    exact mul_nonneg ha (hk.2 N x c hc)⟩

/-- **Fact 2, summation.** If `k̃₁` and `k̃₂` are CPD, so is `k̃₁ + k̃₂`. -/
theorem IsCPDKernel.add {k₁ k₂ : X → X → ℝ} (h₁ : IsCPDKernel k₁) (h₂ : IsCPDKernel k₂) :
    IsCPDKernel fun x y => k₁ x y + k₂ x y :=
  ⟨fun x y => congrArg₂ (· + ·) (h₁.1 x y) (h₂.1 x y), fun N x c hc => by
    rw [sum_sum_add]
    exact add_nonneg (h₁.2 N x c hc) (h₂.2 N x c hc)⟩

/-- PD kernels are closed under nonnegative scaling. -/
theorem IsPDKernel.smul {k : X → X → ℝ} (hk : IsPDKernel k) {a : ℝ} (ha : 0 ≤ a) :
    IsPDKernel fun x y => a * k x y :=
  ⟨fun x y => congrArg (a * ·) (hk.1 x y), fun N x c => by
    rw [sum_sum_smul]
    exact mul_nonneg ha (hk.2 N x c)⟩

/-- PD kernels are closed under sums. -/
theorem IsPDKernel.add {k₁ k₂ : X → X → ℝ} (h₁ : IsPDKernel k₁) (h₂ : IsPDKernel k₂) :
    IsPDKernel fun x y => k₁ x y + k₂ x y :=
  ⟨fun x y => congrArg₂ (· + ·) (h₁.1 x y) (h₂.1 x y), fun N x c => by
    rw [sum_sum_add]
    exact add_nonneg (h₁.2 N x c) (h₂.2 N x c)⟩

/-- A nonnegative constant is a PD kernel. -/
theorem isPDKernel_const {a : ℝ} (ha : 0 ≤ a) : IsPDKernel fun _ _ : X => a :=
  ⟨fun _ _ => rfl, fun N _ c => by rw [sum_sum_const]; positivity⟩

/-- The hypotheses of `IsCPDKernel.smul` and `IsCPDKernel.add` hold for constant kernels. -/
example : IsCPDKernel fun _ _ : ℕ => (2 : ℝ) * 1 + (-5 + 0) :=
  (((isPDKernel_const (X := ℕ) zero_le_one).isCPDKernel.smul zero_le_two).add
    ((isPDKernel_const (X := ℕ) le_rfl).isCPDKernel.add_const (-5)))

/-- A PD kernel is nonnegative on the diagonal. -/
theorem IsPDKernel.diag_nonneg {k : X → X → ℝ} (hk : IsPDKernel k) (x : X) : 0 ≤ k x x := by
  simpa using hk.2 1 (fun _ => x) (fun _ => 1)

/-- **Two points.** A PD kernel has `k(x, y)² ≤ k(x, x) k(y, y)`. -/
theorem IsPDKernel.sq_le_mul {k : X → X → ℝ} (hk : IsPDKernel k) (x y : X) :
    k x y ^ 2 ≤ k x x * k y y := by
  have h : ∀ t : ℝ, 0 ≤ k x x * (t * t) + 2 * k x y * t + k y y := fun t => by
    have := hk.2 2 ![x, y] ![t, 1]
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one] at this
    rw [hk.1 y x] at this
    linarith
  have := discrim_le_zero h
  rw [discrim] at this
  nlinarith

/-- The hypothesis of `IsPDKernel.sq_le_mul` holds for a nonnegative constant. -/
example : (1 : ℝ) ^ 2 ≤ 1 * 1 := (isPDKernel_const (X := ℕ) zero_le_one).sq_le_mul 0 1

/-- A CPD kernel vanishing on the diagonal is `≤ 0`: the coefficients `(1, −1)` at `x, y` give
`−2 k̃(x, y) ≥ 0`. -/
theorem IsCPDKernel.nonpos_of_diag {k : X → X → ℝ} (hk : IsCPDKernel k)
    (hdiag : ∀ x, k x x = 0) (x y : X) : k x y ≤ 0 := by
  have := hk.2 2 ![x, y] ![1, -1] (by simp)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one, hdiag] at this
  rw [hk.1 y x] at this
  linarith

/-- **Fact 4.** The negative squared distance `−‖x − x'‖²` is CPD on a real inner product space:
for `Σ_i c_i = 0` its form is `2 ‖Σ_i c_i x_i‖²`. -/
theorem isCPDKernel_neg_norm_sub_sq {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] :
    IsCPDKernel fun x y : E => -‖x - y‖ ^ 2 := by
  refine ⟨fun x y => congrArg (fun t => -t ^ 2) (norm_sub_rev x y), fun N x c hc => ?_⟩
  have hsplit : ∀ i j, c i * c j * -‖x i - x j‖ ^ 2 =
      2 * (c i * c j * inner ℝ (x i) (x j)) - c i * c j * ‖x i‖ ^ 2 -
        c i * c j * ‖x j‖ ^ 2 := fun i j => by
    rw [norm_sub_sq_real]
    ring
  have hgram : ∑ i, ∑ j, c i * c j * inner ℝ (x i) (x j) = ‖∑ i, c i • x i‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    refine sum_congr rfl fun i _ => ?_
    rw [inner_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [real_inner_smul_left, real_inner_smul_right]
    ring
  simp only [hsplit, sum_sub_distrib, ← mul_sum]
  rw [sum_sum_mul_left (fun i => ‖x i‖ ^ 2), sum_sum_mul_right (fun j => ‖x j‖ ^ 2), hc, hgram]
  simp only [mul_zero, zero_mul, sub_zero]
  positivity

end Transformer.KERPLE
