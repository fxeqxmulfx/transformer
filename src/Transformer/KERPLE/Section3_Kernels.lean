module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# §3.1 PD and CPD kernels

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, "KERPLE: Kernelized Relative Positional
Embedding for Length Extrapolation" (NeurIPS 2022), §3.1–3.2.

A kernel on `X` is a function `k : X → X → ℝ`, the paper's `X × X → ℝ`.

* Definition 1 (`IsPDKernel`): `k` is symmetric and `Σ_i Σ_j c_i c_j k(x_i, x_j) ≥ 0` for every
  `N`, points `x_1, …, x_N` and reals `c_1, …, c_N`. This is Mathlib's `Matrix.PosSemidef` of `k`
  read as an `X × X` matrix (`isPDKernel_iff_posSemidef`).
* Definition 2 (`IsCPDKernel`): the same for the coefficients with `Σ_i c_i = 0`.
* "If one takes any PD kernel and offsets it by a constant, the result is at least a CPD
  kernel" (`IsPDKernel.isCPDKernel_add_const`): a PD kernel is CPD, and a constant does not
  change a form whose coefficients sum to `0` (`IsCPDKernel.add_const`).
* The forms are taken over `Fin N`; over any finite index type they are nonnegative as well
  (`IsPDKernel.sum_nonneg`, `IsCPDKernel.sum_nonneg`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset

variable {X : Type*}

/-- **Definition 1** (PD kernel). A real symmetric function `k : X × X → ℝ` is a positive definite
kernel if for every integer `N`, points `x_1, …, x_N` and reals `c_1, …, c_N` the quadratic form is
nonnegative: `Σ_i Σ_j c_i c_j k(x_i, x_j) ≥ 0` (§3.1). -/
def IsPDKernel (k : X → X → ℝ) : Prop :=
  (∀ x y, k x y = k y x) ∧
    ∀ (N : ℕ) (x : Fin N → X) (c : Fin N → ℝ), 0 ≤ ∑ i, ∑ j, c i * c j * k (x i) (x j)

/-- **Definition 2** (CPD kernel). A real symmetric function `k̃ : X × X → ℝ` is a conditionally
positive definite kernel if for every integer `N` and points `x_1, …, x_N` the quadratic form is
nonnegative for the reals `c_1, …, c_N` with `Σ_i c_i = 0` (§3.1). -/
def IsCPDKernel (k : X → X → ℝ) : Prop :=
  (∀ x y, k x y = k y x) ∧
    ∀ (N : ℕ) (x : Fin N → X) (c : Fin N → ℝ), ∑ i, c i = 0 →
      0 ≤ ∑ i, ∑ j, c i * c j * k (x i) (x j)

/-- Reindexing a quadratic form along an equivalence of finite index types. -/
theorem sum_sum_equiv {ι κ : Type*} [Fintype ι] [Fintype κ] (e : κ ≃ ι) (k : X → X → ℝ)
    (x : ι → X) (c : ι → ℝ) :
    ∑ i, ∑ j, c (e i) * c (e j) * k (x (e i)) (x (e j)) =
      ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  rw [e.sum_comp (fun i => ∑ j, c i * c (e j) * k (x i) (x (e j)))]
  exact sum_congr rfl fun i _ => e.sum_comp (fun j => c i * c j * k (x i) (x j))

/-- The form of a PD kernel is nonnegative on families indexed by any finite type. -/
theorem IsPDKernel.sum_nonneg {k : X → X → ℝ} (hk : IsPDKernel k) {ι : Type*} [Fintype ι]
    (x : ι → X) (c : ι → ℝ) : 0 ≤ ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  rw [← sum_sum_equiv (Fintype.equivFin ι).symm]
  exact hk.2 _ _ _

/-- The form of a CPD kernel is nonnegative on zero-sum families indexed by any finite type. -/
theorem IsCPDKernel.sum_nonneg {k : X → X → ℝ} (hk : IsCPDKernel k) {ι : Type*} [Fintype ι]
    (x : ι → X) {c : ι → ℝ} (hc : ∑ i, c i = 0) : 0 ≤ ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  rw [← sum_sum_equiv (Fintype.equivFin ι).symm]
  refine hk.2 _ _ _ ?_
  rwa [(Fintype.equivFin ι).symm.sum_comp c]

/-- The form `star c ⬝ᵥ M c` of the matrix of `k` at the points `x` is the paper's double sum. -/
theorem dotProduct_mulVec_submatrix {ι : Type*} [Fintype ι] (k : X → X → ℝ) (x : ι → X)
    (c : ι → ℝ) :
    star c ⬝ᵥ ((Matrix.of k).submatrix x x).mulVec c = ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  simp only [dotProduct, Matrix.mulVec, Matrix.submatrix_apply, Matrix.of_apply, Pi.star_apply,
    star_trivial, mul_sum]
  exact sum_congr rfl fun i _ => sum_congr rfl fun j _ => by ring

/-- **Definition 1 is Mathlib's positive semidefiniteness.** `k` is a PD kernel iff the matrix
`(k(x, y))_{x, y ∈ X}` is positive semidefinite, a condition on its finite submatrices. -/
theorem isPDKernel_iff_posSemidef (k : X → X → ℝ) :
    IsPDKernel k ↔ (Matrix.of k).PosSemidef := by
  classical
  constructor
  · intro hk
    rw [Matrix.posSemidef_iff_submatrix_finset]
    intro s
    refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun v => ?_
    · ext i j
      simp [Matrix.conjTranspose_apply, hk.1 i j]
    · rw [dotProduct_mulVec_submatrix]
      exact hk.sum_nonneg _ v
  · intro hM
    refine ⟨fun x y => ?_, fun N x c => ?_⟩
    · simpa using (hM.isHermitian.apply x y).symm
    · rw [← dotProduct_mulVec_submatrix]
      exact (hM.submatrix x).dotProduct_mulVec_nonneg c

/-- A PD kernel is CPD. -/
theorem IsPDKernel.isCPDKernel {k : X → X → ℝ} (hk : IsPDKernel k) : IsCPDKernel k :=
  ⟨hk.1, fun N x c _ => hk.2 N x c⟩

/-- The constant `c` contributes `c (Σ_i c_i)²` to a quadratic form. -/
theorem sum_sum_const {ι : Type*} [Fintype ι] (a : ℝ) (c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * a = a * (∑ i, c i) ^ 2 := by
  rw [sq, sum_mul_sum, mul_sum]
  exact sum_congr rfl fun i _ => by rw [mul_sum]; exact sum_congr rfl fun j _ => by ring

/-- The form of `k + c` is that of `k` plus `c (Σ_i c_i)²`. -/
theorem sum_sum_add_const {ι : Type*} [Fintype ι] (k : X → X → ℝ) (a : ℝ) (x : ι → X)
    (c : ι → ℝ) :
    ∑ i, ∑ j, c i * c j * (a + k (x i) (x j)) =
      a * (∑ i, c i) ^ 2 + ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  rw [← sum_sum_const, ← sum_add_distrib]
  exact sum_congr rfl fun i _ => by rw [← sum_add_distrib]; exact sum_congr rfl fun j _ => by ring

/-- A constant offset keeps a kernel CPD. -/
theorem IsCPDKernel.add_const {k : X → X → ℝ} (hk : IsCPDKernel k) (a : ℝ) :
    IsCPDKernel fun x y => a + k x y := by
  refine ⟨fun x y => congrArg (a + ·) (hk.1 x y), fun N x c hc => ?_⟩
  rw [sum_sum_add_const, hc]
  simpa using hk.2 N x c hc

/-- **§3.1.** "If one takes any PD kernel and offsets it by a constant, the result is at least a
CPD kernel." -/
theorem IsPDKernel.isCPDKernel_add_const {k : X → X → ℝ} (hk : IsPDKernel k) (a : ℝ) :
    IsCPDKernel fun x y => a + k x y :=
  hk.isCPDKernel.add_const a

/-- The hypothesis of `IsPDKernel.isCPDKernel_add_const` holds for the zero kernel. -/
example : IsCPDKernel fun _ _ : ℕ => (-3 : ℝ) + 0 :=
  IsPDKernel.isCPDKernel_add_const (k := fun _ _ : ℕ => (0 : ℝ))
    ⟨fun _ _ => rfl, fun _ _ _ => by simp⟩ (-3)

end Transformer.KERPLE
