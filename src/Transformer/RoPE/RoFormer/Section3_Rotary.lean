module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Data.Matrix.Block
public import Mathlib.LinearAlgebra.UnitaryGroup

/-!
# §3.2 The rotary position embedding

Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864v5,
§3.2 and §3.4.2.

* `rotate` is the 2D solution of Eq. (12), `f(x, m) = x e^{imθ}` on `ℂ = ℝ²`;
  `inner_rotate`: `⟨f_q(x_m, m), f_k(x_n, n)⟩ = Re[q k* e^{i(m−n)θ}]`, the `g` of Eq. (12);
* `rotation` is the matrix of Eq. (13), and `rotation_mulVec` identifies it with
  multiplication by `e^{imθ}`;
* `rotary` is the block-diagonal `R^d_{Θ,m}` of Eq. (15), indexed by `Fin 2 × Fin (d/2)`:
  coordinate `(r, i)` is the source's `x_{2i+r+1}`;
* `rotary_transpose_mul`: `R_{n−m} = R_mᵀ R_n` (below Eq. (16));
  `rotary_mem_orthogonalGroup`: `R^d_Θ` is orthogonal;
* `score_eq`: Eq. (16), `q_mᵀ k_n = (R_m W_q x_m)ᵀ(R_n W_k x_n) = x_mᵀ W_qᵀ R_{n−m} W_k x_n`.
  The source prints `x^⊤ W_q R^d_{Θ,n−m} W_k x_n`; its left factor is `(W_q x_m)ᵀ`,
  which is `x_mᵀ W_qᵀ`;
* `rotary_mulVec`: §3.4.2, Eq. (34),
  `R x = x ⊗ (cos mθ_i)_i + (−x_2, x_1, …, −x_d, x_{d−1}) ⊗ (sin mθ_i)_i`.
-/

@[expose] public section

open scoped BigOperators ComplexConjugate InnerProductSpace
open Matrix

namespace Transformer.RoPE.RoFormer

/-- Eq. (12): the 2D rotary embedding `f(x, m) = x e^{imθ}` of a query or key `x ∈ ℂ = ℝ²`. -/
noncomputable def rotate (θ m : ℝ) (x : ℂ) : ℂ :=
  x * Complex.exp ((m * θ : ℝ) * Complex.I)

/-- Eq. (12): the inner product of the embedded query and key is
`g(x_m, x_n, m − n) = Re[q k* e^{i(m−n)θ}]`, a function of `m − n`. -/
theorem inner_rotate (θ m n : ℝ) (q k : ℂ) :
    ⟪rotate θ m q, rotate θ n k⟫_ℝ =
      (q * conj k * Complex.exp (((m - n) * θ : ℝ) * Complex.I)).re := by
  simp only [Complex.inner, rotate, Complex.exp_mul_I, ← Complex.ofReal_cos,
    ← Complex.ofReal_sin, sub_mul, Real.cos_sub, Real.sin_sub, Complex.mul_re, Complex.mul_im,
    Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im, Complex.I_re,
    Complex.I_im, Complex.conj_re, Complex.conj_im, map_mul, map_add, Complex.conj_ofReal,
    Complex.conj_I, Complex.neg_re, Complex.neg_im]
  ring

/-- Eq. (13): the rotation by the angle `φ`. -/
noncomputable def rotation (φ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos φ, -Real.sin φ; Real.sin φ, Real.cos φ]

/-- Eq. (13): the rotation matrix acts on the coordinates of `x` as `x ↦ x e^{imθ}`. -/
theorem rotation_mulVec (θ m : ℝ) (x : ℂ) :
    rotation (m * θ) *ᵥ ![x.re, x.im] = ![(rotate θ m x).re, (rotate θ m x).im] := by
  simp only [rotate, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  ext i
  fin_cases i <;>
    simp [rotation, Matrix.mulVec, dotProduct, Fin.sum_univ_two, Complex.mul_re,
      Complex.mul_im, -Complex.ofReal_cos, -Complex.ofReal_sin] <;> ring

theorem rotation_add (a b : ℝ) : rotation (a + b) = rotation a * rotation b := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rotation, Matrix.mul_apply, Fin.sum_univ_two, Real.cos_add, Real.sin_add] <;> ring

theorem rotation_transpose (a : ℝ) : (rotation a)ᵀ = rotation (-a) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rotation]

theorem rotation_zero : rotation 0 = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rotation]

/-- Eq. (15): the rotary matrix `R^d_{Θ,m}`, block diagonal with the blocks `rotation (m θ_i)`
for the `d/2` pairs. -/
noncomputable def rotary {p : ℕ} (θ : Fin p → ℝ) (m : ℝ) :
    Matrix (Fin 2 × Fin p) (Fin 2 × Fin p) ℝ :=
  blockDiagonal fun i => rotation (m * θ i)

variable {p : ℕ}

theorem rotary_add (θ : Fin p → ℝ) (m n : ℝ) : rotary θ (m + n) = rotary θ m * rotary θ n := by
  simp only [rotary, ← blockDiagonal_mul, add_mul, rotation_add]

theorem rotary_transpose (θ : Fin p → ℝ) (m : ℝ) : (rotary θ m)ᵀ = rotary θ (-m) := by
  simp only [rotary, blockDiagonal_transpose, rotation_transpose, neg_mul]

theorem rotary_zero (θ : Fin p → ℝ) : rotary θ 0 = 1 := by
  simp only [rotary, zero_mul, rotation_zero]
  exact blockDiagonal_one

/-- Below Eq. (16): `R^d_{Θ,n−m} = (R^d_{Θ,m})ᵀ R^d_{Θ,n}`. -/
theorem rotary_transpose_mul (θ : Fin p → ℝ) (m n : ℝ) :
    (rotary θ m)ᵀ * rotary θ n = rotary θ (n - m) := by
  rw [rotary_transpose, ← rotary_add, neg_add_eq_sub]

/-- Below Eq. (16): "`R^d_Θ` is an orthogonal matrix". -/
theorem rotary_mem_orthogonalGroup (θ : Fin p → ℝ) (m : ℝ) :
    rotary θ m ∈ orthogonalGroup (Fin 2 × Fin p) ℝ := by
  rw [mem_orthogonalGroup_iff, rotary_transpose, ← rotary_add, add_neg_cancel, rotary_zero]

theorem mulVec_dotProduct {ι κ : Type*} [Fintype ι] [Fintype κ] (A : Matrix ι κ ℝ) (x : κ → ℝ)
    (w : ι → ℝ) : (A *ᵥ x) ⬝ᵥ w = x ⬝ᵥ (Aᵀ *ᵥ w) := by
  rw [dotProduct_comm, dotProduct_mulVec, mulVec_transpose, dotProduct_comm]

/-- Eq. (16): `q_mᵀ k_n = (R^d_{Θ,m} W_q x_m)ᵀ(R^d_{Θ,n} W_k x_n) = x_mᵀ W_qᵀ R^d_{Θ,n−m} W_k x_n`,
a function of the relative position `n − m` (the source prints the left factor as `x^⊤ W_q`). -/
theorem score_eq {e : ℕ} (θ : Fin p → ℝ) (Wq Wk : Matrix (Fin 2 × Fin p) (Fin e) ℝ)
    (xm xn : Fin e → ℝ) (m n : ℝ) :
    (rotary θ m *ᵥ (Wq *ᵥ xm)) ⬝ᵥ (rotary θ n *ᵥ (Wk *ᵥ xn)) =
      xm ⬝ᵥ ((Wqᵀ * rotary θ (n - m) * Wk) *ᵥ xn) := by
  rw [mulVec_mulVec, mulVec_mulVec, mulVec_dotProduct, mulVec_mulVec, transpose_mul,
    Matrix.mul_assoc, ← Matrix.mul_assoc (rotary θ m)ᵀ, rotary_transpose_mul, ← Matrix.mul_assoc]

/-- The rotary matrix keeps lengths: `(R x)ᵀ(R x) = xᵀx`. -/
theorem dotProduct_rotary_self (θ : Fin p → ℝ) (m : ℝ) (x : Fin 2 × Fin p → ℝ) :
    (rotary θ m *ᵥ x) ⬝ᵥ (rotary θ m *ᵥ x) = x ⬝ᵥ x := by
  rw [mulVec_dotProduct, mulVec_mulVec, rotary_transpose_mul, sub_self, rotary_zero, one_mulVec]

/-- §3.4.2: `(−x_2, x_1, −x_4, x_3, …, −x_d, x_{d−1})`. -/
def rotateHalf (x : Fin 2 × Fin p → ℝ) : Fin 2 × Fin p → ℝ :=
  fun ri => if ri.1 = 0 then -x (1, ri.2) else x (0, ri.2)

/-- §3.4.2, Eq. (34): the sparse product
`R^d_{Θ,m} x = x ⊗ (cos mθ_i)_i + rotateHalf x ⊗ (sin mθ_i)_i`. -/
theorem rotary_mulVec (θ : Fin p → ℝ) (m : ℝ) (x : Fin 2 × Fin p → ℝ) :
    rotary θ m *ᵥ x = fun ri =>
      x ri * Real.cos (m * θ ri.2) + rotateHalf x ri * Real.sin (m * θ ri.2) := by
  ext ⟨r, i⟩
  simp only [rotary, Matrix.mulVec, dotProduct, Fintype.sum_prod_type, blockDiagonal_apply,
    ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true, Fin.sum_univ_two]
  fin_cases r <;> simp [rotation, rotateHalf] <;> ring

end Transformer.RoPE.RoFormer
