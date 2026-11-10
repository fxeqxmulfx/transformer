module

public import Transformer.Grokking.FlatSpectrum.WeightedPhases
public import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Commute

/-!
# The polar factor of `c + A`, `A` skew-Hermitian

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW2.
Let `K + Kᴴ = 2c` with `c > 0`: `K = c + A` with `Aᴴ = −A`. Then `K` is invertible
(`det_ne_zero_of_add_conjTranspose`), as `Re ⟨v, K v⟩ = c ‖v‖²`, and normal. Its polar factor
is `W = K S`, `S = (Kᴴ K)^{-1/2}` (`gramInvSqrt`), and `S` commutes with `K`, so
`W + Wᴴ = (K + Kᴴ) S = 2c S` (`polarFactor_add_conjTranspose`). `S` is positive definite: in
the eigenbasis `U` of `Kᴴ K`, `S_ii = ∑_k |U_ik|² / σ_k > 0` (`gramInvSqrt_apply_self_pos`). So
`Re W_ii = c S_ii > 0` (`re_polarFactor_apply_self`), and by Theorem A such a `K` is not a
minimum of the nuclear norm among the matrices equal to it on `Ω` once `Ω` misses a diagonal
cell (`not_isMinOn_of_add_conjTranspose`). The identity matrix is the case `A = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `(Kᴴ K)^{-1/2}`, the second factor of the polar factor `K (Kᴴ K)^{-1/2}`. -/
noncomputable def gramInvSqrt (K : Matrix n n ℂ) : Matrix n n ℂ :=
  cfc (fun x : ℝ => (√x)⁻¹) (Kᴴ * K)

lemma polarFactor_eq_mul_gramInvSqrt (K : Matrix n n ℂ) :
    polarFactor K = K * gramInvSqrt K := rfl

/-- In the eigenbasis `U` of `Kᴴ K`, `(Kᴴ K)^{-1/2} = U diag (σ⁻¹) Uᴴ`. -/
lemma gramInvSqrt_eq (K : Matrix n n ℂ) :
    gramInvSqrt K =
      ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) *
        diagonal (fun i => (((singularValues K i)⁻¹ : ℝ) : ℂ)) *
        ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ)ᴴ := by
  rw [gramInvSqrt, (isHermitian_conjTranspose_mul_self K).cfc_eq, IsHermitian.cfc,
    Unitary.conjStarAlgAut_apply]
  simp only [Matrix.mul_assoc, Matrix.star_eq_conjTranspose]
  rfl

lemma conjTranspose_gramInvSqrt (K : Matrix n n ℂ) : (gramInvSqrt K)ᴴ = gramInvSqrt K := by
  rw [gramInvSqrt_eq, conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose,
    conjTranspose_diagonal_ofReal, Matrix.mul_assoc]

/-- The diagonal of `(Kᴴ K)^{-1/2}` is positive for an invertible `K`:
`S_ii = ∑_k |U_ik|² / σ_k`. -/
lemma gramInvSqrt_apply_self_pos {K : Matrix n n ℂ} (hK : K.det ≠ 0) (i : n) :
    ∃ s : ℝ, 0 < s ∧ gramInvSqrt K i i = (s : ℂ) := by
  set hG := isHermitian_conjTranspose_mul_self K
  set U : Matrix n n ℂ := (hG.eigenvectorUnitary : Matrix n n ℂ)
  refine ⟨∑ k, ‖U i k‖ ^ 2 * (singularValues K k)⁻¹, ?_, ?_⟩
  · have h1 := congrFun (congrFun (eigenvectorUnitary_mul_conjTranspose hG) i) i
    rw [Matrix.mul_apply, one_apply_eq] at h1
    obtain ⟨k, -, hk⟩ : ∃ k ∈ univ, U i k * Uᴴ k i ≠ 0 := by
      by_contra h
      push Not at h
      rw [sum_eq_zero h] at h1
      exact zero_ne_one h1
    have hUk : U i k ≠ 0 := left_ne_zero_of_mul hk
    exact sum_pos'
      (fun l _ => mul_nonneg (sq_nonneg _) (inv_nonneg.mpr (singularValues_pos hK l).le))
      ⟨k, mem_univ k, mul_pos (by positivity) (inv_pos.mpr (singularValues_pos hK k))⟩
  · rw [gramInvSqrt_eq, Matrix.mul_apply]
    push_cast
    refine sum_congr rfl fun k _ => ?_
    rw [mul_diagonal, conjTranspose_apply, Complex.star_def, mul_right_comm, Complex.mul_conj']

example := gramInvSqrt_apply_self_pos (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) 0

/-- `K + Kᴴ = 2c` with `c > 0` makes `K` invertible: `Re ⟨v, K v⟩ = c ‖v‖²`. -/
lemma det_ne_zero_of_add_conjTranspose {K : Matrix n n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K + Kᴴ = ((2 * c : ℝ) : ℂ) • 1) : K.det ≠ 0 := by
  intro h0
  obtain ⟨v, hv, hKv⟩ := exists_mulVec_eq_zero_iff.mpr h0
  have h1 : star v ⬝ᵥ ((K + Kᴴ) *ᵥ v) = 0 := by
    rw [add_mulVec, dotProduct_add, hKv, dotProduct_zero, zero_add, dotProduct_mulVec,
      ← star_mulVec, hKv, star_zero, zero_dotProduct]
  rw [hK, smul_mulVec, one_mulVec, dotProduct_smul, star_dotProduct_self_eq_sum_norm_sq,
    smul_eq_mul, ← Complex.ofReal_mul, Complex.ofReal_eq_zero] at h1
  have h2 : ∑ j, ‖v j‖ ^ 2 = 0 := by
    rcases mul_eq_zero.mp h1 with h | h
    · linarith
    · exact h
  refine hv (funext fun j => ?_)
  have h3 := (sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg ‖v j‖).mp h2 j (mem_univ j)
  simpa using h3

example := det_ne_zero_of_add_conjTranspose (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1)
  one_pos (by simp [two_smul])

/-- `W + Wᴴ = 2c (Kᴴ K)^{-1/2}` for the polar factor `W` of a `K` with `K + Kᴴ = 2c`. -/
theorem polarFactor_add_conjTranspose {K : Matrix n n ℂ} {c : ℝ}
    (hK : K + Kᴴ = ((2 * c : ℝ) : ℂ) • 1) :
    polarFactor K + (polarFactor K)ᴴ = ((2 * c : ℝ) : ℂ) • gramInvSqrt K := by
  have hKh : Kᴴ = ((2 * c : ℝ) : ℂ) • 1 - K := eq_sub_of_add_eq' hK
  have hcomm : Commute (Kᴴ * K) K := by
    change Kᴴ * K * K = K * (Kᴴ * K)
    rw [hKh]
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul,
      Matrix.mul_assoc]
  have hSK : gramInvSqrt K * K = K * gramInvSqrt K := (hcomm.cfc_real fun x : ℝ => (√x)⁻¹).eq
  rw [polarFactor_eq_mul_gramInvSqrt, conjTranspose_mul, conjTranspose_gramInvSqrt, hKh,
    Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_one, hSK]
  abel

example := polarFactor_add_conjTranspose (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1)
  (by simp [two_smul])

/-- **The polar factor has a positive diagonal**, `review/flat_spectrum.md`, item CW2: for
`K + Kᴴ = 2c`, `c > 0`, `Re W_ii = c S_ii` with `S = (Kᴴ K)^{-1/2}`, and `S_ii > 0`. -/
theorem re_polarFactor_apply_self {K : Matrix n n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K + Kᴴ = ((2 * c : ℝ) : ℂ) • 1) (i : n) :
    ∃ s : ℝ, 0 < s ∧ gramInvSqrt K i i = (s : ℂ) ∧ (polarFactor K i i).re = c * s := by
  obtain ⟨s, hs, hsi⟩ := gramInvSqrt_apply_self_pos (det_ne_zero_of_add_conjTranspose hc hK) i
  refine ⟨s, hs, hsi, ?_⟩
  have h := congrArg Complex.re (congrFun (congrFun (polarFactor_add_conjTranspose hK) i) i)
  rw [Matrix.add_apply, conjTranspose_apply, Matrix.smul_apply, hsi, smul_eq_mul, Complex.add_re,
    Complex.star_def, Complex.conj_re, ← Complex.ofReal_mul, Complex.ofReal_re] at h
  linarith

example (i : Fin 2) := re_polarFactor_apply_self (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1)
  one_pos (by simp [two_smul]) i

/-- **A missing diagonal cell is not recovered**, `review/flat_spectrum.md`, item CW2: a `K`
with `K + Kᴴ = 2c`, `c > 0`, is not a minimum of the nuclear norm among the matrices equal to it
on `Ω` when `Ω` misses a diagonal cell. -/
theorem not_isMinOn_of_add_conjTranspose {K : Matrix n n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K + Kᴴ = ((2 * c : ℝ) : ℂ) • 1) {Ω : Set (n × n)} {i : n} (hi : (i, i) ∉ Ω) :
    ¬IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  intro hmin
  have h0 := (isMinOn_nuclearNorm_iff (det_ne_zero_of_add_conjTranspose hc hK) Ω).mp hmin i i hi
  obtain ⟨s, hs, -, h⟩ := re_polarFactor_apply_self hc hK i
  rw [h0, Complex.zero_re] at h
  nlinarith

/-- The identity loses every diagonal cell it does not see. -/
example : ¬IsMinOn nuclearNorm
    {X | ∀ ij ∈ ({(0, 0)}ᶜ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 =
      (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_add_conjTranspose (c := 1) (i := 0) one_pos (by simp [two_smul]) (by simp)

end Transformer.Grokking.FlatSpectrum
