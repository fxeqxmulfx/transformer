module

public import Transformer.Grokking.FlatSpectrum.OnesRecovery
public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# Exact recovery of a table of any rank

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 1: Theorem D's items 1 and 2
for a table `K` of any rank. Let `U`, `V` have orthonormal columns with `U Uᴴ K = K` and
`tr((U Vᴴ)ᴴ K) = ‖K‖_S1`; a compact singular value decomposition `K = U diag(s) Vᴴ`, `s ≥ 0`,
gives both (`svd_hypotheses`). Let `Y = U Vᴴ + W` vanish off `Ω`, with `Uᴴ W = 0`, `W V = 0`
and `c W` a contraction, `c > 0`. With `Π = 1 − U Uᴴ`, `Π' = 1 − V Vᴴ` and a contraction `Z₀`
attaining `‖Π H Π'‖_S1`, `U Vᴴ + Π Z₀ Π'` is a contraction
(`isContraction_mul_conjTranspose_add`); its pairing with `K + H` is
`‖K‖_S1 + tr((U Vᴴ)ᴴ H) + ‖Π H Π'‖_S1`, and for `H` vanishing on `Ω`, `tr(Yᴴ H) = 0` gives
`tr((U Vᴴ)ᴴ H) = −tr(Wᴴ Π H Π')`, of modulus at most `‖Π H Π'‖_S1 / c`. Hence
`‖K‖_S1 + (1 − 1/c) ‖Π H Π'‖_S1 ≤ ‖K + H‖_S1` (`le_nuclearNorm_add_of_certificate`). For `c > 1`
a minimum has `Π H Π' = 0`, so `H = U Aᴴ + B Vᴴ`; if no nonzero matrix of that form vanishes on
`Ω`, `K` is the unique minimum (`isMinOn_iff_of_certificate`). Theorem D's items 1 and 2 are the
case `K = J`, `U = V = 1/√N`, `W = Y − J/N`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

/-- For orthonormal columns `U`, `V` and a contraction `Z₀`,
`U Vᴴ + (1 − U Uᴴ) Z₀ (1 − V Vᴴ)` is a contraction. -/
lemma isContraction_mul_conjTranspose_add {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {Z₀ : Matrix m n ℂ} (hZ : IsContraction Z₀) :
    IsContraction (U * Vᴴ + (1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ)) := by
  have hU' (M : Matrix r n ℂ) : Uᴴ * (U * M) = M := by rw [← Matrix.mul_assoc, hU, Matrix.one_mul]
  have hV' (M : Matrix r n ℂ) : Vᴴ * (V * M) = M := by rw [← Matrix.mul_assoc, hV, Matrix.one_mul]
  have key : 1 - (U * Vᴴ + (1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ))ᴴ *
      (U * Vᴴ + (1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ)) =
      (1 - V * Vᴴ)ᴴ * ((1 - Z₀ᴴ * Z₀) + (Uᴴ * Z₀)ᴴ * (Uᴴ * Z₀)) * (1 - V * Vᴴ) := by
    simp only [conjTranspose_add, conjTranspose_mul, conjTranspose_sub, conjTranspose_one,
      conjTranspose_conjTranspose, Matrix.mul_add, Matrix.add_mul, Matrix.mul_sub,
      Matrix.sub_mul, Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc, hU', hV']
    abel
  rw [IsContraction, key]
  exact (hZ.add (posSemidef_conjTranspose_mul_self _)).conjTranspose_mul_mul_same _

example := isContraction_mul_conjTranspose_add (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) isContraction_zero

/-- `U Vᴴ` is a contraction for orthonormal columns `U`, `V`. -/
lemma isContraction_mul_conjTranspose {U : Matrix m r ℂ} {V : Matrix n r ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) : IsContraction (U * Vᴴ) := by
  simpa using isContraction_mul_conjTranspose_add hU hV (isContraction_zero (m := m) (n := n))

example := isContraction_mul_conjTranspose (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp)

omit [DecidableEq m] in
/-- **The compact singular value decomposition gives the hypotheses**: for orthonormal columns
`U`, `V` and `s ≥ 0`, `K = U diag(s) Vᴴ` has `U Uᴴ K = K`, `tr((U Vᴴ)ᴴ K) = ‖K‖_S1` and
`‖K‖_S1 = ∑ s`. -/
theorem svd_hypotheses {U : Matrix m r ℂ} {V : Matrix n r ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 ≤ s k) :
    U * Uᴴ * (U * diagonal (fun k => (s k : ℂ)) * Vᴴ) = U * diagonal (fun k => (s k : ℂ)) * Vᴴ ∧
      ((U * Vᴴ)ᴴ * (U * diagonal (fun k => (s k : ℂ)) * Vᴴ)).trace =
        nuclearNorm (U * diagonal (fun k => (s k : ℂ)) * Vᴴ) ∧
      nuclearNorm (U * diagonal (fun k => (s k : ℂ)) * Vᴴ) = ∑ k, s k := by
  have hnorm : nuclearNorm (U * diagonal (fun k => (s k : ℂ)) * Vᴴ) = ∑ k, s k := by
    rw [nuclearNorm_isometry hU hV]
    refine nuclearNorm_eq_of_diagonalization (W := 1) hs (by simp) ?_
    rw [conjTranspose_one, Matrix.one_mul, Matrix.mul_one, diagonal_conjTranspose,
      diagonal_mul_diagonal]
    congr 1
    funext k
    simp only [Pi.star_apply, Complex.star_def, Complex.conj_ofReal]
    push_cast
    ring
  refine ⟨?_, ?_, hnorm⟩
  · simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Uᴴ U, hU, Matrix.one_mul]
  · rw [hnorm, conjTranspose_mul, conjTranspose_conjTranspose,
      show V * Uᴴ * (U * diagonal (fun k => (s k : ℂ)) * Vᴴ) =
          V * (diagonal (fun k => (s k : ℂ)) * Vᴴ) by
        simp only [Matrix.mul_assoc]
        rw [← Matrix.mul_assoc Uᴴ U, hU, Matrix.one_mul],
      Matrix.trace_mul_comm, Matrix.mul_assoc, hV, Matrix.mul_one, trace_diagonal]
    push_cast
    rfl

example := svd_hypotheses (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp)
  (s := ![1, 2]) fun k => by fin_cases k <;> norm_num

/-- **The certificate bound at every rank**, the rank-`r` form of Theorem D, item 1: with `U`,
`V` as above and `Y = U Vᴴ + W` vanishing off `Ω`, `Uᴴ W = 0`, `W V = 0` and `c W` a
contraction, `c > 0`, every `H` vanishing on `Ω` has
`‖K‖_S1 + (1 − 1/c) ‖(1 − U Uᴴ) H (1 − V Vᴴ)‖_S1 ≤ ‖K + H‖_S1`. -/
theorem le_nuclearNorm_add_of_certificate {K : Matrix m n ℂ} {U : Matrix m r ℂ}
    {V : Matrix n r ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K) {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0) (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 0 < c) (hW : IsContraction ((c : ℂ) • W)) {H : Matrix m n ℂ}
    (hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0) :
    nuclearNorm K + (1 - c⁻¹) * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) ≤
      nuclearNorm (K + H) := by
  have hPh : (1 - U * Uᴴ)ᴴ = 1 - U * Uᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
  have hP'h : (1 - V * Vᴴ)ᴴ = 1 - V * Vᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
  set A := (1 - U * Uᴴ) * H * (1 - V * Vᴴ) with hAdef
  obtain ⟨Z₀, hZ₀, htr⟩ := exists_isContraction_trace_eq A
  have hPK : (1 - U * Uᴴ) * K = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hKU, sub_self]
  -- the pairing of `K + H` with the contraction `U Vᴴ + Π Z₀ Π'`
  have h1 : ((U * Vᴴ + (1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ))ᴴ * (K + H)).trace =
      nuclearNorm K + ((U * Vᴴ)ᴴ * H).trace + nuclearNorm A := by
    have h0 : (((1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ))ᴴ * K).trace = 0 := by
      rw [conjTranspose_mul, conjTranspose_mul, Matrix.mul_assoc, Matrix.mul_assoc, hPh, hPK,
        Matrix.mul_zero, Matrix.mul_zero, trace_zero]
    have hA : (((1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ))ᴴ * H).trace = (Z₀ᴴ * A).trace := by
      rw [conjTranspose_mul, conjTranspose_mul, hPh, hP'h, Matrix.mul_assoc,
        Matrix.trace_mul_comm, hAdef]
      simp only [Matrix.mul_assoc]
    rw [conjTranspose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, trace_add, trace_add,
      trace_add, hK, h0, hA, htr]
    ring
  -- `tr((U Vᴴ)ᴴ H) = −tr(Wᴴ A)`
  have hWP : (1 - U * Uᴴ) * W * (1 - V * Vᴴ) = W := by
    have e1 : U * Uᴴ * W = 0 := by rw [Matrix.mul_assoc, hWU, Matrix.mul_zero]
    have e2 : W * (V * Vᴴ) = 0 := by rw [← Matrix.mul_assoc, hWV, Matrix.zero_mul]
    rw [Matrix.sub_mul, Matrix.one_mul, e1, sub_zero, Matrix.mul_sub, Matrix.mul_one, e2,
      sub_zero]
  have hWA : (Wᴴ * A).trace = (Wᴴ * H).trace := by
    conv_rhs => rw [← hWP, conjTranspose_mul, conjTranspose_mul, hPh, hP'h, Matrix.mul_assoc,
      Matrix.trace_mul_comm]
    simp only [hAdef, Matrix.mul_assoc]
  have hYH : ((U * Vᴴ + W)ᴴ * H).trace = 0 := by
    rw [trace_conjTranspose_mul]
    refine sum_eq_zero fun i _ => sum_eq_zero fun j _ => ?_
    by_cases hij : (i, j) ∈ Ω
    · rw [hH (i, j) hij, mul_zero]
    · rw [hYΩ i j hij, star_zero, zero_mul]
  have h2 : ((U * Vᴴ)ᴴ * H).trace = -(Wᴴ * A).trace := by
    rw [conjTranspose_add, Matrix.add_mul, trace_add] at hYH
    rw [hWA]
    linear_combination hYH
  -- `|tr(Wᴴ A)| ≤ ‖A‖_S1 / c`
  have h3 : ‖(Wᴴ * A).trace‖ ≤ c⁻¹ * nuclearNorm A := by
    have h := norm_trace_le_nuclearNorm hW A
    rw [conjTranspose_smul, Complex.star_def, Complex.conj_ofReal, Matrix.smul_mul, trace_smul,
      smul_eq_mul, norm_mul, Complex.norm_real, Real.norm_of_nonneg hc.le] at h
    rwa [le_inv_mul_iff₀ hc]
  have h4 : ‖((U * Vᴴ + (1 - U * Uᴴ) * Z₀ * (1 - V * Vᴴ))ᴴ * (K + H)).trace‖ ≤
      nuclearNorm (K + H) :=
    norm_trace_le_nuclearNorm (isContraction_mul_conjTranspose_add hU hV hZ₀) (K + H)
  rw [h1, h2] at h4
  have h5 := Complex.re_le_norm
    ((nuclearNorm K : ℂ) + -(Wᴴ * A).trace + (nuclearNorm A : ℂ))
  have h6 := Complex.re_le_norm (Wᴴ * A).trace
  simp only [Complex.add_re, Complex.neg_re, Complex.ofReal_re] at h5
  nlinarith

example :=
  have h := svd_hypotheses (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp)
    (s := ![1, 2]) fun k => by fin_cases k <;> norm_num
  le_nuclearNorm_add_of_certificate (by simp) (by simp) h.1 h.2.1 (Ω := Set.univ) (W := 0)
    (fun _ _ hij => absurd (Set.mem_univ _) hij) (by simp) (by simp) two_pos
    (by rw [smul_zero]; exact isContraction_zero) (H := 0) fun _ _ => rfl

end Transformer.Grokking.FlatSpectrum
