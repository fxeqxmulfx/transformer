module

public import Transformer.Grokking.FlatSpectrum.Factorization

/-!
# Theorem A: when the target is its own minimum-norm interpolant

Source: `review/flat_spectrum.md`, Theorem A. The polar factor `W = K (K* K)^{-1/2}` is a
contraction with `tr (W* K) = ‖K‖_S1`. If `W` vanishes off `Ω`, then `tr (W* X) = tr (W* K)`
for every `X` agreeing with `K` on `Ω`, so `‖K‖_S1 ≤ ‖X‖_S1`. Conversely, for invertible `K`
the balanced factorization `K = U V*` gives `K + t E = (U + t M) V*` and so
`‖K + t E‖_S1 ≤ ‖K‖_S1 + t Re tr (W* E) + C t²`; along `E = -W_ij e_ij` off `Ω` the norm
drops unless `W_ij = 0`. The source argues by differentiability of the nuclear norm at `K`;
this one-sided bound is all the converse needs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

lemma nuclearNorm_nonneg (X : Matrix m n ℂ) : 0 ≤ nuclearNorm X :=
  Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _

/-- The polar factor in the eigenbasis `U` of `Kᴴ K`: `W = K U diag (σ⁻¹) Uᴴ`. -/
lemma polarFactor_eq (K : Matrix m n ℂ) :
    polarFactor K =
      K * ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) *
        diagonal (fun i => (((singularValues K i)⁻¹ : ℝ) : ℂ)) *
        ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ)ᴴ := by
  rw [polarFactor, (isHermitian_conjTranspose_mul_self K).cfc_eq, IsHermitian.cfc,
    Unitary.conjStarAlgAut_apply]
  simp only [Matrix.mul_assoc, Matrix.star_eq_conjTranspose]
  rfl

/-- The polar factor is a contraction attaining the nuclear norm, `tr (W* K) = ‖K‖_S1`,
`review/flat_spectrum.md`, Theorem A. -/
theorem polarFactor_isContraction_trace (K : Matrix m n ℂ) :
    IsContraction (polarFactor K) ∧ ((polarFactor K)ᴴ * K).trace = nuclearNorm K := by
  rw [polarFactor_eq]
  obtain ⟨h1, h2⟩ := polar_of_diagonalization (d := singularValues K)
    (eigenvectorUnitary_conjTranspose_mul _) (gram_diagonalization K)
  exact ⟨h1, by rw [h2, nuclearNorm]; push_cast; rfl⟩

omit [DecidableEq n] in
/-- `‖A + t B‖_S2² = ‖A‖_S2² + 2 t Re tr (A* B) + t² ‖B‖_S2²`. -/
lemma frobeniusSq_add_smul (A B : Matrix m n ℂ) (t : ℝ) :
    frobeniusSq (A + (t : ℂ) • B) =
      frobeniusSq A + 2 * t * (Aᴴ * B).trace.re + t ^ 2 * frobeniusSq B := by
  have hBA : (Bᴴ * A).trace.re = (Aᴴ * B).trace.re := by
    rw [show Bᴴ * A = (Aᴴ * B)ᴴ by rw [conjTranspose_mul, conjTranspose_conjTranspose],
      trace_conjTranspose, Complex.star_def, Complex.conj_re]
  simp only [frobeniusSq_eq_re_trace, conjTranspose_add, conjTranspose_smul, Complex.star_def,
    Complex.conj_ofReal, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
    trace_add, trace_smul, smul_eq_mul, Complex.add_re, Complex.re_ofReal_mul, hBA]
  ring

/-- `tr (Aᴴ e_ij c) = conj (A_ij) c`. -/
lemma trace_conjTranspose_mul_single [DecidableEq m] (A : Matrix m n ℂ) (i : m) (j : n)
    (c : ℂ) :
    (Aᴴ * Matrix.single i j c).trace = star (A i j) * c := by
  rw [trace_conjTranspose_mul]
  simp [Matrix.single_apply, ite_and]

/-- For `K` with positive singular values,
`‖K + t E‖_S1 ≤ ‖K‖_S1 + t Re tr (W* E) + C t²`: write `K = U V*` balanced and
`K + t E = (U + t M) V*`. -/
theorem nuclearNorm_add_smul_le {K : Matrix m n ℂ} (hK : ∀ i, 0 < singularValues K i)
    (E : Matrix m n ℂ) : ∃ C : ℝ, ∀ t : ℝ, nuclearNorm (K + (t : ℂ) • E) ≤
      nuclearNorm K + t * ((polarFactor K)ᴴ * E).trace.re + t ^ 2 * C := by
  set hG := isHermitian_conjTranspose_mul_self K
  set W : Matrix n n ℂ := (hG.eigenvectorUnitary : Matrix n n ℂ)
  have hW : Wᴴ * W = 1 := eigenvectorUnitary_conjTranspose_mul hG
  have hW' : W * Wᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hG
  obtain ⟨hfac, hU, hV⟩ := factor_of_diagonalization (d := singularValues K)
    (fun i => (hK i).le) hW (gram_diagonalization K)
  set Dm : Matrix n n ℂ := diagonal fun i => (((√(singularValues K i))⁻¹ : ℝ) : ℂ)
  set Dp : Matrix n n ℂ := diagonal fun i => ((√(singularValues K i) : ℝ) : ℂ)
  set M := E * W * Dm
  refine ⟨frobeniusSq M / 2, fun t => ?_⟩
  have hDD : Dm * Dp = 1 := by
    rw [diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext i
    rw [← Complex.ofReal_mul, inv_mul_cancel₀ (Real.sqrt_pos.mpr (hK i)).ne', Complex.ofReal_one]
  have hMV : M * (W * Dp)ᴴ = E := by
    rw [conjTranspose_mul, conjTranspose_diagonal_ofReal,
      show M * (Dp * Wᴴ) = E * W * (Dm * Dp) * Wᴴ by simp only [M, Matrix.mul_assoc], hDD,
      Matrix.mul_one, Matrix.mul_assoc, hW', Matrix.mul_one]
  have hsum : K + (t : ℂ) • E = (K * W * Dm + (t : ℂ) • M) * (W * Dp)ᴴ := by
    rw [Matrix.add_mul, Matrix.smul_mul, hMV, ← hfac]
  have hDm2 : Dm * Dm = diagonal fun i => (((singularValues K i)⁻¹ : ℝ) : ℂ) := by
    rw [diagonal_mul_diagonal]
    congr 1
    funext i
    rw [← Complex.ofReal_mul, ← mul_inv, Real.mul_self_sqrt (hK i).le]
  have htr : ((K * W * Dm)ᴴ * M).trace = ((polarFactor K)ᴴ * E).trace := by
    have hDmh : Dmᴴ = Dm := conjTranspose_diagonal_ofReal _
    rw [polarFactor_eq]
    simp only [conjTranspose_mul, conjTranspose_diagonal_ofReal, conjTranspose_conjTranspose,
      hDmh]
    calc (Dm * (Wᴴ * Kᴴ) * M).trace = ((Dm * (Wᴴ * Kᴴ * E)) * (W * Dm)).trace := by
          simp only [M, Matrix.mul_assoc]
      _ = ((W * Dm) * (Dm * (Wᴴ * Kᴴ * E))).trace := Matrix.trace_mul_comm _ _
      _ = _ := by
          rw [show W * Dm * (Dm * (Wᴴ * Kᴴ * E)) = W * (Dm * Dm) * (Wᴴ * Kᴴ) * E by
            simp only [Matrix.mul_assoc], hDm2]
          simp only [Matrix.mul_assoc]
          rfl
  have hle := nuclearNorm_le_of_eq_mul _ _ hsum
  rw [frobeniusSq_add_smul, hU, hV, htr] at hle
  have hN : nuclearNorm K = ∑ i, singularValues K i := rfl
  rw [hN]
  linarith

/-- An invertible matrix has positive singular values. -/
lemma singularValues_pos {K : Matrix n n ℂ} (hK : K.det ≠ 0) (i : n) :
    0 < singularValues K i := by
  have h := (isHermitian_conjTranspose_mul_self K).det_eq_prod_eigenvalues
  rw [det_mul, det_conjTranspose] at h
  have hne : star K.det * K.det ≠ 0 := mul_ne_zero (star_ne_zero.mpr hK) hK
  rw [h] at hne
  have hi : (isHermitian_conjTranspose_mul_self K).eigenvalues i ≠ 0 := fun h0 =>
    hne (Finset.prod_eq_zero (Finset.mem_univ i) (by rw [h0]; simp))
  exact Real.sqrt_pos.mpr (lt_of_le_of_ne (eigenvalues_gram_nonneg K i) (Ne.symm hi))

example (i : Fin 2) := singularValues_pos (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) i

example (E : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_add_smul_le (singularValues_pos (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp)) E

/-- **Theorem A**, `review/flat_spectrum.md`: an invertible `K` minimizes `‖X‖_S1` over
`{X = K on Ω}` iff its polar factor `W = K (K* K)^{-1/2}` vanishes on every unseen entry. -/
theorem isMinOn_nuclearNorm_iff {K : Matrix n n ℂ} (hK : K.det ≠ 0) (Ω : Set (n × n)) :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∀ i j, (i, j) ∉ Ω → polarFactor K i j = 0 := by
  obtain ⟨hW, htrK⟩ := polarFactor_isContraction_trace K
  rw [isMinOn_iff]
  constructor
  · intro hmin i j hij
    set w := polarFactor K i j
    obtain ⟨C, hC⟩ := nuclearNorm_add_smul_le (singularValues_pos hK) (Matrix.single i j (-w))
    have htr : ((polarFactor K)ᴴ * Matrix.single i j (-w)).trace.re = -‖w‖ ^ 2 := by
      rw [trace_conjTranspose_mul_single, mul_neg, Complex.neg_re, Complex.star_def,
        Complex.conj_mul']
      norm_cast
    have hmem (t : ℝ) : K + (t : ℂ) • Matrix.single i j (-w) ∈
        {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} := by
      intro ij hij'
      have : ¬(i = ij.1 ∧ j = ij.2) := fun ⟨h1, h2⟩ => hij (by rw [h1, h2]; exact hij')
      simp [this]
    have key (t : ℝ) (ht : 0 < t) : ‖w‖ ^ 2 ≤ t * C := by
      have h1 := hmin _ (hmem t)
      have h2 := hC t
      rw [htr] at h2
      have : t * ‖w‖ ^ 2 ≤ t * (t * C) := by nlinarith
      exact le_of_mul_le_mul_left this ht
    by_contra hw
    have hpos : 0 < ‖w‖ ^ 2 := by positivity
    have hC1 : 0 < 2 * (|C| + 1) := by positivity
    have := key (‖w‖ ^ 2 / (2 * (|C| + 1))) (div_pos hpos hC1)
    have h3 : ‖w‖ ^ 2 / (2 * (|C| + 1)) * C ≤ ‖w‖ ^ 2 / (2 * (|C| + 1)) * |C| :=
      mul_le_mul_of_nonneg_left (le_abs_self C) (div_pos hpos hC1).le
    have h4 : ‖w‖ ^ 2 / (2 * (|C| + 1)) * |C| < ‖w‖ ^ 2 := by
      rw [div_mul_eq_mul_div, div_lt_iff₀ hC1]
      nlinarith [abs_nonneg C]
    linarith
  · intro hzero X hX
    have h : ((polarFactor K)ᴴ * X).trace = ((polarFactor K)ᴴ * K).trace := by
      rw [trace_conjTranspose_mul, trace_conjTranspose_mul]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      by_cases hij : (i, j) ∈ Ω
      · rw [hX (i, j) hij]
      · rw [hzero i j hij, star_zero, zero_mul, zero_mul]
    have hle := norm_trace_le_nuclearNorm hW X
    rwa [h, htrK, Complex.norm_real, Real.norm_of_nonneg (nuclearNorm_nonneg K)] at hle

end Transformer.Grokking.FlatSpectrum
