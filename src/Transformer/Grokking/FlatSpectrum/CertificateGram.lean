module

public import Transformer.Grokking.FlatSpectrum.CertificateOrbit
public import Transformer.Grokking.FlatSpectrum.ContractionDual
public import Transformer.Grokking.FlatSpectrum.OneCell

/-!
# A certificate of `K` is a polar factor of `K`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Every certificate `Y`
of `K` has `Yᴴ K = (Kᴴ K)^{1/2}` and `Y (Kᴴ K)^{1/2} = K`. So `Yᴴ K` is Hermitian, and a zero row of
`Y` is a zero row of `K`. For an invertible `K` the certificate is unique. If `K T = c 1`, then
`Y T` is Hermitian up to a phase, and when `T` is sparse this relation ties a few entries of `Y`
together, so a single live entry in it has to vanish (`entry_eq_zero_of_peel`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The square root of `Kᴴ K`, from its eigenbasis. -/
noncomputable def gramSqrt (K : Matrix m n ℂ) : Matrix n n ℂ :=
  ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) *
    diagonal (fun k => (singularValues K k : ℂ)) *
    ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ)ᴴ

omit [DecidableEq m] in
lemma gramSqrt_conjTranspose (K : Matrix m n ℂ) : (gramSqrt K)ᴴ = gramSqrt K := by
  simp only [gramSqrt, conjTranspose_mul, conjTranspose_conjTranspose,
    conjTranspose_diagonal_ofReal, Matrix.mul_assoc]

omit [DecidableEq m] in
/-- The polar factor gives the square root of the Gram matrix. -/
lemma conjTranspose_polarFactor_mul (K : Matrix m n ℂ) : (polarFactor K)ᴴ * K = gramSqrt K := by
  set hG := isHermitian_conjTranspose_mul_self K
  set W : Matrix n n ℂ := (hG.eigenvectorUnitary : Matrix n n ℂ) with hWdef
  have hW' : W * Wᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hG
  have hgram : Wᴴ * (Kᴴ * K) = diagonal (fun k => ((singularValues K k ^ 2 : ℝ) : ℂ)) * Wᴴ := by
    rw [← gram_diagonalization K]
    simp only [Matrix.mul_assoc]
    rw [← hWdef, hW', Matrix.mul_one]
  have hd : diagonal (fun i => (((singularValues K i)⁻¹ : ℝ) : ℂ)) *
      diagonal (fun k => ((singularValues K k ^ 2 : ℝ) : ℂ)) =
      diagonal fun k => (singularValues K k : ℂ) := by
    rw [diagonal_mul_diagonal]
    congr 1
    funext k
    by_cases h : singularValues K k = 0
    · simp [h]
    · rw [← Complex.ofReal_mul]
      congr 1
      field_simp
  set D' : Matrix n n ℂ := diagonal fun i => (((singularValues K i)⁻¹ : ℝ) : ℂ) with hD'
  have hstep : (K * W * D' * Wᴴ)ᴴ * K = W * D' * (Wᴴ * (Kᴴ * K)) := by
    simp only [conjTranspose_mul, conjTranspose_conjTranspose, hD',
      conjTranspose_diagonal_ofReal, Matrix.mul_assoc]
  rw [polarFactor_eq, gramSqrt]
  change (K * W * D' * Wᴴ)ᴴ * K = W * (diagonal fun k => (singularValues K k : ℂ)) * Wᴴ
  rw [hstep, hgram]
  calc W * D' * (diagonal (fun k => ((singularValues K k ^ 2 : ℝ) : ℂ)) * Wᴴ)
      = W * (D' * diagonal fun k => ((singularValues K k ^ 2 : ℝ) : ℂ)) * Wᴴ := by
        simp only [Matrix.mul_assoc]
    _ = W * (diagonal fun k => (singularValues K k : ℂ)) * Wᴴ := by rw [hd]

/-- A certificate of `K` gives a certificate of `Kᴴ`. -/
lemma trace_conjTranspose_certificate {K Y : Matrix m n ℂ}
    (htr : (Yᴴ * K).trace = nuclearNorm K) : ((Yᴴ)ᴴ * Kᴴ).trace = nuclearNorm Kᴴ := by
  rw [conjTranspose_conjTranspose, Matrix.trace_mul_comm,
    show Kᴴ * Y = (Yᴴ * K)ᴴ by rw [conjTranspose_mul, conjTranspose_conjTranspose],
    trace_conjTranspose, htr, nuclearNorm_conjTranspose]
  simp

example (K : Matrix (Fin 2) (Fin 2) ℂ) :=
  trace_conjTranspose_certificate (polarFactor_isContraction_trace K).2

/-- **For any certificate, `Yᴴ K = (Kᴴ K)^{1/2}`.** -/
theorem conjTranspose_mul_of_certificate {K Y : Matrix m n ℂ} (hY : IsContraction Y)
    (htr : (Yᴴ * K).trace = nuclearNorm K) : Yᴴ * K = gramSqrt K := by
  have hZ := polarFactor_isContraction_trace K
  have key := mul_conjTranspose_eq_of_certificates (K := Kᴴ) (Y := Yᴴ) (Z := (polarFactor K)ᴴ)
    hY.conjTranspose (trace_conjTranspose_certificate htr) hZ.1.conjTranspose
    (trace_conjTranspose_certificate hZ.2)
  rw [conjTranspose_conjTranspose] at key
  rw [key, conjTranspose_polarFactor_mul]

example (K : Matrix (Fin 2) (Fin 2) ℂ) :=
  conjTranspose_mul_of_certificate (polarFactor_isContraction_trace K).1
    (polarFactor_isContraction_trace K).2

omit [DecidableEq m] in
/-- **For any certificate, `Y (Kᴴ K)^{1/2} = K`.** -/
theorem mul_gramSqrt_of_certificate {K Y : Matrix m n ℂ} (hY : IsContraction Y)
    (htr : (Yᴴ * K).trace = nuclearNorm K) : Y * gramSqrt K = K := by
  have h := mul_eigenbasis_of_certificate hY htr
  have hW' := eigenvectorUnitary_mul_conjTranspose (isHermitian_conjTranspose_mul_self K)
  rw [gramSqrt, ← Matrix.mul_assoc, ← Matrix.mul_assoc, h, Matrix.mul_assoc, hW', Matrix.mul_one]

example (K : Matrix (Fin 2) (Fin 2) ℂ) :=
  mul_gramSqrt_of_certificate (polarFactor_isContraction_trace K).1
    (polarFactor_isContraction_trace K).2

/-- The transpose of a contraction is a contraction. -/
theorem IsContraction.transpose {Y : Matrix m n ℂ} (hY : IsContraction Y) : IsContraction Yᵀ := by
  have hd := hY.conjTranspose
  rw [isContraction_iff_sum_norm_sq_le] at hd ⊢
  intro v
  have hmul (j : n) : (Yᵀ *ᵥ v) j = star ((Yᴴ *ᵥ fun i => star (v i)) j) := by
    simp [mulVec, dotProduct, conjTranspose_apply, transpose_apply]
  calc ∑ j, ‖(Yᵀ *ᵥ v) j‖ ^ 2 = ∑ j, ‖(Yᴴ *ᵥ fun i => star (v i)) j‖ ^ 2 := by
        simp only [hmul, norm_star]
    _ ≤ ∑ i, ‖star (v i)‖ ^ 2 := hd _
    _ = ∑ i, ‖v i‖ ^ 2 := by simp only [norm_star]

example := (isContraction_one (n := Fin 2)).transpose

omit [DecidableEq m] [DecidableEq n] in
/-- Transposition keeps the pairing of a certificate with `K`. -/
lemma trace_transpose_certificate (K Y : Matrix m n ℂ) :
    ((Yᵀ)ᴴ * Kᵀ).trace = (Yᴴ * K).trace := by
  rw [trace_conjTranspose_mul, trace_conjTranspose_mul, Finset.sum_comm]
  rfl

/-- **An invertible `K` has one certificate.** -/
theorem eq_of_certificates {K Y Z : Matrix n n ℂ} (hK : K.det ≠ 0) (hY : IsContraction Y)
    (htrY : (Yᴴ * K).trace = nuclearNorm K) (hZ : IsContraction Z)
    (htrZ : (Zᴴ * K).trace = nuclearNorm K) : Y = Z := by
  have h := mul_conjTranspose_eq_of_certificates hY htrY hZ htrZ
  have hunit : IsUnit (Kᴴ).det := by
    rw [det_conjTranspose]
    exact (star_ne_zero.2 hK).isUnit
  calc Y = Y * Kᴴ * (Kᴴ)⁻¹ := by rw [Matrix.mul_assoc, mul_nonsing_inv _ hunit, Matrix.mul_one]
    _ = Z * Kᴴ * (Kᴴ)⁻¹ := by rw [h]
    _ = Z := by rw [Matrix.mul_assoc, mul_nonsing_inv _ hunit, Matrix.mul_one]

example : polarFactor (1 : Matrix (Fin 2) (Fin 2) ℂ) = polarFactor 1 :=
  eq_of_certificates (by simp) (polarFactor_isContraction_trace 1).1
    (polarFactor_isContraction_trace 1).2 (polarFactor_isContraction_trace 1).1
    (polarFactor_isContraction_trace 1).2

/-- **If `K T = c 1`, then `Y T` is Hermitian up to a phase.** -/
theorem smul_mul_eq_of_certificate {K Y T : Matrix n n ℂ} {c : ℂ} (hY : IsContraction Y)
    (htr : (Yᴴ * K).trace = nuclearNorm K) (hKT : K * T = c • 1) :
    star c • (Y * T) = c • (Y * T)ᴴ := by
  have hG := conjTranspose_mul_of_certificate hY htr
  have hH : Kᴴ * Y = Yᴴ * K := by
    rw [show Kᴴ * Y = (Yᴴ * K)ᴴ by rw [conjTranspose_mul, conjTranspose_conjTranspose], hG,
      gramSqrt_conjTranspose]
  have hKTh : Tᴴ * Kᴴ = star c • 1 := by
    rw [← conjTranspose_mul, hKT, conjTranspose_smul, conjTranspose_one]
  calc star c • (Y * T) = (Tᴴ * Kᴴ) * (Y * T) := by rw [hKTh, Matrix.smul_mul, Matrix.one_mul]
    _ = Tᴴ * (Kᴴ * Y) * T := by simp only [Matrix.mul_assoc]
    _ = Tᴴ * Yᴴ * (K * T) := by rw [hH]; simp only [Matrix.mul_assoc]
    _ = c • (Y * T)ᴴ := by
        rw [hKT, Matrix.mul_smul, Matrix.mul_one, conjTranspose_mul]

example : star (1 : ℂ) • (polarFactor (1 : Matrix (Fin 2) (Fin 2) ℂ) * 1) =
    (1 : ℂ) • (polarFactor (1 : Matrix (Fin 2) (Fin 2) ℂ) * 1)ᴴ :=
  smul_mul_eq_of_certificate (K := 1) (polarFactor_isContraction_trace 1).1
    (polarFactor_isContraction_trace 1).2 (by simp)

omit [DecidableEq n] in
/-- **One live entry in a sparse relation has to vanish.** If `Y T` is Hermitian up to a phase, the
row `j` meets the support of the column `i` of `T` only in dead cells, and the row `i` meets the
support of the column `j` only in the cell `(i, l)`, then `Y i l = 0`. -/
theorem entry_eq_zero_of_peel {Y T : Matrix n n ℂ} {c : ℂ} (hc : c ≠ 0)
    (hrel : star c • (Y * T) = c • (Y * T)ᴴ) {S : Set (n × n)}
    (hS : ∀ i j, (i, j) ∉ S → Y i j = 0) {i l j : n} (hT : T l j ≠ 0)
    (h1 : ∀ l', l' ≠ l → T l' j ≠ 0 → (i, l') ∉ S)
    (h2 : ∀ l', T l' i ≠ 0 → (j, l') ∉ S) : Y i l = 0 := by
  have hji : (Y * T) j i = 0 := by
    rw [Matrix.mul_apply]
    refine Finset.sum_eq_zero fun l' _ => ?_
    by_cases h : T l' i = 0
    · rw [h, mul_zero]
    · rw [hS _ _ (h2 l' h), zero_mul]
  have hij : (Y * T) i j = 0 := by
    have h : (star c • (Y * T)) i j = (c • (Y * T)ᴴ) i j := by rw [hrel]
    rw [Matrix.smul_apply, Matrix.smul_apply, conjTranspose_apply, hji, star_zero, smul_zero,
      smul_eq_mul] at h
    exact (mul_eq_zero.1 h).resolve_left (star_ne_zero.2 hc)
  rw [Matrix.mul_apply, Finset.sum_eq_single l] at hij
  · exact (mul_eq_zero.1 hij).resolve_right hT
  · intro l' _ hne
    by_cases h : T l' j = 0
    · rw [h, mul_zero]
    · rw [hS _ _ (h1 l' hne h), zero_mul]
  · exact fun h => (h (mem_univ l)).elim

example : (0 : Matrix (Fin 2) (Fin 2) ℂ) 0 1 = 0 :=
  entry_eq_zero_of_peel (T := 1) (c := 1) (S := ∅) one_ne_zero (by simp) (by simp)
    (j := 1) (by simp) (fun _ _ _ => Set.notMem_empty _) (fun _ _ => Set.notMem_empty _)

end Transformer.Grokking.FlatSpectrum
