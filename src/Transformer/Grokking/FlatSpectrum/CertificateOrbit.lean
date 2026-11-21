module

public import Transformer.Grokking.FlatSpectrum.DualCertificate
public import Transformer.Grokking.FlatSpectrum.Interpolation
public import Transformer.Grokking.FlatSpectrum.OnesCertificate

/-!
# All certificates of `K` agree on its row space

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A certificate of `K`
is a contraction `Y` with `tr(Yᴴ K) = ‖K‖_S1`. In an eigenbasis `v_k` of `Kᴴ K` this trace is a sum
of `⟨Y v_k, K v_k⟩`, each at most `s_k`, so each equals `s_k` and `s_k Y v_k = K v_k`
(`mul_eigenbasis_of_certificate`). Hence `Y Kᴴ = Z Kᴴ` for any two certificates
(`mul_conjTranspose_eq_of_certificates`), and `c Y x = K x` for every eigenvector `x` of `Kᴴ K` with
eigenvalue `c²` (`smul_mulVec_of_certificate`).

Criteria for "M does not return `K`" follow in `CertificateTests` and `CertificateLines`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- `‖s a - b‖² = s² ‖a‖² - 2 s Re (conj a * b) + ‖b‖²`. -/
lemma norm_sq_ofReal_mul_sub (s : ℝ) (a b : ℂ) :
    ‖(s : ℂ) * a - b‖ ^ 2 = s ^ 2 * ‖a‖ ^ 2 - 2 * s * (star a * b).re + ‖b‖ ^ 2 := by
  simp only [Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im, Complex.mul_re,
    Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, Complex.star_def, Complex.conj_re,
    Complex.conj_im]
  ring

omit [DecidableEq m] in
/-- **A certificate is fixed on the eigenbasis of `Kᴴ K`**: `s_k Y v_k = K v_k`. -/
theorem mul_eigenbasis_of_certificate {K Y : Matrix m n ℂ} (hY : IsContraction Y)
    (htr : (Yᴴ * K).trace = nuclearNorm K) :
    Y * ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) *
        diagonal (fun k => (singularValues K k : ℂ)) =
      K * ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) := by
  set hG := isHermitian_conjTranspose_mul_self K
  set W : Matrix n n ℂ := (hG.eigenvectorUnitary : Matrix n n ℂ) with hWdef
  have hW : Wᴴ * W = 1 := eigenvectorUnitary_conjTranspose_mul hG
  have hW' : W * Wᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hG
  have hs (k : n) : 0 ≤ singularValues K k := Real.sqrt_nonneg _
  have hA (k : n) : ∑ i, ‖(Y * W) i k‖ ^ 2 ≤ 1 := hY.sum_norm_sq_le hW k
  have hB (k : n) : ∑ i, ‖(K * W) i k‖ ^ 2 = singularValues K k ^ 2 := by
    have h := congrFun (congrFun (gram_diagonalization K) k) k
    rw [diagonal_apply_eq, show Wᴴ * (Kᴴ * K) * W = (K * W)ᴴ * (K * W) by
      simp only [conjTranspose_mul, Matrix.mul_assoc], conjTranspose_mul_self_apply_self] at h
    exact_mod_cast h
  set c : n → ℂ := fun k => ∑ i, star ((Y * W) i k) * (K * W) i k with hc
  have hsum : ∑ k, c k = ((∑ k, singularValues K k : ℝ) : ℂ) := by
    have h : ((Y * W)ᴴ * (K * W)).trace = (Yᴴ * K).trace := by
      rw [conjTranspose_mul, Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc,
        Matrix.mul_assoc, hW', Matrix.mul_one]
    rw [htr, trace_conjTranspose_mul, Finset.sum_comm] at h
    exact h
  have hle (k : n) : (c k).re ≤ singularValues K k := by
    calc (c k).re ≤ ‖c k‖ := Complex.re_le_norm _
      _ ≤ √(∑ i, ‖(Y * W) i k‖ ^ 2) * √(∑ i, ‖(K * W) i k‖ ^ 2) := norm_sum_star_mul_le _ _ _
      _ ≤ 1 * singularValues K k := by
        rw [hB k, Real.sqrt_sq (hs k)]
        exact mul_le_mul_of_nonneg_right (Real.sqrt_le_one.2 (hA k)) (hs k)
      _ = singularValues K k := one_mul _
  have hre : ∑ k, (c k).re = ∑ k, singularValues K k := by
    have h := congrArg Complex.re hsum
    rwa [Complex.re_sum, Complex.ofReal_re] at h
  have heq (k : n) : (c k).re = singularValues K k :=
    (Finset.sum_eq_sum_iff_of_le fun k _ => hle k).1 hre k (mem_univ k)
  ext i k
  rw [Matrix.mul_diagonal, mul_comm]
  have hzero : ∑ i', ‖(singularValues K k : ℂ) * (Y * W) i' k - (K * W) i' k‖ ^ 2 ≤ 0 := by
    simp_rw [norm_sq_ofReal_mul_sub]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      ← Complex.re_sum, hB k]
    have h1 : (∑ i', star ((Y * W) i' k) * (K * W) i' k).re = singularValues K k := heq k
    rw [h1]
    nlinarith [mul_le_mul_of_nonneg_left (hA k) (sq_nonneg (singularValues K k))]
  have hterm := (Finset.sum_eq_zero_iff_of_nonneg fun i' _ => sq_nonneg _).1
    (le_antisymm hzero (Finset.sum_nonneg fun i' _ => sq_nonneg _)) i (mem_univ i)
  exact sub_eq_zero.1 (norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 hterm))

example (K : Matrix (Fin 2) (Fin 2) ℂ) :=
  mul_eigenbasis_of_certificate (polarFactor_isContraction_trace K).1
    (polarFactor_isContraction_trace K).2

omit [DecidableEq m] in
/-- **Any two certificates act alike on the row space of `K`**: `Y Kᴴ = Z Kᴴ`. -/
theorem mul_conjTranspose_eq_of_certificates {K Y Z : Matrix m n ℂ} (hY : IsContraction Y)
    (htrY : (Yᴴ * K).trace = nuclearNorm K) (hZ : IsContraction Z)
    (htrZ : (Zᴴ * K).trace = nuclearNorm K) : Y * Kᴴ = Z * Kᴴ := by
  set W : Matrix n n ℂ :=
    ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ)
  have hW' : W * Wᴴ = 1 :=
    eigenvectorUnitary_mul_conjTranspose (isHermitian_conjTranspose_mul_self K)
  set D : Matrix n n ℂ := diagonal fun k => (singularValues K k : ℂ)
  have hDh : Dᴴ = D := conjTranspose_diagonal_ofReal _
  have key {A B : Matrix m n ℂ} (hA : A * W * D = K * W) (hB : B * W * D = K * W) :
      A * Kᴴ = K * Bᴴ := by
    have hK : K = B * W * D * Wᴴ := by rw [hB, Matrix.mul_assoc, hW', Matrix.mul_one]
    have hKh : Kᴴ = W * D * Wᴴ * Bᴴ := by
      conv_lhs => rw [hK]
      simp only [conjTranspose_mul, conjTranspose_conjTranspose, hDh, Matrix.mul_assoc]
    calc A * Kᴴ = (A * W * D) * Wᴴ * Bᴴ := by rw [hKh]; simp only [Matrix.mul_assoc]
      _ = K * Bᴴ := by rw [hA, Matrix.mul_assoc K, hW', Matrix.mul_one]
  have hYc : Y * W * D = K * W := mul_eigenbasis_of_certificate hY htrY
  have hZc : Z * W * D = K * W := mul_eigenbasis_of_certificate hZ htrZ
  rw [key hYc hYc, key hZc hYc]

example (K : Matrix (Fin 2) (Fin 2) ℂ) :=
  mul_conjTranspose_eq_of_certificates (polarFactor_isContraction_trace K).1
    (polarFactor_isContraction_trace K).2 (polarFactor_isContraction_trace K).1
    (polarFactor_isContraction_trace K).2

omit [DecidableEq m] in
/-- **On an eigenvector of `Kᴴ K` with eigenvalue `c²` a certificate is `K / c`**. -/
theorem smul_mulVec_of_certificate {K Y : Matrix m n ℂ} (hY : IsContraction Y)
    (htr : (Yᴴ * K).trace = nuclearNorm K) {x : n → ℂ} {c : ℝ} (hc : 0 ≤ c)
    (hx : (Kᴴ * K) *ᵥ x = ((c ^ 2 : ℝ) : ℂ) • x) : (c : ℂ) • (Y *ᵥ x) = K *ᵥ x := by
  set hG := isHermitian_conjTranspose_mul_self K
  set W : Matrix n n ℂ := (hG.eigenvectorUnitary : Matrix n n ℂ) with hWdef
  have hW' : W * Wᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hG
  have hcert : Y * W * diagonal (fun k => (singularValues K k : ℂ)) = K * W :=
    mul_eigenbasis_of_certificate hY htr
  set ξ : n → ℂ := Wᴴ *ᵥ x with hξ
  have hxW : W *ᵥ ξ = x := by rw [hξ, mulVec_mulVec, hW', one_mulVec]
  have hgram : Wᴴ * (Kᴴ * K) = diagonal (fun k => ((singularValues K k ^ 2 : ℝ) : ℂ)) * Wᴴ := by
    rw [← gram_diagonalization K]
    simp only [Matrix.mul_assoc]
    rw [← hWdef, hW', Matrix.mul_one]
  have hk (k : n) : (singularValues K k : ℂ) * ξ k = c * ξ k := by
    have h1 : ((singularValues K k ^ 2 : ℝ) : ℂ) * ξ k = ((c ^ 2 : ℝ) : ℂ) * ξ k := by
      have h : (Wᴴ *ᵥ ((Kᴴ * K) *ᵥ x)) k = (Wᴴ *ᵥ (((c ^ 2 : ℝ) : ℂ) • x)) k := by rw [hx]
      rw [mulVec_mulVec, hgram, ← mulVec_mulVec, mulVec_diagonal, mulVec_smul, Pi.smul_apply,
        smul_eq_mul] at h
      exact h
    by_cases h0 : ξ k = 0
    · simp [h0]
    · have h2 : singularValues K k ^ 2 = c ^ 2 := by exact_mod_cast mul_right_cancel₀ h0 h1
      have h3 : singularValues K k = c :=
        (pow_left_inj₀ (Real.sqrt_nonneg _) hc two_ne_zero).1 h2
      rw [h3]
  have hD : diagonal (fun k => (singularValues K k : ℂ)) *ᵥ ξ = (c : ℂ) • ξ := by
    funext k
    rw [mulVec_diagonal, Pi.smul_apply, smul_eq_mul, hk]
  calc (c : ℂ) • (Y *ᵥ x) = (Y * W) *ᵥ ((c : ℂ) • ξ) := by
        rw [mulVec_smul, ← mulVec_mulVec, hxW]
    _ = (Y * W * diagonal fun k => (singularValues K k : ℂ)) *ᵥ ξ := by
        rw [← hD, mulVec_mulVec]
    _ = K *ᵥ x := by rw [hcert, ← mulVec_mulVec, hxW]

end Transformer.Grokking.FlatSpectrum
