module

public import Transformer.Grokking.FlatSpectrum.Blocks

/-!
# One missing entry of a flat kernel: the exact nuclear norm

Source: `review/flat_spectrum.md`, Corollary, proof sketch. Let `K* K = N I` and `|K_ij| = 1`.
Changing the entry `(i, j)` by `t` gives `X = K (1 + (t/N) u e_jᵀ)` with `u = K* e_i`, since
`K u = N e_i`. In the orthonormal frame `[e_j, w]`, `w = (u − u_j e_j)/√(N − 1)`, the vector `u`
reads `(u_j, √(N − 1))`, so the factor is the identity off the frame and the block
`B = [[1 + t u_j/N, 0], [t √(N − 1)/N, 1]]` on it. Hence
`‖X‖_S1 = √N [(N − 2) + √(‖B‖_S2² + 2 |det B|)]`, `det B = 1 + t u_j/N`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The frame `[e_j, (v − v_j e_j) / √(N − 1)]`, `N = card n`, `review/flat_spectrum.md`,
Corollary. -/
noncomputable def frame (v : n → ℂ) (j : n) : Matrix n (Fin 2) ℂ :=
  of fun k c => if c = 0 then (if k = j then 1 else 0) else
    (((√((Fintype.card n : ℝ) - 1))⁻¹ : ℝ) : ℂ) * (v k - if k = j then v j else 0)

lemma frame_zero (v : n → ℂ) (j k : n) : frame v j k 0 = if k = j then 1 else 0 := rfl

lemma frame_one (v : n → ℂ) (j k : n) : frame v j k 1 =
    (((√((Fintype.card n : ℝ) - 1))⁻¹ : ℝ) : ℂ) * (v k - if k = j then v j else 0) := rfl

/-- The frame coordinates of `v` are `(v_j, √(N − 1))`. -/
lemma frame_reconstruct {v : n → ℂ} {j : n} (hN : 2 ≤ Fintype.card n) (k : n) :
    frame v j k 0 * v j + frame v j k 1 * ((√((Fintype.card n : ℝ) - 1) : ℝ) : ℂ) = v k := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have hr : ((√((Fintype.card n : ℝ) - 1) : ℝ) : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.mpr (by linarith)).ne'
  rw [frame_zero, frame_one]
  split_ifs with h
  · subst h
    simp
  · push_cast
    field_simp
    ring

example := frame_reconstruct (v := fun _ : Fin 2 => 1) (j := 0) le_rfl 1

/-- The frame is orthonormal when `|v_j| = 1` and `∑ |v_k|² = N ≥ 2`. -/
theorem conjTranspose_mul_frame {v : n → ℂ} {j : n} (hj : ‖v j‖ = 1)
    (hv : ∑ k, ‖v k‖ ^ 2 = Fintype.card n) (hN : 2 ≤ Fintype.card n) :
    (frame v j)ᴴ * frame v j = 1 := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have hr0 : 0 < √((Fintype.card n : ℝ) - 1) := Real.sqrt_pos.mpr (by linarith)
  have hr2 : √((Fintype.card n : ℝ) - 1) ^ 2 = Fintype.card n - 1 := Real.sq_sqrt (by linarith)
  have hw : ∑ k, ‖v k - (if k = j then v j else 0)‖ ^ 2 = Fintype.card n - 1 := by
    have h (k : n) :
        ‖v k - (if k = j then v j else 0)‖ ^ 2 = ‖v k‖ ^ 2 - if k = j then 1 else 0 := by
      split_ifs with hk
      · subst hk
        simp [hj]
      · simp
    rw [Finset.sum_congr rfl fun k _ => h k, Finset.sum_sub_distrib, hv]
    simp
  have h00 : ((frame v j)ᴴ * frame v j) 0 0 = 1 := by
    simp [mul_apply, frame_zero]
  have h01 : ((frame v j)ᴴ * frame v j) 0 1 = 0 := by
    simp [mul_apply, frame_zero, frame_one]
  have h10 : ((frame v j)ᴴ * frame v j) 1 0 = 0 := by
    simp [mul_apply, frame_zero, frame_one]
  have h11 : ((frame v j)ᴴ * frame v j) 1 1 = 1 := by
    rw [conjTranspose_mul_self_apply_self]
    have h (k : n) : ‖frame v j k 1‖ ^ 2 =
        (√((Fintype.card n : ℝ) - 1))⁻¹ ^ 2 * ‖v k - (if k = j then v j else 0)‖ ^ 2 := by
      rw [frame_one, norm_mul, Complex.norm_real, Real.norm_of_nonneg (inv_nonneg.mpr hr0.le),
        mul_pow]
    rw [Finset.sum_congr rfl fun k _ => h k, ← Finset.mul_sum, hw, inv_pow, hr2,
      inv_mul_cancel₀ (by linarith : (0 : ℝ) < Fintype.card n - 1).ne', Complex.ofReal_one]
  rw [eta_fin_two ((frame v j)ᴴ * frame v j), h00, h01, h10, h11, one_fin_two]

example := conjTranspose_mul_frame (v := fun _ : Fin 2 => 1) (j := 0) (by simp) (by simp) le_rfl

/-- A flat square `K` (`K* K = N I`) also has `K K* = N I`. -/
lemma mul_conjTranspose_of_flat {K : Matrix n n ℂ} (hN : Fintype.card n ≠ 0)
    (hK : Kᴴ * K = (Fintype.card n : ℂ) • 1) : K * Kᴴ = (Fintype.card n : ℂ) • 1 := by
  have hN0 : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hN
  have h1 : ((Fintype.card n : ℂ)⁻¹ • Kᴴ) * K = 1 := by
    rw [Matrix.smul_mul, hK, smul_smul, inv_mul_cancel₀ hN0, one_smul]
  have h2 := mul_eq_one_comm.mp h1
  rw [Matrix.mul_smul] at h2
  rw [← h2, smul_smul, mul_inv_cancel₀ hN0, one_smul]

example := mul_conjTranspose_of_flat (K := !![(1 : ℂ), 1; 1, -1]) (by simp)
  (by ext a b; fin_cases a <;> fin_cases b <;> simp [mul_apply, Fin.sum_univ_two] <;> norm_num)

/-- **Corollary**, exact form, `review/flat_spectrum.md`: for flat `K` (`K* K = N I`) with
`|K_ij| = 1` and `N ≥ 2`, changing the entry `(i, j)` by `t` gives
`‖K + t e_ij‖_S1 = √N [(N − 2) + √(|d|² + |t|² (N − 1)/N² + 1 + 2 |d|)]`, where
`d = 1 + t conj(K_ij)/N` is the determinant of the changed `2 × 2` block. -/
theorem nuclearNorm_flat_add_single {K : Matrix n n ℂ}
    (hK : Kᴴ * K = (Fintype.card n : ℂ) • 1) {i j : n} (hij : ‖K i j‖ = 1)
    (hN : 2 ≤ Fintype.card n) (t : ℂ) :
    nuclearNorm (K + t • single i j 1) = √(Fintype.card n : ℝ) *
      ((Fintype.card n - 2) + √(‖1 + t * star (K i j) / Fintype.card n‖ ^ 2 +
        ‖t‖ ^ 2 * (Fintype.card n - 1) / Fintype.card n ^ 2 + 1 +
        2 * ‖1 + t * star (K i j) / Fintype.card n‖)) := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have hN0 : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hrow (k : n) : ∑ m, K k m * star (K i m) = if k = i then (Fintype.card n : ℂ) else 0 := by
    have h := congrFun (congrFun (mul_conjTranspose_of_flat (by omega) hK) k) i
    simpa [mul_apply, one_apply] using h
  set u : n → ℂ := fun k => star (K i k) with hudef
  have hu : ∑ k, ‖u k‖ ^ 2 = Fintype.card n := by
    have h := hrow i
    simp only [ite_true, Complex.mul_conj', ← Complex.ofReal_pow, ← Complex.ofReal_sum,
      Complex.star_def] at h
    simpa [u] using (by exact_mod_cast h : ∑ m, ‖K i m‖ ^ 2 = (Fintype.card n : ℝ))
  have hJ := conjTranspose_mul_frame (v := u) (j := j) (by simp [u, hij]) hu hN
  set J := frame u j
  set r : ℝ := √((Fintype.card n : ℝ) - 1) with hrdef
  set B : Matrix (Fin 2) (Fin 2) ℂ :=
    !![1 + t * u j / Fintype.card n, 0; t * r / Fintype.card n, 1] with hBdef
  have hJB : J * B * Jᴴ = J * Jᴴ + (t / Fintype.card n) •
      vecMulVec u (fun l => if l = j then 1 else 0) := by
    ext k l
    have hk := frame_reconstruct (v := u) (j := j) hN k
    have hl : star (J l 0) = if l = j then 1 else 0 := by
      simp only [J, frame_zero]
      split_ifs <;> simp
    simp only [mul_apply, Fin.sum_univ_two, conjTranspose_apply, Matrix.add_apply,
      Matrix.smul_apply, vecMulVec_apply, smul_eq_mul, hBdef, of_apply, cons_val', cons_val_zero,
      cons_val_one, empty_val', cons_val_fin_one, hl]
    linear_combination (t / Fintype.card n * (if l = j then 1 else 0)) * hk
  set M : Matrix n n ℂ := ((1 : ℝ) : ℂ) • (1 - J * Jᴴ) + J * B * Jᴴ with hMdef
  have hM : M = 1 + (t / Fintype.card n) • vecMulVec u (fun l => if l = j then 1 else 0) := by
    rw [hMdef, hJB, Complex.ofReal_one, one_smul]
    abel
  have hX : K + t • single i j 1 = K * M := by
    rw [hM, Matrix.mul_add, Matrix.mul_one, Matrix.mul_smul]
    congr 1
    ext k l
    simp only [Matrix.smul_apply, single_apply, smul_eq_mul, mul_apply, vecMulVec_apply]
    rw [show ∑ m, K k m * (u m * (if l = j then 1 else 0)) =
        (∑ m, K k m * star (K i m)) * (if l = j then 1 else 0) by
      rw [Finset.sum_mul]
      exact Finset.sum_congr rfl fun m _ => by rw [hudef]; ring, hrow k]
    by_cases hki : k = i
    · subst hki
      by_cases hlj : l = j
      · subst hlj
        simp [hN0]
      · simp [hlj, Ne.symm hlj]
    · simp [hki, Ne.symm hki]
  have hgram : (K * M)ᴴ * (K * M) =
      (((√(Fintype.card n : ℝ) : ℝ) : ℂ) • M)ᴴ * (((√(Fintype.card n : ℝ) : ℝ) : ℂ) • M) := by
    have h1 : (K * M)ᴴ * (K * M) = (Fintype.card n : ℂ) • (Mᴴ * M) := by
      rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Kᴴ, hK, Matrix.smul_mul,
        Matrix.one_mul, Matrix.mul_smul]
    have h2 : (((√(Fintype.card n : ℝ) : ℝ) : ℂ) • M)ᴴ * (((√(Fintype.card n : ℝ) : ℝ) : ℂ) • M) =
        (Fintype.card n : ℂ) • (Mᴴ * M) := by
      rw [conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul, Complex.star_def,
        Complex.conj_ofReal, ← Complex.ofReal_mul, Real.mul_self_sqrt (Nat.cast_nonneg _),
        Complex.ofReal_natCast]
    rw [h1, h2]
  have hdet : B.det = 1 + t * star (K i j) / Fintype.card n := by
    simp [hBdef, u]
  have hfro : frobeniusSq B = ‖1 + t * star (K i j) / Fintype.card n‖ ^ 2 +
      ‖t‖ ^ 2 * (Fintype.card n - 1) / Fintype.card n ^ 2 + 1 := by
    simp only [frobeniusSq, Fin.sum_univ_two, hBdef, of_apply, cons_val', cons_val_zero,
      cons_val_one, empty_val', cons_val_fin_one, norm_zero, norm_one, norm_div,
      norm_mul, Complex.norm_real, Complex.norm_natCast]
    rw [show u j = star (K i j) from rfl, hrdef, Real.norm_of_nonneg (Real.sqrt_nonneg _)]
    linear_combination (‖t‖ ^ 2 / (Fintype.card n : ℝ) ^ 2) *
      Real.sq_sqrt (by linarith : (0 : ℝ) ≤ Fintype.card n - 1)
  rw [hX, nuclearNorm_eq_of_gram_eq hgram, nuclearNorm_smul, hMdef,
    nuclearNorm_block hJ zero_le_one, nuclearNorm_fin_two, hdet, hfro, Complex.norm_real,
    Real.norm_of_nonneg (Real.sqrt_nonneg _), Fintype.card_fin]
  push_cast
  ring

example := nuclearNorm_flat_add_single (K := !![(1 : ℂ), 1; 1, -1]) (i := 0) (j := 0)
  (by ext a b; fin_cases a <;> fin_cases b <;> simp [mul_apply, Fin.sum_univ_two] <;> norm_num)
  (by simp) le_rfl 1

end Transformer.Grokking.FlatSpectrum
