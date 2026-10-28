module

public import Transformer.Grokking.FlatSpectrum.PermutationRecovery
public import Transformer.Grokking.FlatSpectrum.Transport

/-!
# A transport plan bounds the nuclear norm from below

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 1:
Theorem F, item 1, where the table `m × n` may be rectangular and `a`, `b` may have zero
entries. Let `T` be a transport plan on `Ω` from `|a_i|²` to `|b_j|²` (`IsTransportPlan`), so
that `∑ |a_i|² = ∑ |b_j|²`. Let `X` equal `a bᵀ` on `Ω`, and write `X = U Vᴴ` with
`½ (‖U‖² + ‖V‖²) = ‖X‖_S1` (Lemma 1). The rows `x_i = u_i / a_i` and `y_j = v_j / conj b_j`,
`0` where `a_i` or `b_j` vanishes, have `⟨x_i, y_j⟩ = X_ij / (a_i b_j)` (`rowDiv_mul_apply`).
That is `1` on every cell that carries mass, since a row or a column of mass `0` carries none
(`IsTransportPlan.eq_zero_of_row`). So
`∑_{ij} T_ij ‖x_i − y_j‖² + ∑_{a_i = 0} ‖u_i‖² + ∑_{b_j = 0} ‖v_j‖² = ‖U‖² + ‖V‖² − 2 ∑ |a_i|²`
(`sum_sum_mul_norm_sq_sub`, `sum_sum_mul_norm_sq_sub_of_transport`), and
`‖X‖_S1 ≥ ∑ |a_i|²` (`sum_norm_sq_le_nuclearNorm_of_transport`), which is at least
`‖a bᵀ‖_S1` (`nuclearNorm_vecMulVec_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n]

/-- `∑_{ij} T_ij ‖x_i − y_j‖² = ∑_i α_i ‖x_i‖² + ∑_j β_j ‖y_j‖² − 2 ∑_{ij} T_ij Re ⟨x_i, y_j⟩`,
with `α` and `β` the row and column sums of `T`. -/
lemma sum_sum_mul_norm_sq_sub {r : Type*} [Fintype r] (T : m → n → ℝ) (x : Matrix m r ℂ)
    (y : Matrix n r ℂ) :
    ∑ i, ∑ j, T i j * ∑ k, ‖x i k - y j k‖ ^ 2 =
      ∑ i, (∑ j, T i j) * ∑ k, ‖x i k‖ ^ 2 + ∑ j, (∑ i, T i j) * ∑ k, ‖y j k‖ ^ 2 -
        2 * ∑ i, ∑ j, T i j * ((x * yᴴ) i j).re := by
  have hA : ∑ i, (∑ j, T i j) * ∑ k, ‖x i k‖ ^ 2 = ∑ i, ∑ j, T i j * ∑ k, ‖x i k‖ ^ 2 :=
    sum_congr rfl fun i _ => sum_mul _ _ _
  have hB : ∑ j, (∑ i, T i j) * ∑ k, ‖y j k‖ ^ 2 = ∑ i, ∑ j, T i j * ∑ k, ‖y j k‖ ^ 2 :=
    (sum_congr rfl fun j _ => sum_mul _ _ _).trans sum_comm
  rw [hA, hB, mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  refine sum_congr rfl fun i _ => ?_
  rw [mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  refine sum_congr rfl fun j _ => ?_
  rw [sum_norm_sq_sub_eq]
  ring

omit [Fintype m] [Fintype n] in
/-- Dividing the rows of `U` by `a` and those of `V` by `conj b` divides `(U Vᴴ)_ij` by
`a_i b_j`. -/
lemma rowDiv_mul_apply {r : Type*} [Fintype r] (a : m → ℂ) (b : n → ℂ) (U : Matrix m r ℂ)
    (V : Matrix n r ℂ) (i : m) (j : n) :
    ((of fun i k => (a i)⁻¹ * U i k) * (of fun j k => (star (b j))⁻¹ * V j k)ᴴ) i j =
      (a i)⁻¹ * (U * Vᴴ) i j * (b j)⁻¹ := by
  simp only [mul_apply, conjTranspose_apply, of_apply, star_mul', star_inv₀, star_star, mul_sum,
    sum_mul]
  exact sum_congr rfl fun k _ => by ring

/-- Under a transport plan from `|a_i|²` to `|b_j|²`, with `U Vᴴ = a bᵀ` on `Ω`,
`∑_{ij} T_ij ‖u_i / a_i − v_j / conj b_j‖² + ∑_{a_i = 0} ‖u_i‖² + ∑_{b_j = 0} ‖v_j‖²` is
`‖U‖² + ‖V‖² − 2 ∑ |a_i|²`. -/
lemma sum_sum_mul_norm_sq_sub_of_transport {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {T : m → n → ℝ} (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T)
    {r : Type*} [Fintype r] {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hX : ∀ ij ∈ Ω, (U * Vᴴ) ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    ∑ i, ∑ j, T i j * ∑ k, ‖(a i)⁻¹ * U i k - (star (b j))⁻¹ * V j k‖ ^ 2 +
        ∑ i, (if a i = 0 then ∑ k, ‖U i k‖ ^ 2 else 0) +
        ∑ j, (if b j = 0 then ∑ k, ‖V j k‖ ^ 2 else 0) =
      frobeniusSq U + frobeniusSq V - 2 * ∑ i, ‖a i‖ ^ 2 := by
  have hTa (i : m) (hi : a i = 0) (j : n) : T i j = 0 := hT.eq_zero_of_row (by simp [hi]) j
  have hTb (j : n) (hj : b j = 0) (i : m) : T i j = 0 := hT.eq_zero_of_col (by simp [hj]) i
  obtain ⟨_, hTΩ, hrow, hcol⟩ := hT
  have hrow' : ∀ i, ∑ j, T i j = ‖a i‖ ^ 2 := hrow
  have hcol' : ∀ j, ∑ i, T i j = ‖b j‖ ^ 2 := hcol
  have h := sum_sum_mul_norm_sq_sub T (of fun i k => (a i)⁻¹ * U i k)
    (of fun j k => (star (b j))⁻¹ * V j k)
  simp only [of_apply, rowDiv_mul_apply] at h
  have hA (i : m) : (∑ j, T i j) * ∑ k, ‖(a i)⁻¹ * U i k‖ ^ 2 =
      ∑ k, ‖U i k‖ ^ 2 - if a i = 0 then ∑ k, ‖U i k‖ ^ 2 else 0 := by
    rw [hrow' i]
    by_cases hi : a i = 0
    · simp [hi]
    rw [ite_eq_right hi, sub_zero, mul_sum]
    refine sum_congr rfl fun k _ => ?_
    rw [norm_mul, norm_inv, mul_pow, inv_pow, ← mul_assoc,
      mul_inv_cancel₀ (pow_ne_zero 2 (norm_ne_zero_iff.mpr hi)), one_mul]
  have hB (j : n) : (∑ i, T i j) * ∑ k, ‖(star (b j))⁻¹ * V j k‖ ^ 2 =
      ∑ k, ‖V j k‖ ^ 2 - if b j = 0 then ∑ k, ‖V j k‖ ^ 2 else 0 := by
    rw [hcol' j]
    by_cases hj : b j = 0
    · simp [hj]
    rw [ite_eq_right hj, sub_zero, mul_sum]
    refine sum_congr rfl fun k _ => ?_
    rw [norm_mul, norm_inv, norm_star, mul_pow, inv_pow, ← mul_assoc,
      mul_inv_cancel₀ (pow_ne_zero 2 (norm_ne_zero_iff.mpr hj)), one_mul]
  have hC (i : m) (j : n) : T i j * ((a i)⁻¹ * (U * Vᴴ) i j * (b j)⁻¹).re = T i j := by
    by_cases hi : a i = 0
    · rw [hTa i hi j, zero_mul]
    by_cases hj : b j = 0
    · rw [hTb j hj i, zero_mul]
    by_cases hij : (i, j) ∈ Ω
    · have h1 : (a i)⁻¹ * (U * Vᴴ) i j * (b j)⁻¹ = 1 := by
        rw [hX _ hij, vecMulVec_apply]
        field_simp
      rw [h1, Complex.one_re, mul_one]
    · rw [hTΩ i j hij, zero_mul]
  simp only [hA, hB, hC, sum_sub_distrib] at h
  simp only [hrow'] at h
  rw [frobeniusSq, frobeniusSq]
  linarith

example := sum_sum_mul_norm_sq_sub_of_transport (a := ![2, 1]) (b := ![2, 1])
  (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _) (U := of fun i (_ : Fin 1) => ![2, 1] i)
  (V := of fun j (_ : Fin 1) => star (![2, 1] j)) fun _ _ => by
    simp [mul_apply, vecMulVec_apply]

/-- **A transport plan bounds the norm**, `review/flat_spectrum.md`, Theorem G, item 1: if `Ω`
carries a plan from `|a_i|²` to `|b_j|²`, every `X` equal to `a bᵀ` on `Ω` has
`∑ |a_i|² ≤ ‖X‖_S1`. -/
theorem sum_norm_sq_le_nuclearNorm_of_transport [DecidableEq n] {a : m → ℂ} {b : n → ℂ}
    {Ω : Set (m × n)} {T : m → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T) {X : Matrix m n ℂ}
    (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    ∑ i, ‖a i‖ ^ 2 ≤ nuclearNorm X := by
  obtain ⟨r, U, V, hUV, hc⟩ := (nuclearNorm_isLeast X).1
  have h := sum_sum_mul_norm_sq_sub_of_transport hT (U := U) (V := V) fun ij hij => by
    rw [← hUV]
    exact hX ij hij
  have h0 : 0 ≤ ∑ i, ∑ j, T i j * ∑ k, ‖(a i)⁻¹ * U i k - (star (b j))⁻¹ * V j k‖ ^ 2 :=
    sum_nonneg fun i _ => sum_nonneg fun j _ =>
      mul_nonneg (hT.1 i j) (sum_nonneg fun _ _ => sq_nonneg _)
  have h1 : 0 ≤ ∑ i, (if a i = 0 then ∑ k, ‖U i k‖ ^ 2 else 0) := sum_nonneg fun i _ => by
    split_ifs
    exacts [sum_nonneg fun _ _ => sq_nonneg _, le_rfl]
  have h2 : 0 ≤ ∑ j, (if b j = 0 then ∑ k, ‖V j k‖ ^ 2 else 0) := sum_nonneg fun j _ => by
    split_ifs
    exacts [sum_nonneg fun _ _ => sq_nonneg _, le_rfl]
  linarith

/-- From the entry `1` alone, every completion of the row `(1, x)` has `‖(1, x)‖_S1 ≥ 1`. -/
example := sum_norm_sq_le_nuclearNorm_of_transport (a := ![(1 : ℂ)]) (b := ![1, 0])
  (Ω := {((0 : Fin 1), (0 : Fin 2))}) (T := fun _ j => ![1, 0] j)
  ⟨by simp [Fin.forall_fin_two], by simp [Fin.forall_fin_two], by simp [Fin.sum_univ_two],
    by simp [Fin.forall_fin_two]⟩ (X := of ![![1, 5]]) fun ij hij => by
    rw [Set.mem_singleton_iff.mp hij]
    simp [vecMulVec_apply]

/-- `‖a bᵀ‖_S1 ≤ (∑ |a_i|² + ∑ |b_j|²) / 2`, from the factorization through one column. -/
theorem nuclearNorm_vecMulVec_le [DecidableEq n] (a : m → ℂ) (b : n → ℂ) :
    nuclearNorm (vecMulVec a b) ≤ (∑ i, ‖a i‖ ^ 2 + ∑ j, ‖b j‖ ^ 2) / 2 := by
  have h := nuclearNorm_le_of_eq_mul (of fun i (_ : Fin 1) => a i)
    (of fun j (_ : Fin 1) => star (b j)) (X := vecMulVec a b) (by
      ext i j
      simp [mul_apply, vecMulVec_apply])
  simpa [frobeniusSq] using h

end Transformer.Grokking.FlatSpectrum
