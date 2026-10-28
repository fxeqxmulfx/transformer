module

public import Transformer.Grokking.FlatSpectrum.WeightedRecovery

/-!
# A transport plan inside a rigid pattern pins `a bᵀ`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 2:
Theorem F, item 2, for any `a`, `b`. Let `Ω ⊆ m × n` carry a plan `T` from `|a_i|²` to
`|b_j|²`, and let `X` equal `a bᵀ` on `Ω` with `‖X‖_S1 ≤ ∑ |a_i|²`. The three sums of
`sum_sum_mul_norm_sq_sub_of_transport` then vanish. So `u_i = 0` where `a_i = 0` and `v_j = 0`
where `b_j = 0`: `X` vanishes on these rows and columns, as `a bᵀ` does. On the supports,
`x_i = y_j` wherever `T_ij > 0`. Row `i` of the support carries the mass `|a_i|² > 0`, so
`‖x_i‖² = ⟨x_i, y_j⟩ = 1` for some `j`; likewise `‖y_j‖ = 1`. On every cell of `Ω` inside the
supports, `‖x_i − y_j‖² = 1 + 1 − 2 = 0`. Rigidity on the supports, applied to each coordinate,
makes all these rows equal, so `X_ij = a_i ‖x_i‖² b_j = a_i b_j` (`eq_vecMulVec_of_transport`).
Without zero entries, rigidity on the supports is rigidity (`rigid_support_of_ne_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- **A plan inside a rigid pattern pins `a bᵀ`**, `review/flat_spectrum.md`, Theorem G, item 2:
if `Ω` carries a plan from `|a_i|²` to `|b_j|²` and is rigid on the supports of `a` and `b`,
every `X` equal to `a bᵀ` on `Ω` with `‖X‖_S1 ≤ ∑ |a_i|²` is `a bᵀ`. -/
theorem eq_vecMulVec_of_transport {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)} {T : m → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T)
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ),
      (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
        ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0)
    {X : Matrix m n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2)
    (hle : nuclearNorm X ≤ ∑ i, ‖a i‖ ^ 2) : X = vecMulVec a b := by
  obtain ⟨r, U, V, hUV, hc⟩ := (nuclearNorm_isLeast X).1
  have hid := sum_sum_mul_norm_sq_sub_of_transport hT (U := U) (V := V) fun ij hij => by
    rw [← hUV]
    exact hX ij hij
  have hTa (i : m) (hi : a i = 0) (j : n) : T i j = 0 := hT.eq_zero_of_row (by simp [hi]) j
  have hTb (j : n) (hj : b j = 0) (i : m) : T i j = 0 := hT.eq_zero_of_col (by simp [hj]) i
  obtain ⟨hT0, hTΩ, hrow, hcol⟩ := hT
  set x : Matrix m (Fin r) ℂ := of fun i k => (a i)⁻¹ * U i k with hx
  set y : Matrix n (Fin r) ℂ := of fun j k => (star (b j))⁻¹ * V j k with hy
  have hxe (i : m) (k : Fin r) : (a i)⁻¹ * U i k = x i k := rfl
  have hye (j : n) (k : Fin r) : (star (b j))⁻¹ * V j k = y j k := rfl
  simp only [hxe, hye] at hid
  -- The three sums are nonnegative and add up to `2 (‖X‖_S1 − ∑ |a_i|²) ≤ 0`.
  have hnn (i : m) (j : n) : ∀ k ∈ univ, 0 ≤ ‖x i k - y j k‖ ^ 2 := fun _ _ => sq_nonneg _
  have hTnn (i : m) : ∀ j ∈ univ, 0 ≤ T i j * ∑ k, ‖x i k - y j k‖ ^ 2 :=
    fun j _ => mul_nonneg (hT0 i j) (sum_nonneg (hnn i j))
  have hDU : ∀ i ∈ univ, 0 ≤ if a i = 0 then ∑ k, ‖U i k‖ ^ 2 else 0 := fun i _ => by
    split_ifs
    exacts [sum_nonneg fun _ _ => sq_nonneg _, le_rfl]
  have hDV : ∀ j ∈ univ, 0 ≤ if b j = 0 then ∑ k, ‖V j k‖ ^ 2 else 0 := fun j _ => by
    split_ifs
    exacts [sum_nonneg fun _ _ => sq_nonneg _, le_rfl]
  have hW0 := sum_nonneg fun i (_ : i ∈ univ) => sum_nonneg (hTnn i)
  have hU0 := sum_nonneg hDU
  have hV0 := sum_nonneg hDV
  have h0 : ∑ i, ∑ j, T i j * ∑ k, ‖x i k - y j k‖ ^ 2 = 0 := by linarith
  have hU1 : ∑ i, (if a i = 0 then ∑ k, ‖U i k‖ ^ 2 else 0) = 0 := by linarith
  have hV1 : ∑ j, (if b j = 0 then ∑ k, ‖V j k‖ ^ 2 else 0) = 0 := by linarith
  have hsq (z w : ℂ) (hzw : ‖z - w‖ ^ 2 = 0) : z = w := by
    rwa [sq_eq_zero_iff, norm_eq_zero, sub_eq_zero] at hzw
  -- `X` vanishes on the rows and columns of mass `0`.
  have hUz (i : m) (hi : a i = 0) (k : Fin r) : U i k = 0 := by
    have h := (sum_eq_zero_iff_of_nonneg hDU).mp hU1 i (mem_univ i)
    rw [ite_eq_left hi] at h
    simpa using (sum_eq_zero_iff_of_nonneg fun k _ => sq_nonneg ‖U i k‖).mp h k (mem_univ k)
  have hVz (j : n) (hj : b j = 0) (k : Fin r) : V j k = 0 := by
    have h := (sum_eq_zero_iff_of_nonneg hDV).mp hV1 j (mem_univ j)
    rw [ite_eq_left hj] at h
    simpa using (sum_eq_zero_iff_of_nonneg fun k _ => sq_nonneg ‖V j k‖).mp h k (mem_univ k)
  -- On the supports, the argument of Theorem F.
  have hxy (i : m) (j : n) : (x * yᴴ) i j = (a i)⁻¹ * X i j * (b j)⁻¹ := by
    rw [hx, hy, rowDiv_mul_apply, ← hUV]
  have hΩ1 (ij : m × n) (hij : ij ∈ Ω) (hi : a ij.1 ≠ 0) (hj : b ij.2 ≠ 0) :
      (x * yᴴ) ij.1 ij.2 = 1 := by
    rw [hxy, hX ij hij, vecMulVec_apply]
    field_simp
  have hTD (i : m) (j : n) : T i j * ∑ k, ‖x i k - y j k‖ ^ 2 = 0 :=
    (sum_eq_zero_iff_of_nonneg (hTnn i)).mp ((sum_eq_zero_iff_of_nonneg fun i _ =>
      sum_nonneg (hTnn i)).mp h0 i (mem_univ i)) j (mem_univ j)
  have hpos (i : m) (j : n) (h : 0 < T i j) : (i, j) ∈ Ω ∧ a i ≠ 0 ∧ b j ≠ 0 :=
    ⟨by_contra fun hn => h.ne' (hTΩ i j hn), fun hi => h.ne' (hTa i hi j),
      fun hj => h.ne' (hTb j hj i)⟩
  have hxy0 (i : m) (j : n) (h : 0 < T i j) (k : Fin r) : x i k = y j k :=
    hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn i j)).mp
      ((mul_eq_zero.mp (hTD i j)).resolve_left h.ne') k (mem_univ k))
  have hexr (i : m) (hi : a i ≠ 0) : ∃ j, 0 < T i j := by
    by_contra h
    have h' : ∑ j, T i j = 0 :=
      sum_eq_zero fun j _ => le_antisymm (not_lt.mp fun hj => h ⟨j, hj⟩) (hT0 i j)
    exact hi (by simpa [hrow i] using h')
  have hexc (j : n) (hj : b j ≠ 0) : ∃ i, 0 < T i j := by
    by_contra h
    have h' : ∑ i, T i j = 0 :=
      sum_eq_zero fun i _ => le_antisymm (not_lt.mp fun hi => h ⟨i, hi⟩) (hT0 i j)
    exact hj (by simpa [hcol j] using h')
  have hu (i : m) (hi : a i ≠ 0) : ∑ k, ‖x i k‖ ^ 2 = 1 := by
    obtain ⟨j, hj⟩ := hexr i hi
    obtain ⟨hijΩ, -, hbj⟩ := hpos i j hj
    have h := sum_norm_sq_sub_eq x y i j
    rw [(sum_eq_zero_iff_of_nonneg (hnn i j)).mpr fun k _ => by rw [hxy0 i j hj k, sub_self,
      norm_zero, zero_pow two_ne_zero], hΩ1 (i, j) hijΩ hi hbj, Complex.one_re] at h
    have hb' : ∑ k, ‖y j k‖ ^ 2 = ∑ k, ‖x i k‖ ^ 2 :=
      sum_congr rfl fun k _ => by rw [hxy0 i j hj k]
    linarith
  have hv (j : n) (hj : b j ≠ 0) : ∑ k, ‖y j k‖ ^ 2 = 1 := by
    obtain ⟨i, hi⟩ := hexc j hj
    rw [← hu i (hpos i j hi).2.1]
    exact sum_congr rfl fun k _ => by rw [hxy0 i j hi k]
  have hcell (ij : m × n) (hij : ij ∈ Ω) (hi : a ij.1 ≠ 0) (hj : b ij.2 ≠ 0) (k : Fin r) :
      x ij.1 k + -y ij.2 k = 0 := by
    have h := sum_norm_sq_sub_eq x y ij.1 ij.2
    rw [hu _ hi, hv _ hj, hΩ1 ij hij hi hj, Complex.one_re] at h
    rw [hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn ij.1 ij.2)).mp (by rw [h]; norm_num) k
      (mem_univ k)), add_neg_cancel]
  ext i j
  by_cases hi : a i = 0
  · rw [vecMulVec_apply, hi, zero_mul, hUV, mul_apply]
    exact sum_eq_zero fun k _ => by rw [hUz i hi k, zero_mul]
  by_cases hj : b j = 0
  · rw [vecMulVec_apply, hj, mul_zero, hUV, mul_apply]
    exact sum_eq_zero fun k _ => by rw [conjTranspose_apply, hVz j hj k, star_zero, mul_zero]
  have hall (k : Fin r) : x i k = y j k := by
    have h := hΩ (fun i => x i k) (fun j => -y j k) (fun ij hij hi hj => hcell ij hij hi hj k)
      i j hi hj
    rwa [← sub_eq_add_neg, sub_eq_zero] at h
  have h1 : (x * yᴴ) i j = 1 := by
    rw [mul_apply, ← Complex.ofReal_one, ← hu i hi, Complex.ofReal_sum]
    refine sum_congr rfl fun k _ => ?_
    rw [conjTranspose_apply, ← hall k, Complex.star_def, Complex.mul_conj', Complex.ofReal_pow]
  rw [hxy] at h1
  rw [vecMulVec_apply]
  have hai : a i ≠ 0 := hi
  have hbj : b j ≠ 0 := hj
  field_simp at h1
  linear_combination h1

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Without zero entries, rigidity on the supports is rigidity. -/
lemma rigid_support_of_ne_zero {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, a i ≠ 0) (hb : ∀ j, b j ≠ 0)
    {Ω : Set (m × n)}
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) :
    ∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
      ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0 :=
  fun r s h i j _ _ => hΩ r s (fun ij hij => h ij hij (ha _) (hb _)) i j

example := rigid_support_of_ne_zero (a := ![2, 1]) (b := ![2, 1]) (by simp [Fin.forall_fin_two])
  (by simp [Fin.forall_fin_two]) rigid_ne_one_zero

example := eq_vecMulVec_of_transport (a := ![2, 1]) (b := ![2, 1])
  (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _)
  (rigid_support_of_ne_zero (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two])
    rigid_ne_one_zero) (X := vecMulVec ![2, 1] ![2, 1]) (fun _ _ => rfl)
  (nuclearNorm_vecMulVec_le _ _ |>.trans_eq (by simp [Fin.sum_univ_two]))

end Transformer.Grokking.FlatSpectrum
