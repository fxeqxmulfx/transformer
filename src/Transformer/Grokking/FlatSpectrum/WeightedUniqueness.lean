module

public import Transformer.Grokking.FlatSpectrum.WeightedRecovery

/-!
# A transport plan inside a rigid pattern pins `a bᵀ`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F (added).
Let no entry of `a` or `b` vanish, let `Ω` carry a plan `T` from `|a_i|²` to `|b_j|²`, and let
`X` equal `a bᵀ` on `Ω` with `‖X‖_S1 ≤ ∑ |a_i|²`. With `x_i = u_i / a_i` and
`y_j = v_j / conj b_j` as in `sum_sum_mul_norm_sq_sub_of_transport`, the weighted sum
`∑ T_ij ‖x_i − y_j‖²` vanishes, so `x_i = y_j` wherever `T_ij > 0`. Row `i` carries the mass
`|a_i|² > 0`, so some `T_ij > 0`, and then `‖x_i‖² = ⟨x_i, y_j⟩ = 1`; likewise `‖y_j‖ = 1`. On
every cell of `Ω`, `‖x_i − y_j‖² = 1 + 1 − 2 = 0`. Rigidity, applied to each coordinate, makes
all the rows equal, so `X_ij = a_i ‖x_i‖² b_j = a_i b_j` (`eq_vecMulVec_of_transport`). Since
`‖a bᵀ‖_S1 = ∑ |a_i|²` (`nuclearNorm_vecMulVec_of_transport`), `a bᵀ` is the unique minimum
(`isMinOn_vecMulVec_iff_of_transport`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **A plan inside a rigid pattern pins `a bᵀ`**, `review/flat_spectrum.md`, Theorem F: if no
entry of `a`, `b` vanishes, `Ω` carries a plan from `|a_i|²` to `|b_j|²` and `Ω` is rigid, every
`X` equal to `a bᵀ` on `Ω` with `‖X‖_S1 ≤ ∑ |a_i|²` is `a bᵀ`. -/
theorem eq_vecMulVec_of_transport {a b : n → ℂ} (ha : ∀ i, a i ≠ 0) (hb : ∀ j, b j ≠ 0)
    {Ω : Set (n × n)} {T : n → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2)
    (hle : nuclearNorm X ≤ ∑ i, ‖a i‖ ^ 2) : X = vecMulVec a b := by
  obtain ⟨r, U, V, hUV, hc⟩ := (nuclearNorm_isLeast X).1
  have hid := sum_sum_mul_norm_sq_sub_of_transport ha hb hT (U := U) (V := V) fun ij hij => by
    rw [← hUV]
    exact hX ij hij
  obtain ⟨hT0, hTΩ, hrow, hcol⟩ := hT
  set x : Matrix n (Fin r) ℂ := of fun i k => (a i)⁻¹ * U i k with hx
  set y : Matrix n (Fin r) ℂ := of fun j k => (star (b j))⁻¹ * V j k with hy
  have hxe (i : n) (k : Fin r) : (a i)⁻¹ * U i k = x i k := rfl
  have hye (j : n) (k : Fin r) : (star (b j))⁻¹ * V j k = y j k := rfl
  simp only [hxe, hye] at hid
  have hxy (i j : n) : (x * yᴴ) i j = (a i)⁻¹ * X i j * (b j)⁻¹ := by
    rw [hx, hy, rowDiv_mul_apply, ← hUV]
  have hΩ1 (ij : n × n) (hij : ij ∈ Ω) : (x * yᴴ) ij.1 ij.2 = 1 := by
    rw [hxy, hX ij hij, vecMulVec_apply]
    field_simp [ha ij.1, hb ij.2]
  have hsq (z w : ℂ) (hzw : ‖z - w‖ ^ 2 = 0) : z = w := by
    rwa [sq_eq_zero_iff, norm_eq_zero, sub_eq_zero] at hzw
  have hnn (i j : n) : ∀ k ∈ univ, 0 ≤ ‖x i k - y j k‖ ^ 2 := fun _ _ => sq_nonneg _
  have hTnn (i : n) : ∀ j ∈ univ, 0 ≤ T i j * ∑ k, ‖x i k - y j k‖ ^ 2 :=
    fun j _ => mul_nonneg (hT0 i j) (sum_nonneg (hnn i j))
  have h0 : ∑ i, ∑ j, T i j * ∑ k, ‖x i k - y j k‖ ^ 2 = 0 :=
    le_antisymm (by linarith) (sum_nonneg fun i _ => sum_nonneg (hTnn i))
  have hTD (i j : n) : T i j * ∑ k, ‖x i k - y j k‖ ^ 2 = 0 :=
    (sum_eq_zero_iff_of_nonneg (hTnn i)).mp ((sum_eq_zero_iff_of_nonneg fun i _ =>
      sum_nonneg (hTnn i)).mp h0 i (mem_univ i)) j (mem_univ j)
  have hmem (i j : n) (h : 0 < T i j) : (i, j) ∈ Ω := by
    by_contra hn
    exact h.ne' (hTΩ i j hn)
  have hrow0 (i j : n) (h : 0 < T i j) : ∑ k, ‖x i k - y j k‖ ^ 2 = 0 :=
    (mul_eq_zero.mp (hTD i j)).resolve_left h.ne'
  have hxy0 (i j : n) (h : 0 < T i j) (k : Fin r) : x i k = y j k :=
    hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn i j)).mp (hrow0 i j h) k (mem_univ k))
  have hexr (i : n) : ∃ j, 0 < T i j := by
    by_contra h
    have h' : ∑ j, T i j = 0 :=
      sum_eq_zero fun j _ => le_antisymm (not_lt.mp fun hj => h ⟨j, hj⟩) (hT0 i j)
    exact ha i (by simpa [hrow i] using h')
  have hexc (j : n) : ∃ i, 0 < T i j := by
    by_contra h
    have h' : ∑ i, T i j = 0 :=
      sum_eq_zero fun i _ => le_antisymm (not_lt.mp fun hi => h ⟨i, hi⟩) (hT0 i j)
    exact hb j (by simpa [hcol j] using h')
  have hu (i : n) : ∑ k, ‖x i k‖ ^ 2 = 1 := by
    obtain ⟨j, hj⟩ := hexr i
    have h := sum_norm_sq_sub_eq x y i j
    rw [hrow0 i j hj, hΩ1 (i, j) (hmem i j hj), Complex.one_re] at h
    have hb' : ∑ k, ‖y j k‖ ^ 2 = ∑ k, ‖x i k‖ ^ 2 :=
      sum_congr rfl fun k _ => by rw [hxy0 i j hj k]
    linarith
  have hv (j : n) : ∑ k, ‖y j k‖ ^ 2 = 1 := by
    obtain ⟨i, hi⟩ := hexc j
    rw [← hu i]
    exact sum_congr rfl fun k _ => by rw [hxy0 i j hi k]
  have hcell (ij : n × n) (hij : ij ∈ Ω) (k : Fin r) : x ij.1 k + -y ij.2 k = 0 := by
    have h := sum_norm_sq_sub_eq x y ij.1 ij.2
    rw [hu, hv, hΩ1 ij hij, Complex.one_re] at h
    rw [hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn ij.1 ij.2)).mp (by rw [h]; norm_num) k
      (mem_univ k)), add_neg_cancel]
  have hall (i j : n) (k : Fin r) : x i k = y j k := by
    have h := hΩ (fun i => x i k) (fun j => -y j k) (fun ij hij => hcell ij hij k) i j
    rwa [← sub_eq_add_neg, sub_eq_zero] at h
  ext i j
  have h1 : (x * yᴴ) i j = 1 := by
    rw [mul_apply, ← Complex.ofReal_one, ← hu i, Complex.ofReal_sum]
    refine sum_congr rfl fun k _ => ?_
    rw [conjTranspose_apply, ← hall i j k, Complex.star_def, Complex.mul_conj',
      Complex.ofReal_pow]
  rw [hxy] at h1
  rw [vecMulVec_apply]
  have hai := ha i
  have hbj := hb j
  field_simp at h1
  linear_combination h1

example := eq_vecMulVec_of_transport (n := Fin 2) (a := ![2, 1]) (b := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (Ω := {ij | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _) rigid_ne_one_zero (X := vecMulVec ![2, 1] ![2, 1]) (fun _ _ => rfl)
  (nuclearNorm_vecMulVec_le _ _ |>.trans_eq (by simp [Fin.sum_univ_two]))

/-- Under a plan from `|a_i|²` to `|b_j|²`, `‖a bᵀ‖_S1 = ∑ |a_i|²`. -/
theorem nuclearNorm_vecMulVec_of_transport {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) {Ω : Set (n × n)} {T : n → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T) :
    nuclearNorm (vecMulVec a b) = ∑ i, ‖a i‖ ^ 2 := by
  have hab : ∑ i, ‖a i‖ ^ 2 = ∑ j, ‖b j‖ ^ 2 := hT.sum_eq
  refine le_antisymm ?_ (sum_norm_sq_le_nuclearNorm_of_transport ha hb hT fun _ _ => rfl)
  have h := nuclearNorm_vecMulVec_le a b
  linarith

example := nuclearNorm_vecMulVec_of_transport (n := Fin 2) (a := ![2, 1]) (b := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (Ω := {ij | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _)

/-- **Theorem F, sufficiency**, `review/flat_spectrum.md`: if no entry of `a`, `b` vanishes and
the rigid pattern `Ω` carries a plan from `|a_i|²` to `|b_j|²`, then `a bᵀ` is the unique
minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_transport {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) {Ω : Set (n × n)} {T : n → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b := by
  have hK := nuclearNorm_vecMulVec_of_transport ha hb hT
  rw [isMinOn_iff]
  refine ⟨fun hmin => eq_vecMulVec_of_transport ha hb hT hΩ hX
    ((hmin _ fun _ _ => rfl).trans hK.le), ?_⟩
  rintro rfl X' hX'
  rw [hK]
  exact sum_norm_sq_le_nuclearNorm_of_transport ha hb hT hX'

example := (isMinOn_vecMulVec_iff_of_transport (n := Fin 2) (a := ![2, 1]) (b := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (Ω := {ij | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _) rigid_ne_one_zero (X := vecMulVec ![2, 1] ![2, 1])
  fun _ _ => rfl).mpr rfl

/-- The rigid pattern `{(0, 0), (0, 1), (1, 1)}` carries the plan `[[1, 3], [0, 1]]` from
`|a_i|² = (4, 1)` to `|b_j|² = (1, 4)`, which no permutation gives, so it recovers
`(2, 1)(1, 2)ᵀ`. -/
example := (isMinOn_vecMulVec_iff_of_transport (n := Fin 2) (a := ![2, 1]) (b := ![1, 2])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (Ω := {ij | ij ≠ (1, 0)})
  (T := fun i j => ![![1, 3], ![0, 1]] i j)
  ⟨by norm_num [Fin.forall_fin_two], by simp,
    by norm_num [Fin.forall_fin_two, Fin.sum_univ_two],
    by norm_num [Fin.forall_fin_two, Fin.sum_univ_two]⟩
  rigid_ne_one_zero (X := vecMulVec ![2, 1] ![1, 2]) fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum
