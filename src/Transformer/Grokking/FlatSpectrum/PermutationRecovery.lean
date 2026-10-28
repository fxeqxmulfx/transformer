module

public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.TwoDiagonals

/-!
# A permutation inside a rigid pattern recovers `J`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem E (added).
Let `Ω` contain the graph of a permutation `σ`, and let `X` equal `J` on `Ω`. Lemma 1
(`nuclearNorm_isLeast`) writes `X = U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1`; let `u_i` and `v_j`
be the rows of `U` and `V`. Since `⟨u_i, v_{σ i}⟩ = 1`,
`∑_i ‖u_i − v_{σ i}‖² = ‖U‖² + ‖V‖² − 2N` (`sum_sum_norm_sq_sub_perm`), so `‖X‖_S1 ≥ N`
(`card_le_nuclearNorm_of_perm`). If `‖X‖_S1 ≤ N`, then `u_i = v_{σ i}`, so every row has unit
length, and `⟨u_i, v_j⟩ = 1` forces `u_i = v_j` on every cell of `Ω`. Rigidity, applied to each
coordinate, makes all the rows equal, so `X = J` (`eq_onesMatrix_of_perm`). Hence `J` is the
unique minimum (`isMinOn_onesMatrix_iff_of_perm`), and so is every `a bᵀ` with
`|a i| = |b j| = 1` (`isMinOn_vecMulVec_iff_of_perm`), `ψ(a + b)` among them
(`isMinOn_addKernel_iff_of_perm`). No certificate with `c > 1` is needed: the pattern
`{(0, 0), (0, 1), (1, 1)}` of the examples has a row with a single cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- `‖u − v‖² = ‖u‖² + ‖v‖² − 2 Re ⟨u, v⟩` for the rows `u = U i` and `v = V j`, with
`⟨u, v⟩ = (U Vᴴ)_ij`. -/
lemma sum_norm_sq_sub_eq {r : Type*} [Fintype r] (U V : Matrix n r ℂ) (i j : n) :
    ∑ k, ‖U i k - V j k‖ ^ 2 =
      ∑ k, ‖U i k‖ ^ 2 + ∑ k, ‖V j k‖ ^ 2 - 2 * ((U * Vᴴ) i j).re := by
  rw [mul_apply, Complex.re_sum, mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  refine sum_congr rfl fun k _ => ?_
  simp only [conjTranspose_apply, Complex.star_def, ← Complex.normSq_eq_norm_sq,
    Complex.normSq_sub]

omit [DecidableEq n] in
/-- Along a permutation `σ` with `(U Vᴴ)_{i, σ i} = 1`,
`∑_i ‖u_i − v_{σ i}‖² = ‖U‖² + ‖V‖² − 2N`. -/
lemma sum_sum_norm_sq_sub_perm {r : Type*} [Fintype r] (U V : Matrix n r ℂ)
    (σ : Equiv.Perm n) (h : ∀ i, (U * Vᴴ) i (σ i) = 1) :
    ∑ i, ∑ k, ‖U i k - V (σ i) k‖ ^ 2 =
      frobeniusSq U + frobeniusSq V - 2 * Fintype.card n := by
  simp only [sum_norm_sq_sub_eq, h, Complex.one_re, mul_one]
  rw [sum_sub_distrib, sum_add_distrib, Equiv.sum_comp σ fun j => ∑ k, ‖V j k‖ ^ 2, sum_const,
    card_univ, nsmul_eq_mul, frobeniusSq, frobeniusSq]
  ring

example := sum_sum_norm_sq_sub_perm (n := Fin 2) (r := Fin 2) 1 1 1 fun i => by simp

/-- **A permutation inside the pattern bounds the norm**, `review/flat_spectrum.md`,
Theorem E: if `Ω` contains the graph of `σ`, every `X` equal to `J` on `Ω` has
`N ≤ ‖X‖_S1`. -/
theorem card_le_nuclearNorm_of_perm {Ω : Set (n × n)} {σ : Equiv.Perm n}
    (hσ : ∀ i, (i, σ i) ∈ Ω) {X : Matrix n n ℂ}
    (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
    (Fintype.card n : ℝ) ≤ nuclearNorm X := by
  obtain ⟨r, U, V, hUV, hc⟩ := (nuclearNorm_isLeast X).1
  have h := sum_sum_norm_sq_sub_perm U V σ fun i => by rw [← hUV]; exact hX _ (hσ i)
  have h0 : 0 ≤ ∑ i, ∑ k, ‖U i k - V (σ i) k‖ ^ 2 :=
    sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _
  linarith

example := card_le_nuclearNorm_of_perm (n := Fin 2) (Ω := Set.univ) (σ := 1)
  (fun _ => Set.mem_univ _) (X := onesMatrix (Fin 2)) fun _ _ => rfl

/-- **A permutation inside a rigid pattern pins `J`**, `review/flat_spectrum.md`, Theorem E:
if `Ω` contains the graph of `σ` and is rigid — `r_i + s_j = 0` on `Ω` only if everywhere —
then every `X` equal to `J` on `Ω` with `‖X‖_S1 ≤ N` is `J`. -/
theorem eq_onesMatrix_of_perm {Ω : Set (n × n)} {σ : Equiv.Perm n} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2)
    (hle : nuclearNorm X ≤ Fintype.card n) : X = onesMatrix n := by
  obtain ⟨r, U, V, hUV, hc⟩ := (nuclearNorm_isLeast X).1
  have hΩ1 (ij : n × n) (hij : ij ∈ Ω) : (U * Vᴴ) ij.1 ij.2 = 1 := by
    rw [← hUV]
    exact hX ij hij
  have hsq (z w : ℂ) (hzw : ‖z - w‖ ^ 2 = 0) : z = w := by
    rwa [sq_eq_zero_iff, norm_eq_zero, sub_eq_zero] at hzw
  have hnn (i j : n) : ∀ k ∈ univ, 0 ≤ ‖U i k - V j k‖ ^ 2 := fun _ _ => sq_nonneg _
  have hσ0 : ∑ i, ∑ k, ‖U i k - V (σ i) k‖ ^ 2 = 0 := by
    have h := sum_sum_norm_sq_sub_perm U V σ fun i => hΩ1 _ (hσ i)
    exact le_antisymm (by linarith) (sum_nonneg fun i _ => sum_nonneg (hnn i (σ i)))
  have hrow0 (i : n) : ∑ k, ‖U i k - V (σ i) k‖ ^ 2 = 0 :=
    (sum_eq_zero_iff_of_nonneg fun i _ => sum_nonneg (hnn i (σ i))).mp hσ0 i (mem_univ i)
  have hUσ (i : n) (k : Fin r) : U i k = V (σ i) k :=
    hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn i (σ i))).mp (hrow0 i) k (mem_univ k))
  have hu (i : n) : ∑ k, ‖U i k‖ ^ 2 = 1 := by
    have h := sum_norm_sq_sub_eq U V i (σ i)
    rw [hrow0 i, hΩ1 _ (hσ i), Complex.one_re] at h
    have hb : ∑ k, ‖V (σ i) k‖ ^ 2 = ∑ k, ‖U i k‖ ^ 2 := sum_congr rfl fun k _ => by rw [hUσ]
    linarith
  have hv (j : n) : ∑ k, ‖V j k‖ ^ 2 = 1 := by
    rw [← hu (σ.symm j)]
    exact sum_congr rfl fun k _ => by rw [hUσ, Equiv.apply_symm_apply]
  have hcell (ij : n × n) (hij : ij ∈ Ω) (k : Fin r) : U ij.1 k + -V ij.2 k = 0 := by
    have h := sum_norm_sq_sub_eq U V ij.1 ij.2
    rw [hu, hv, hΩ1 ij hij, Complex.one_re] at h
    rw [hsq _ _ ((sum_eq_zero_iff_of_nonneg (hnn ij.1 ij.2)).mp (by rw [h]; norm_num) k
      (mem_univ k)), add_neg_cancel]
  have hall (i j : n) (k : Fin r) : U i k = V j k := by
    have h := hΩ (fun i => U i k) (fun j => -V j k) (fun ij hij => hcell ij hij k) i j
    rwa [← sub_eq_add_neg, sub_eq_zero] at h
  ext i j
  rw [hUV, mul_apply, onesMatrix, of_apply, ← Complex.ofReal_one, ← hu i, Complex.ofReal_sum]
  refine sum_congr rfl fun k _ => ?_
  rw [conjTranspose_apply, ← hall i j k, Complex.star_def, Complex.mul_conj', Complex.ofReal_pow]

example := eq_onesMatrix_of_perm (n := Fin 2) (Ω := Set.univ) (σ := 1) (fun _ => Set.mem_univ _)
  (fun _ _ h i j => h (i, j) (Set.mem_univ _)) (X := onesMatrix (Fin 2)) (fun _ _ => rfl)
  nuclearNorm_onesMatrix.le

/-- **Exact recovery from a rigid pattern with a permutation**, `review/flat_spectrum.md`,
Theorem E: if `Ω` contains the graph of a permutation and is rigid, `J` is the unique minimum
of the nuclear norm among the matrices equal to `J` on `Ω`. -/
theorem isMinOn_onesMatrix_iff_of_perm {Ω : Set (n × n)} {σ : Equiv.Perm n}
    (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2} X ↔
      X = onesMatrix n := by
  rw [isMinOn_iff]
  refine ⟨fun hmin => eq_onesMatrix_of_perm hσ hΩ hX
    ((hmin _ fun _ _ => rfl).trans nuclearNorm_onesMatrix.le), ?_⟩
  rintro rfl X' hX'
  rw [nuclearNorm_onesMatrix]
  exact card_le_nuclearNorm_of_perm hσ hX'

/-- The pattern `{(0, 0), (0, 1), (1, 1)}` is rigid. -/
lemma rigid_ne_one_zero (r s : Fin 2 → ℂ)
    (h : ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)}, r ij.1 + s ij.2 = 0) :
    ∀ i j, r i + s j = 0 := by
  have h00 : r 0 + s 0 = 0 := h (0, 0) (by simp)
  have h01 : r 0 + s 1 = 0 := h (0, 1) (by simp)
  have h11 : r 1 + s 1 = 0 := h (1, 1) (by simp)
  simp only [Fin.forall_fin_two]
  exact ⟨⟨h00, h01⟩, by linear_combination h11 - h01 + h00, h11⟩

example := rigid_ne_one_zero (fun _ => 1) (fun _ => -1) fun _ _ => by norm_num

example := (isMinOn_onesMatrix_iff_of_perm (n := Fin 2) (Ω := {ij | ij ≠ (1, 0)}) (σ := 1)
  (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩) rigid_ne_one_zero (X := onesMatrix (Fin 2))
  fun _ _ => rfl).mpr rfl

/-- **Rescaled kernels**, `review/flat_spectrum.md`, Theorem E: under the same hypotheses, every
`a bᵀ` with `|a i| = |b j| = 1` is the unique minimum of the nuclear norm among the matrices
equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_perm {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)} {σ : Equiv.Perm n} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b :=
  isMinOn_vecMulVec_iff_of_onesMatrix ha hb (fun _ hY => isMinOn_onesMatrix_iff_of_perm hσ hΩ hY)
    hX

example := (isMinOn_vecMulVec_iff_of_perm (n := Fin 2) (a := fun _ => 1) (b := fun _ => 1)
  (by simp) (by simp) (Ω := {ij | ij ≠ (1, 0)}) (σ := 1) (Fin.forall_fin_two.mpr
  ⟨by simp, by simp⟩) rigid_ne_one_zero (X := vecMulVec (fun _ => 1) fun _ => 1)
  fun _ _ => rfl).mpr rfl

/-- **`ψ(a + b)` from a rigid pattern with a permutation**, `review/flat_spectrum.md`,
Theorem E: if `Ω ⊆ ℤ/p × ℤ/p` contains the graph of a permutation and is rigid, the `a + b`
kernel is the unique minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_addKernel_iff_of_perm {p : ℕ} [NeZero p] {Ω : Set (ZMod p × ZMod p)}
    {σ : Equiv.Perm (ZMod p)} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hΩ : ∀ r s : ZMod p → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix (ZMod p) (ZMod p) ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = addKernel p ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
      X = addKernel p := by
  rw [addKernel_eq_vecMulVec] at hX ⊢
  exact isMinOn_vecMulVec_iff_of_perm (by simp) (by simp) hσ hΩ hX

example := (isMinOn_addKernel_iff_of_perm (p := 3) (Ω := Set.univ) (σ := 1)
  (fun _ => Set.mem_univ _) (fun _ _ h i j => h (i, j) (Set.mem_univ _)) (X := addKernel 3)
  fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum
