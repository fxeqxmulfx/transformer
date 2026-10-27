module

public import Transformer.Grokking.FlatSpectrum.OneMissing
public import Transformer.Grokking.FlatSpectrum.Kernels

/-!
# The one-entry fill of a rank-one kernel is the truth

Source: `review/flat_spectrum.md`, Corollary: "For a+b the fill is +K_ij, which is right."
Let `K = a bᵀ` with `|a_k| = |b_l| = 1`, `N ≥ 2` and `r = √(N − 1)`. In the frames
`J₁ = [e_i, (a − a_i e_i)/r]` and `J₂ = [e_j, (b̄ − b̄_j e_j)/r]`, `a = J₁ (a_i, r)` and
`b̄ = J₂ (b̄_j, r)`, so `K + t e_ij = J₁ B J₂*` with `B = [[a_i b_j + t, a_i r], [r b_j, r²]]`:
`det B = t r²` and `‖B‖_S2² = |a_i b_j + t|² + N² − 1`. Hence
`‖K + t e_ij‖_S1 = √(|K_ij + t|² + N² − 1 + 2 (N − 1) |t|) ≥ √(N² + |t|²)`, and the only
minimum-norm fill is `t = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The one-entry change of a unimodular rank-one `K = a bᵀ`, `N ≥ 2`:
`‖K + t e_ij‖_S1 = √(|a_i b_j + t|² + N² − 1 + 2 (N − 1) |t|)`, `review/flat_spectrum.md`,
Corollary. -/
theorem nuclearNorm_vecMulVec_add_single {a b : n → ℂ} (ha : ∀ k, ‖a k‖ = 1)
    (hb : ∀ l, ‖b l‖ = 1) (hN : 2 ≤ Fintype.card n) (i j : n) (t : ℂ) :
    nuclearNorm (vecMulVec a b + t • single i j 1) =
      √(‖a i * b j + t‖ ^ 2 + (Fintype.card n : ℝ) ^ 2 - 1 +
        2 * ((Fintype.card n : ℝ) - 1) * ‖t‖) := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  set r : ℝ := √((Fintype.card n : ℝ) - 1) with hrdef
  have hr0 : 0 ≤ r := Real.sqrt_nonneg _
  have hr2 : r ^ 2 = (Fintype.card n : ℝ) - 1 := Real.sq_sqrt (by linarith)
  have h₁ : (frame a i)ᴴ * frame a i = 1 := conjTranspose_mul_frame (ha i) (by simp [ha]) hN
  have h₂ : (frame (fun l => star (b l)) j)ᴴ * frame (fun l => star (b l)) j = 1 :=
    conjTranspose_mul_frame (by simp [hb]) (by simp [hb]) hN
  have hα : frame a i *ᵥ ![a i, (r : ℂ)] = a := by
    funext k
    simpa [mulVec, dotProduct, Fin.sum_univ_two] using frame_reconstruct (v := a) (j := i) hN k
  have hβ : ![b j, (r : ℂ)] ᵥ* (frame (fun l => star (b l)) j)ᴴ = b := by
    funext l
    have hr : star (r : ℂ) = r := Complex.conj_ofReal r
    have h := congrArg star (frame_reconstruct (v := fun l => star (b l)) (j := j) hN l)
    rw [← hrdef] at h
    simp only [star_add, star_mul', star_star, hr] at h
    simp only [vecMul, dotProduct, Fin.sum_univ_two, conjTranspose_apply, Matrix.cons_val_zero,
      Matrix.cons_val_one]
    linear_combination h
  have hsingle : frame a i * single (0 : Fin 2) (0 : Fin 2) (1 : ℂ) *
      (frame (fun l => star (b l)) j)ᴴ = single i j 1 := by
    ext k l
    simp only [mul_apply, single_apply, conjTranspose_apply, Fin.sum_univ_two, frame_zero]
    rcases eq_or_ne k i with rfl | hk
    · rcases eq_or_ne l j with rfl | hl
      · simp
      · simp [hl, hl.symm]
    · simp [hk, hk.symm]
  have hX : vecMulVec a b + t • single i j 1 = frame a i *
      (vecMulVec ![a i, (r : ℂ)] ![b j, (r : ℂ)] + t • single 0 0 1) *
        (frame (fun l => star (b l)) j)ᴴ := by
    rw [Matrix.mul_add, Matrix.add_mul, mul_vecMulVec, vecMulVec_mul, hα, hβ, Matrix.mul_smul,
      Matrix.smul_mul, hsingle]
  have hdet : (vecMulVec ![a i, (r : ℂ)] ![b j, (r : ℂ)] + t • single 0 0 1).det =
      t * ((r ^ 2 : ℝ) : ℂ) := by
    rw [det_fin_two]
    simp
    ring
  have hfro : frobeniusSq (vecMulVec ![a i, (r : ℂ)] ![b j, (r : ℂ)] + t • single 0 0 1) =
      ‖a i * b j + t‖ ^ 2 + 2 * r ^ 2 + (r ^ 2) ^ 2 := by
    simp [frobeniusSq, Fin.sum_univ_two, single_apply, ha, hb, abs_of_nonneg hr0]
    ring
  rw [hX, nuclearNorm_isometry h₁ h₂, nuclearNorm_fin_two, hdet, hfro, norm_mul,
    Complex.norm_real, Real.norm_of_nonneg (sq_nonneg r), hr2]
  ring_nf

example := nuclearNorm_vecMulVec_add_single (a := fun _ : Fin 2 => 1) (b := fun _ => 1)
  (by simp) (by simp) le_rfl 0 0 1

omit [Fintype n] in
/-- A matrix agreeing with `K` off `(i, j)` is `K + (Y_ij − K_ij) e_ij`. -/
lemma eq_add_single_of_mem {K Y : Matrix n n ℂ} {i j : n}
    (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = K kl.1 kl.2) :
    Y = K + (Y i j - K i j) • single i j 1 := by
  ext k l
  by_cases h : k = i ∧ l = j
  · obtain ⟨rfl, rfl⟩ := h
    simp
  · have hkl := hY (k, l) (by simpa using h)
    have hne : ¬(i = k ∧ j = l) := fun ⟨h1, h2⟩ => h ⟨h1.symm, h2.symm⟩
    simp [hne, hkl]

example := eq_add_single_of_mem (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Y := 1) (i := 0) (j := 0)
  (fun _ _ => rfl)

omit [Fintype n] in
/-- Changing the entry `(i, j)` keeps every other entry. -/
lemma add_smul_single_apply_of_mem {K : Matrix n n ℂ} {i j : n} (t : ℂ) :
    ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), (K + t • single i j 1 : Matrix n n ℂ) kl.1 kl.2 =
      K kl.1 kl.2 := by
  intro kl hkl
  have hne : ¬(i = kl.1 ∧ j = kl.2) := fun ⟨h1, h2⟩ => hkl (by simp [h1, h2])
  simp [hne]

/-- **Corollary** of `review/flat_spectrum.md` for rank one: with one entry missing, the
minimum-nuclear-norm completion of a unimodular `K = a bᵀ` (`N ≥ 2`) keeps the true entry,
`X_ij = K_ij`, and no other fill attains the minimum. -/
theorem isMinOn_vecMulVec_fill_iff {a b : n → ℂ} (ha : ∀ k, ‖a k‖ = 1) (hb : ∀ l, ‖b l‖ = 1)
    (hN : 2 ≤ Fintype.card n) {i j : n} {X : Matrix n n ℂ}
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), X kl.1 kl.2 = vecMulVec a b kl.1 kl.2) :
    IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = vecMulVec a b kl.1 kl.2} X ↔
      X i j = vecMulVec a b i j := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have hlow (Y : Matrix n n ℂ)
      (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = vecMulVec a b kl.1 kl.2) :
      √((Fintype.card n : ℝ) ^ 2 + ‖Y i j - vecMulVec a b i j‖ ^ 2) ≤ nuclearNorm Y := by
    have h := nuclearNorm_vecMulVec_add_single ha hb hN i j (Y i j - vecMulVec a b i j)
    rw [← eq_add_single_of_mem hY] at h
    rw [h]
    apply Real.sqrt_le_sqrt
    set t := Y i j - vecMulVec a b i j
    have h1 := abs_norm_sub_norm_le (a i * b j) (-t)
    rw [norm_neg, sub_neg_eq_add, norm_mul, ha, hb, one_mul] at h1
    have h2 : (1 - ‖t‖) ^ 2 ≤ ‖a i * b j + t‖ ^ 2 := by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) h1 2
    nlinarith [norm_nonneg t]
  have hK : nuclearNorm (vecMulVec a b) = Fintype.card n := by
    have h := nuclearNorm_vecMulVec_add_single ha hb hN i j 0
    simp only [zero_smul, add_zero, norm_zero, mul_zero, norm_mul, ha, hb, one_mul,
      one_pow, add_sub_cancel_left] at h
    rw [h, Real.sqrt_sq (Nat.cast_nonneg _)]
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h1 := (hlow X hX).trans ((hmin _ (fun _ _ => rfl)).trans hK.le)
    rw [Real.sqrt_le_left (Nat.cast_nonneg _)] at h1
    have h2 : ‖X i j - vecMulVec a b i j‖ ^ 2 ≤ 0 := by linarith
    simpa [sub_eq_zero] using le_antisymm h2 (sq_nonneg _)
  · intro hXij Y hY
    have hX0 : X = vecMulVec a b := by
      rw [eq_add_single_of_mem hX, hXij, sub_self, zero_smul, add_zero]
    rw [hX0, hK]
    refine le_trans ?_ (hlow Y hY)
    rw [Real.le_sqrt (Nat.cast_nonneg _) (by positivity)]
    exact le_add_of_nonneg_right (sq_nonneg _)

example := isMinOn_vecMulVec_fill_iff (a := fun _ : Fin 2 => (1 : ℂ)) (b := fun _ => 1)
  (by simp) (by simp) le_rfl (i := 0) (j := 0) (X := vecMulVec (fun _ => 1) (fun _ => 1))
  (fun _ _ => rfl)

/-- **Corollary** of `review/flat_spectrum.md` for `a + b`: with one cell `(a, b)` missing, the
minimum-nuclear-norm completion of `ψ(a + b)` keeps the true entry, and no other fill attains
the minimum. -/
theorem isMinOn_add_fill_iff {p : ℕ} [NeZero p] (hp : 2 ≤ p) {a b : ZMod p}
    {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), X kl.1 kl.2 = addKernel p kl.1 kl.2) :
    IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), Y kl.1 kl.2 = addKernel p kl.1 kl.2} X ↔
      X a b = addKernel p a b := by
  rw [addKernel_eq_vecMulVec] at hX ⊢
  exact isMinOn_vecMulVec_fill_iff (by simp) (by simp) (by rw [ZMod.card]; exact hp) hX

example := isMinOn_add_fill_iff (p := 5) (by norm_num) (a := 1) (b := 2) (X := addKernel 5)
  (fun _ _ => rfl)

end Transformer.Grokking.FlatSpectrum
