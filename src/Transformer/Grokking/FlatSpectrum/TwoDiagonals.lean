module

public import Transformer.Grokking.FlatSpectrum.TwoPermutations

/-!
# Two diagonals recover the `a + b` kernel

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). For `a, b`
with entries of modulus one, `Y ↦ diag a · Y · diag b` keeps the nuclear norm
(`nuclearNorm_isometry`), sends `J` to `K = a bᵀ`, and maps the matrices equal to `J` on `Ω`
onto those equal to `K` on `Ω`, with inverse `X ↦ diag (star a) · X · diag (star b)`. So
`isMinOn_onesMatrix_iff_of_cycle` carries over to `K` (`isMinOn_vecMulVec_iff_of_cycle`).
The kernel `ψ(a + b) = ψ(a) ψ(b)` of addition modulo `p` is such a `K`. Its diagonals
`b = a + s₁` and `b = a + s₂` are the graphs of `σ = (· + s₁)` and `τ = (· + s₂)`, and
`τ σ⁻¹ = (· + (s₂ − s₁))` is one `p`-cycle, `e l = l (s₂ − s₁)`, when `s₂ − s₁` is a unit.
Then the completion of least nuclear norm of these `2p` values is `ψ(a + b)` itself
(`isMinOn_addKernel_iff_of_diagonals`); for prime `p` any two distinct diagonals do
(`isMinOn_addKernel_iff_of_ne`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Rescaling rows and columns by entries of modulus one keeps the nuclear norm:
`‖diag a · Y · diag b‖_S1 = ‖Y‖_S1` when `|a i| = |b j| = 1`. -/
lemma nuclearNorm_diagonal_mul_mul_diagonal {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) (Y : Matrix n n ℂ) :
    nuclearNorm (diagonal a * Y * diagonal b) = nuclearNorm Y := by
  have hu (c : n → ℂ) (hc : ∀ i, ‖c i‖ = 1) : (diagonal c)ᴴ * diagonal c = 1 := by
    rw [diagonal_conjTranspose, diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext i
    rw [Pi.star_apply, Complex.star_def, Complex.conj_mul', hc i]
    norm_num
  have h := nuclearNorm_isometry (hu a ha)
    (hu (star b) fun j => by rw [Pi.star_apply, norm_star, hb j]) Y
  rwa [diagonal_conjTranspose, star_star] at h

example := nuclearNorm_diagonal_mul_mul_diagonal (a := fun _ : Fin 2 => -1)
  (b := fun _ => Complex.I) (by simp) (by simp) 1

/-- `diag a · J · diag b = a bᵀ`. -/
lemma diagonal_mul_onesMatrix_mul_diagonal (a b : n → ℂ) :
    diagonal a * onesMatrix n * diagonal b = vecMulVec a b := by
  ext i j
  rw [mul_diagonal, diagonal_mul, vecMulVec_apply, onesMatrix, of_apply, mul_one]

/-- `‖a bᵀ‖_S1 = N` when `|a i| = |b j| = 1`. -/
theorem nuclearNorm_vecMulVec_of_norm_eq_one {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) : nuclearNorm (vecMulVec a b) = Fintype.card n := by
  rw [← diagonal_mul_onesMatrix_mul_diagonal, nuclearNorm_diagonal_mul_mul_diagonal ha hb,
    nuclearNorm_onesMatrix]

example := nuclearNorm_vecMulVec_of_norm_eq_one (a := fun _ : Fin 2 => -1)
  (b := fun _ => Complex.I) (by simp) (by simp)

variable {N : ℕ} [NeZero N]

/-- **Two permutations recover a rank-one matrix of modulus-one entries**,
`review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*: if `|a i| = |b j| = 1`, `Ω`
contains the graphs of `σ` and `τ` and `τ σ⁻¹` is one `N`-cycle, then `a bᵀ` is the unique
minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_cycle {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) {Ω : Set (n × n)} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hτ : ∀ i, (i, τ i) ∈ Ω) {X : Matrix n n ℂ}
    (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b := by
  have hsa (i : n) : star (a i) * a i = 1 := by
    rw [Complex.star_def, Complex.conj_mul', ha i]
    norm_num
  have hsb (j : n) : star (b j) * b j = 1 := by
    rw [Complex.star_def, Complex.conj_mul', hb j]
    norm_num
  have ha' (i : n) : ‖star a i‖ = 1 := by rw [Pi.star_apply, norm_star, ha i]
  have hb' (j : n) : ‖star b j‖ = 1 := by rw [Pi.star_apply, norm_star, hb j]
  -- `Φ Y = diag a · Y · diag b` and its inverse `Ψ X = diag (star a) · X · diag (star b)`
  have hΦ (Y : Matrix n n ℂ) (i j : n) :
      (diagonal a * Y * diagonal b) i j = a i * Y i j * b j := by
    rw [mul_diagonal, diagonal_mul]
  have hΨ (X' : Matrix n n ℂ) (i j : n) :
      (diagonal (star a) * X' * diagonal (star b)) i j = star (a i) * X' i j * star (b j) := by
    rw [mul_diagonal, diagonal_mul, Pi.star_apply, Pi.star_apply]
  have hΦΨ (X' : Matrix n n ℂ) :
      diagonal a * (diagonal (star a) * X' * diagonal (star b)) * diagonal b = X' := by
    ext i j
    rw [hΦ, hΨ]
    linear_combination (X' i j * star (b j) * b j) * hsa i + X' i j * hsb j
  have hΦJ := diagonal_mul_onesMatrix_mul_diagonal a b
  have hΦmem (Y : Matrix n n ℂ) (hY : ∀ ij ∈ Ω, Y ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
      ∀ ij ∈ Ω, (diagonal a * Y * diagonal b) ij.1 ij.2 = vecMulVec a b ij.1 ij.2 :=
    fun ij hij => by rw [hΦ, hY ij hij, ← hΦJ, hΦ]
  have hΨmem (X' : Matrix n n ℂ) (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
      ∀ ij ∈ Ω, (diagonal (star a) * X' * diagonal (star b)) ij.1 ij.2 =
        onesMatrix n ij.1 ij.2 := fun ij hij => by
    rw [hΨ, hX' ij hij, vecMulVec_apply, onesMatrix, of_apply]
    linear_combination (star (b ij.2) * b ij.2) * hsa ij.1 + hsb ij.2
  have hJ := isMinOn_onesMatrix_iff_of_cycle e he hσ hτ (X := onesMatrix n) fun _ _ => rfl
  rw [isMinOn_iff] at hJ ⊢
  constructor
  · intro hmin
    have h := (isMinOn_onesMatrix_iff_of_cycle e he hσ hτ (hΨmem X hX)).mp <| isMinOn_iff.mpr
      fun Y hY => by
        rw [nuclearNorm_diagonal_mul_mul_diagonal ha' hb',
          ← nuclearNorm_diagonal_mul_mul_diagonal ha hb Y]
        exact hmin _ (hΦmem Y hY)
    rw [← hΦΨ X, h, hΦJ]
  · intro hXK X' hX'
    rw [hXK, ← hΦJ, nuclearNorm_diagonal_mul_mul_diagonal ha hb,
      ← nuclearNorm_diagonal_mul_mul_diagonal ha' hb' X']
    exact hJ.mpr rfl _ (hΨmem X' hX')

example := (isMinOn_vecMulVec_iff_of_cycle (N := 2) (a := fun _ : Fin 2 => -1)
  (b := fun _ => Complex.I) (by simp) (by simp) (σ := Equiv.refl (Fin 2)) (τ := Equiv.swap 0 1)
  (Equiv.refl _) (by decide) (Ω := Set.univ) (fun _ => Set.mem_univ _) (fun _ => Set.mem_univ _)
  (X := vecMulVec (fun _ => -1) fun _ => Complex.I) fun _ _ => rfl).mpr rfl

/-- **Two diagonals recover `ψ(a + b)`**, `review/flat_spectrum.md`, *Exact recovery of a
rank-one kernel*: if `Ω` contains the diagonals `b = a + s₁` and `b = a + s₂` of `ℤ/p × ℤ/p`
and `s₂ − s₁` is a unit, the `a + b` kernel is the unique minimum of the nuclear norm among the
matrices equal to it on `Ω`. -/
theorem isMinOn_addKernel_iff_of_diagonals {p : ℕ} [NeZero p] {s₁ s₂ : ZMod p}
    (hs : IsUnit (s₂ - s₁)) {Ω : Set (ZMod p × ZMod p)} (h₁ : ∀ a, (a, a + s₁) ∈ Ω)
    (h₂ : ∀ a, (a, a + s₂) ∈ Ω) {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = addKernel p ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
      X = addKernel p := by
  rw [addKernel_eq_vecMulVec] at hX ⊢
  refine isMinOn_vecMulVec_iff_of_cycle (by simp) (by simp) (σ := Equiv.addRight s₁)
    (τ := Equiv.addRight s₂) hs.unit.mulRight (fun l => ?_) h₁ h₂ hX
  simp only [Equiv.addRight_symm, Equiv.coe_addRight, Units.mulRight_apply, IsUnit.unit_spec]
  ring

example := (isMinOn_addKernel_iff_of_diagonals (p := 4) (s₁ := 0) (s₂ := 1) (by simp)
  (Ω := {ij | ij.2 = ij.1 + 0 ∨ ij.2 = ij.1 + 1}) (fun _ => Or.inl rfl) (fun _ => Or.inr rfl)
  (X := addKernel 4) fun _ _ => rfl).mpr rfl

/-- **Two distinct diagonals recover `ψ(a + b)` for prime `p`**, `review/flat_spectrum.md`,
*Exact recovery of a rank-one kernel*: from its `2p` values on the diagonals `b = a + s₁` and
`b = a + s₂`, `s₁ ≠ s₂`, the completion of least nuclear norm is the `a + b` kernel. -/
theorem isMinOn_addKernel_iff_of_ne {p : ℕ} [Fact p.Prime] {s₁ s₂ : ZMod p} (hs : s₁ ≠ s₂)
    {Ω : Set (ZMod p × ZMod p)} (h₁ : ∀ a, (a, a + s₁) ∈ Ω) (h₂ : ∀ a, (a, a + s₂) ∈ Ω)
    {X : Matrix (ZMod p) (ZMod p) ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = addKernel p ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
      X = addKernel p :=
  isMinOn_addKernel_iff_of_diagonals (sub_ne_zero.mpr hs.symm).isUnit h₁ h₂ hX

example := (isMinOn_addKernel_iff_of_ne (p := 3) (s₁ := 0) (s₂ := 2) (by decide)
  (Ω := {ij | ij.2 = ij.1 + 0 ∨ ij.2 = ij.1 + 2}) (fun _ => Or.inl rfl) (fun _ => Or.inr rfl)
  (X := addKernel 3) fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum
