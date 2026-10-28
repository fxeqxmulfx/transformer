module

public import Transformer.Grokking.FlatSpectrum.TwoDiagonals

/-!
# Rigidity is necessary for recovery

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). The
recovery theorem `isMinOn_onesMatrix_iff` asks that `r i + s j = 0` on `Ω` force
`r i + s j = 0` everywhere: the bipartite graph of the rows and columns, joined by the cells of
`Ω`, is connected. This hypothesis cannot be dropped. If `r i + s j = 0` on `Ω` and
`w = r i₀ + s j₀ ≠ 0`, the phases `u i = exp (π i Re (r i / w))` and `v j = exp (π i Re (s j / w))`
have modulus one, `u i v j = 1` on `Ω` and `u i₀ v j₀ = −1`. So `(u a)(v b)ᵀ` is a second
rank-one matrix with entries of modulus one, equal to `a bᵀ` on `Ω` and different from it: no
method that sees only the values on `Ω` tells the two apart (`exists_vecMulVec_ne_of_not_rigid`),
and as both have nuclear norm `N`, neither is the unique minimum
(`not_forall_isMinOn_iff_of_not_rigid`). For the `a + b` kernel modulo `p`, the diagonals
`b = a + s₁` and `b = a + s₂` fail rigidity when `g = s₂ − s₁` is not a unit: the multiples `H`
of `g` miss `1`, and `r = 1_H`, `s = −1_{s₁ + H}` vanish on both diagonals but not at
`(0, s₁ + 1)`. With `isMinOn_addKernel_iff_of_diagonals`, two diagonals recover `ψ(a + b)`
exactly when `s₂ − s₁` is a unit (`forall_isMinOn_addKernel_iff_isUnit`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- **Without rigidity a second matrix fits**, `review/flat_spectrum.md`, *Exact recovery of a
rank-one kernel*: if `|a i| = |b j| = 1`, `r i + s j = 0` on `Ω` and `r i₀ + s j₀ ≠ 0`, some
`a' b'ᵀ ≠ a bᵀ` with `|a' i| = |b' j| = 1` equals `a bᵀ` on `Ω`. -/
theorem exists_vecMulVec_ne_of_not_rigid {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)} {r s : n → ℂ}
    (hrs : ∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) {i₀ j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0) :
    ∃ a' b' : n → ℂ, (∀ i, ‖a' i‖ = 1) ∧ (∀ j, ‖b' j‖ = 1) ∧
      (∀ ij ∈ Ω, vecMulVec a' b' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) ∧
      vecMulVec a' b' ≠ vecMulVec a b := by
  set w := r i₀ + s j₀
  set u : n → ℂ := fun i => Complex.exp (↑(Real.pi * (r i / w).re) * Complex.I) with hu
  set v : n → ℂ := fun j => Complex.exp (↑(Real.pi * (s j / w).re) * Complex.I) with hv
  have huv (i j : n) : vecMulVec (u * a) (v * b) i j =
      Complex.exp (↑(Real.pi * ((r i + s j) / w).re) * Complex.I) * vecMulVec a b i j := by
    simp only [vecMulVec_apply, Pi.mul_apply, hu, hv, add_div, Complex.add_re, mul_add,
      Complex.ofReal_add, add_mul, Complex.exp_add]
    ring
  refine ⟨u * a, v * b, fun i => ?_, fun j => ?_, fun ij hij => ?_, fun h => ?_⟩
  · rw [Pi.mul_apply, norm_mul, hu, Complex.norm_exp_ofReal_mul_I, ha, one_mul]
  · rw [Pi.mul_apply, norm_mul, hv, Complex.norm_exp_ofReal_mul_I, hb, one_mul]
  · rw [huv, hrs ij hij, zero_div, Complex.zero_re, mul_zero, Complex.ofReal_zero, zero_mul,
      Complex.exp_zero, one_mul]
  · have h1 := congrFun (congrFun h i₀) j₀
    rw [huv, div_self h₀, Complex.one_re, mul_one, Complex.exp_pi_mul_I] at h1
    have h2 : (2 : ℂ) * vecMulVec a b i₀ j₀ = 0 := by linear_combination -h1
    have h3 : ‖vecMulVec a b i₀ j₀‖ = 1 := by rw [vecMulVec_apply, norm_mul, ha, hb, one_mul]
    rcases mul_eq_zero.mp h2 with h4 | h4
    · exact two_ne_zero h4
    · rw [h4, norm_zero] at h3
      exact zero_ne_one h3

example := exists_vecMulVec_ne_of_not_rigid (a := fun _ : Fin 2 => 1) (b := fun _ => 1)
  (by simp) (by simp) (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp)

/-- **Without rigidity the least nuclear norm does not recover**, `review/flat_spectrum.md`,
*Exact recovery of a rank-one kernel*: if `|a i| = |b j| = 1`, `r i + s j = 0` on `Ω` and
`r i₀ + s j₀ ≠ 0`, then `a bᵀ` is not the unique minimum of the nuclear norm among the
matrices equal to it on `Ω`. -/
theorem not_forall_isMinOn_iff_of_not_rigid {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)} {r s : n → ℂ}
    (hrs : ∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) {i₀ j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0) :
    ¬ ∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
        X = vecMulVec a b) := by
  intro H
  obtain ⟨a', b', ha', hb', hmem, hne⟩ := exists_vecMulVec_ne_of_not_rigid ha hb hrs h₀
  have hK := isMinOn_iff.mp ((H _ fun _ _ => rfl).mpr rfl)
  refine hne ((H _ hmem).mp (isMinOn_iff.mpr fun X hX => ?_))
  rw [nuclearNorm_vecMulVec_of_norm_eq_one ha' hb', ← nuclearNorm_vecMulVec_of_norm_eq_one ha hb]
  exact hK X hX

example := not_forall_isMinOn_iff_of_not_rigid (a := fun _ : Fin 2 => 1) (b := fun _ => 1)
  (by simp) (by simp) (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp)

/-- **Two diagonals recover `ψ(a + b)` exactly when their offset is a unit**,
`review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*: if `Ω` is the union of the
diagonals `b = a + s₁` and `b = a + s₂` of `ℤ/p × ℤ/p`, the `a + b` kernel is the unique minimum
of the nuclear norm among the matrices equal to it on `Ω` if and only if `s₂ − s₁` is a unit. -/
theorem forall_isMinOn_addKernel_iff_isUnit {p : ℕ} [NeZero p] {s₁ s₂ : ZMod p}
    {Ω : Set (ZMod p × ZMod p)} (hΩ : ∀ a b, (a, b) ∈ Ω ↔ b = a + s₁ ∨ b = a + s₂) :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
        X = addKernel p)) ↔ IsUnit (s₂ - s₁) := by
  classical
  refine ⟨fun H => ?_, fun hs X hX => isMinOn_addKernel_iff_of_diagonals hs
    (fun a => (hΩ a _).mpr (Or.inl rfl)) (fun a => (hΩ a _).mpr (Or.inr rfl)) hX⟩
  by_contra hs
  set G := AddSubgroup.zmultiples (s₂ - s₁)
  have h1 : (1 : ZMod p) ∉ G := fun h => by
    obtain ⟨k, hk⟩ := AddSubgroup.mem_zmultiples_iff.mp h
    rw [zsmul_eq_mul] at hk
    exact hs (IsUnit.of_mul_eq_one_right _ hk)
  have hG (a : ZMod p) : a + s₂ - s₁ ∈ G ↔ a ∈ G := by
    rw [add_sub_assoc]
    exact G.add_mem_cancel_right (AddSubgroup.mem_zmultiples _)
  rw [addKernel_eq_vecMulVec] at H
  refine not_forall_isMinOn_iff_of_not_rigid (a := ZMod.stdAddChar) (b := ZMod.stdAddChar)
    (by simp) (by simp) (r := fun i => if i ∈ G then 1 else 0)
    (s := fun j => if j - s₁ ∈ G then -1 else 0) (fun ij hij => ?_) (i₀ := 0) (j₀ := s₁ + 1)
    ?_ H
  · obtain ⟨a, b⟩ := ij
    rcases (hΩ a b).mp hij with rfl | rfl
    · simp only [add_sub_cancel_right]
      split_ifs <;> norm_num
    · simp only [hG]
      split_ifs <;> norm_num
  · simp only [G.zero_mem, ↓reduceIte, add_sub_cancel_left, h1]
    norm_num

example := (forall_isMinOn_addKernel_iff_isUnit (p := 3) (s₁ := 0) (s₂ := 1)
  (Ω := {ij | ij.2 = ij.1 + 0 ∨ ij.2 = ij.1 + 1}) fun _ _ => Iff.rfl).mpr (by simp)

example := (forall_isMinOn_addKernel_iff_isUnit (p := 4) (s₁ := 0) (s₂ := 2)
  (Ω := {ij | ij.2 = ij.1 + 0 ∨ ij.2 = ij.1 + 2}) fun _ _ => Iff.rfl).not.mpr (by
    rw [sub_zero, show (2 : ZMod 4) = ((2 : ℕ) : ZMod 4) from rfl, ZMod.isUnit_iff_coprime]
    decide)

end Transformer.Grokking.FlatSpectrum
