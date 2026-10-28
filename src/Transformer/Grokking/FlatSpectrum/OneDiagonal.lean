module

public import Transformer.Grokking.FlatSpectrum.TwoDiagonals

/-!
# One diagonal does not determine the `a + b` kernel

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let `Ω` be
the graph of one permutation `σ` and `|a i| = |b j| = 1`. The matrix
`Z = diag a · P_σ · diag b`, equal to `a i b j` on the graph and zero elsewhere, has
`Zᴴ Z = 1`: it is a contraction and `‖Z‖_S1 = N`. Every `X` equal to `K = a bᵀ` on the graph
pairs with it to `tr (Zᴴ X) = ∑_i |a i b (σ i)|² = N`, so `‖X‖_S1 ≥ N`. Both `K` and `Z`
attain `N`, and they differ once `N ≥ 2`: from one graph the completion of least nuclear norm
is not unique (`isMinOn_vecMulVec_and_graphMatrix`). For the `a + b` kernel one diagonal
`b = a + s`, `p` cells, is not enough (`exists_ne_isMinOn_addKernel_of_diagonal`), while two
are (`isMinOn_addKernel_iff_of_ne`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `diag a · P_σ · diag b`: the entries `a i b j` on the graph of `σ`, zero elsewhere. -/
noncomputable def graphMatrix (σ : Equiv.Perm n) (a b : n → ℂ) : Matrix n n ℂ :=
  of fun i j => if σ i = j then a i * b j else 0

/-- `diag a · P_σ · diag b` is unitary when `|a i| = |b j| = 1`. -/
lemma conjTranspose_graphMatrix_mul {σ : Equiv.Perm n} {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) : (graphMatrix σ a b)ᴴ * graphMatrix σ a b = 1 := by
  ext j k
  rw [mul_apply, ← Equiv.sum_comp σ.symm, sum_eq_single j (fun x _ hx => by simp [graphMatrix, hx])
    fun h => absurd (mem_univ j) h]
  simp only [conjTranspose_apply, graphMatrix, of_apply, Equiv.apply_symm_apply, ↓reduceIte,
    one_apply]
  split_ifs with h
  · subst h
    rw [Complex.star_def, Complex.conj_mul', norm_mul, ha, hb]
    norm_num
  · rw [mul_zero]

example := conjTranspose_graphMatrix_mul (σ := Equiv.swap (0 : Fin 2) 1) (a := fun _ => -1)
  (b := fun _ => Complex.I) (by simp) (by simp)

/-- **One graph is not enough**, `review/flat_spectrum.md`, *Exact recovery of a rank-one
kernel*: if `Ω` is the graph of `σ`, `|a i| = |b j| = 1` and `N ≥ 2`, then `a bᵀ` and the
different matrix `Z = diag a · P_σ · diag b`, equal to it on `Ω`, both minimize the nuclear
norm among the matrices equal to `a bᵀ` on `Ω`. -/
theorem isMinOn_vecMulVec_and_graphMatrix [Nontrivial n] {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {σ : Equiv.Perm n} {Ω : Set (n × n)} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hΩ : ∀ ij ∈ Ω, σ ij.1 = ij.2) :
    graphMatrix σ a b ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} ∧
      graphMatrix σ a b ≠ vecMulVec a b ∧
      IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2}
        (vecMulVec a b) ∧
      IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2}
        (graphMatrix σ a b) := by
  have hZZ := conjTranspose_graphMatrix_mul (σ := σ) ha hb
  have hZ : IsContraction (graphMatrix σ a b) := by
    rw [IsContraction, hZZ, sub_self]
    exact PosSemidef.zero
  have hnZ : nuclearNorm (graphMatrix σ a b) = Fintype.card n := by
    have h := nuclearNorm_eq_trace_of_mul_self (X := graphMatrix σ a b) PosSemidef.one
      (by rw [Matrix.one_mul, hZZ])
    rw [trace_one] at h
    exact_mod_cast h
  -- every completion pairs with `Z` to `N`
  have hlow (X' : Matrix n n ℂ) (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
      (Fintype.card n : ℝ) ≤ nuclearNorm X' := by
    have hi (i : n) : star (a i * b (σ i)) * X' i (σ i) = 1 := by
      rw [hX' (i, σ i) (hσ i), vecMulVec_apply, Complex.star_def, Complex.conj_mul', norm_mul,
        ha, hb]
      norm_num
    have htr : ((graphMatrix σ a b)ᴴ * X').trace = Fintype.card n := by
      rw [trace_conjTranspose_mul]
      simp only [graphMatrix, of_apply, apply_ite star, star_zero, ite_mul, zero_mul, sum_ite_eq,
        mem_univ, ↓reduceIte, hi, sum_const, card_univ, nsmul_eq_mul, mul_one]
    have h := norm_trace_le_nuclearNorm hZ X'
    rwa [htr, Complex.norm_natCast] at h
  have hmem : ∀ ij ∈ Ω, graphMatrix σ a b ij.1 ij.2 = vecMulVec a b ij.1 ij.2 :=
    fun ij hij => by simp [graphMatrix, vecMulVec_apply, hΩ ij hij]
  refine ⟨hmem, fun h => ?_, isMinOn_iff.mpr fun X' hX' => ?_,
    isMinOn_iff.mpr fun X' hX' => hnZ ▸ hlow X' hX'⟩
  · obtain ⟨i⟩ : Nonempty n := inferInstance
    obtain ⟨j, hj⟩ := exists_ne (σ i)
    have h1 := congrFun (congrFun h i) j
    simp only [graphMatrix, of_apply, hj.symm, ↓reduceIte, vecMulVec_apply] at h1
    have h2 : ‖a i * b j‖ = 1 := by rw [norm_mul, ha, hb, one_mul]
    rw [← h1, norm_zero] at h2
    exact zero_ne_one h2
  · rw [nuclearNorm_vecMulVec_of_norm_eq_one ha hb]
    exact hlow X' hX'

example := isMinOn_vecMulVec_and_graphMatrix (a := fun _ : Fin 2 => 1) (b := fun _ => 1) (by simp)
  (by simp) (σ := Equiv.swap 0 1) (Ω := {ij | Equiv.swap 0 1 ij.1 = ij.2}) (fun _ => rfl)
  fun _ h => h

/-- **One diagonal does not determine `ψ(a + b)`**, `review/flat_spectrum.md`, *Exact recovery
of a rank-one kernel*: for `p ≥ 2` and `Ω` the diagonal `b = a + s` of `ℤ/p × ℤ/p`, the `a + b`
kernel minimizes the nuclear norm among the matrices equal to it on `Ω`, and so does a
different matrix. -/
theorem exists_ne_isMinOn_addKernel_of_diagonal {p : ℕ} [NeZero p] [Fact (1 < p)] (s : ZMod p)
    {Ω : Set (ZMod p × ZMod p)} (hs : ∀ a, (a, a + s) ∈ Ω) (hΩ : ∀ ij ∈ Ω, ij.2 = ij.1 + s) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} (addKernel p) ∧
      ∃ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
        X ≠ addKernel p ∧
          IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X := by
  rw [addKernel_eq_vecMulVec]
  obtain ⟨hmem, hne, hK, hZ⟩ := isMinOn_vecMulVec_and_graphMatrix
    (a := (ZMod.stdAddChar : ZMod p → ℂ)) (b := ZMod.stdAddChar) (by simp) (by simp)
    (σ := Equiv.addRight s) hs fun ij hij => (hΩ ij hij).symm
  exact ⟨hK, _, hmem, hne, hZ⟩

example := exists_ne_isMinOn_addKernel_of_diagonal (p := 3) 1 (Ω := {ij | ij.2 = ij.1 + 1})
  (fun _ => rfl) fun _ h => h

end Transformer.Grokking.FlatSpectrum
