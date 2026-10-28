module

public import Transformer.Grokking.FlatSpectrum.HallWitness
public import Transformer.Grokking.FlatSpectrum.Rigidity

/-!
# Theorem E: the patterns that recover a unimodular rank-one kernel

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem E (added).
Let `|a i| = |b j| = 1`. The kernel `a bᵀ` is the unique minimum of the nuclear norm among the
matrices equal to it on `Ω` if and only if `Ω` is rigid and contains the graph of a permutation
(`forall_isMinOn_vecMulVec_iff`).

* Sufficiency is `isMinOn_vecMulVec_iff_of_perm`.
* A pattern that is not rigid admits a second kernel `a' bᵀ'` of the same norm
  (`not_forall_isMinOn_iff_of_not_rigid`).
* A pattern without a permutation admits `Y` equal to `J` on `Ω` with `‖Y‖_S1 < N`
  (`exists_nuclearNorm_lt_of_not_perm`). Then `diag a · Y · diag b` equals `a bᵀ` on `Ω` and
  has the same norm (`nuclearNorm_diagonal_mul_mul_diagonal`), so `a bᵀ` is not a minimum.

The cases `J` (`forall_isMinOn_onesMatrix_iff`) and `ψ(a + b)`
(`forall_isMinOn_addKernel_iff`) follow.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **Theorem E**, `review/flat_spectrum.md`: for `|a i| = |b j| = 1`, the kernel `a bᵀ` is the
unique minimum of the nuclear norm among the matrices equal to it on `Ω` if and only if `Ω` is
rigid and contains the graph of a permutation. -/
theorem forall_isMinOn_vecMulVec_iff {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)} :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
        X = vecMulVec a b)) ↔
      (∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω := by
  refine ⟨fun H => ⟨fun r s hrs i j => ?_, ?_⟩,
    fun ⟨hΩ, σ, hσ⟩ X hX => isMinOn_vecMulVec_iff_of_perm ha hb hσ hΩ hX⟩
  · by_contra h₀
    exact not_forall_isMinOn_iff_of_not_rigid ha hb hrs h₀ H
  · by_contra hσ
    obtain ⟨Y, hY, hlt⟩ := exists_nuclearNorm_lt_of_not_perm hσ
    have hK := isMinOn_iff.mp ((H _ fun _ _ => rfl).mpr rfl) (diagonal a * Y * diagonal b)
      fun ij hij => by
        rw [mul_diagonal, diagonal_mul, hY ij hij, onesMatrix, of_apply, mul_one, vecMulVec_apply]
    rw [nuclearNorm_diagonal_mul_mul_diagonal ha hb, nuclearNorm_vecMulVec_of_norm_eq_one ha hb]
      at hK
    linarith

example := (forall_isMinOn_vecMulVec_iff (n := Fin 2) (a := fun _ => 1) (b := fun _ => 1)
  (by simp) (by simp) (Ω := {ij | ij ≠ (1, 0)})).mpr
  ⟨rigid_ne_one_zero, 1, Fin.forall_fin_two.mpr ⟨by simp, by simp⟩⟩

/-- **Theorem E for `J`**, `review/flat_spectrum.md`: `J` is the unique minimum of the nuclear
norm among the matrices equal to it on `Ω` if and only if `Ω` is rigid and contains the graph
of a permutation. -/
theorem forall_isMinOn_onesMatrix_iff {Ω : Set (n × n)} :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2} X ↔
        X = onesMatrix n)) ↔
      (∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω := by
  have h1 : vecMulVec (fun _ : n => (1 : ℂ)) (fun _ => 1) = onesMatrix n := by
    ext i j
    rw [vecMulVec_apply, onesMatrix, of_apply, mul_one]
  rw [← h1]
  exact forall_isMinOn_vecMulVec_iff (by simp) (by simp)

/-- The cross `{ij | ij.1 = 0 ∨ ij.2 = 0}` on `Fin 3` is rigid, but rows `1` and `2` share their
only column, so `J` is not recovered from it. -/
example : ¬ ∀ X ∈ {X' : Matrix (Fin 3) (Fin 3) ℂ |
      ∀ ij ∈ {ij : Fin 3 × Fin 3 | ij.1 = 0 ∨ ij.2 = 0}, X' ij.1 ij.2 = onesMatrix _ ij.1 ij.2},
    (IsMinOn nuclearNorm {X' | ∀ ij ∈ {ij : Fin 3 × Fin 3 | ij.1 = 0 ∨ ij.2 = 0},
      X' ij.1 ij.2 = onesMatrix _ ij.1 ij.2} X ↔ X = onesMatrix _) := fun H => by
  obtain ⟨σ, hσ⟩ := (forall_isMinOn_onesMatrix_iff.mp H).2
  have h1 : σ 1 = 0 := (hσ 1).resolve_left (by decide : (1 : Fin 3) ≠ 0)
  have h2 : σ 2 = 0 := (hσ 2).resolve_left (by decide : (2 : Fin 3) ≠ 0)
  exact absurd (σ.injective (h1.trans h2.symm)) (by decide)

/-- **Theorem E for `ψ(a + b)`**, `review/flat_spectrum.md`: the `a + b` kernel is the unique
minimum of the nuclear norm among the matrices equal to it on `Ω ⊆ ℤ/p × ℤ/p` if and only if
`Ω` is rigid and contains the graph of a permutation. -/
theorem forall_isMinOn_addKernel_iff {p : ℕ} [NeZero p] {Ω : Set (ZMod p × ZMod p)} :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
        X = addKernel p)) ↔
      (∀ r s : ZMod p → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ σ : Equiv.Perm (ZMod p), ∀ i, (i, σ i) ∈ Ω := by
  rw [addKernel_eq_vecMulVec]
  exact forall_isMinOn_vecMulVec_iff (by simp) (by simp)

example := (forall_isMinOn_addKernel_iff (p := 3) (Ω := Set.univ)).mpr
  ⟨fun _ _ h i j => h (i, j) (Set.mem_univ _), 1, fun _ => Set.mem_univ _⟩

end Transformer.Grokking.FlatSpectrum
