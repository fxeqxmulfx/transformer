module

public import Transformer.Grokking.FlatSpectrum.CharacterBases
public import Transformer.Grokking.FlatSpectrum.JacobiSums
public import Transformer.Grokking.FlatSpectrum.SupportRecovery

/-!
# Theorem G for `χ(a b)` and `χ(a + a b)`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G,
corollaries (added). A multiplicative character `χ` of `ZMod p` vanishes exactly at `0`
(`mulChar_ne_zero_iff`), has modulus one elsewhere (`norm_mulChar_eq_one`), and
`[χ(a b)] = χ χᵀ` (`taskKernel_mul_mulChar`). So Theorem G for vectors of modulus one on their
supports (`forall_isMinOn_vecMulVec_iff_equiv_support`) says: M recovers `[χ(a b)]` from `Ω`
exactly when the cells of `Ω` with `a, b ≠ 0` are rigid on the nonzero residues and contain the
graph of a permutation of them (`forall_isMinOn_taskKernel_mul_mulChar_iff`). The zero row and
the zero column need no cell. For `p = 3` the three cells `(1, 1)`, `(1, 2)`, `(2, 2)` of the
nine suffice; `(1, 1)` and `(2, 1)` do not.

Likewise `χ(a + a b) = χ(a) χ(1 + b)` (`taskKernel_add_mul_mulChar`), so M recovers
`[χ(a + a b)]` exactly when the cells of `Ω` with `a ≠ 0` and `b ≠ −1` are rigid on these rows
and columns and contain the graph of a bijection between them
(`forall_isMinOn_taskKernel_add_mul_mulChar_iff`). Row `0` and column `−1` need no cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {p : ℕ} [Fact p.Prime]

/-- A multiplicative character of `ZMod p` vanishes exactly at `0`. -/
lemma mulChar_ne_zero_iff (χ : MulChar (ZMod p) ℂ) (x : ZMod p) :
    χ x ≠ 0 ↔ x ≠ 0 :=
  ⟨fun h hx => h (by rw [hx, MulChar.map_zero]), fun hx h => by
    simpa [h] using norm_mulChar_eq_one χ hx⟩

example := (mulChar_ne_zero_iff (p := 3) 1 1).mpr (by decide)

/-- **`χ(a b)` from Theorem G** (`review/flat_spectrum.md`): for a multiplicative character `χ`
of `ZMod p`, M recovers `[χ(a b)]` from `Ω` exactly when the cells of `Ω` with `a, b ≠ 0` are
rigid on the nonzero residues and contain the graph of a permutation of them. The zero row and
the zero column need no cell. -/
theorem forall_isMinOn_taskKernel_mul_mulChar_iff (χ : MulChar (ZMod p) ℂ)
    (Ω : Set (ZMod p × ZMod p)) :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ)) ↔
      (∀ r s : {x : ZMod p // x ≠ 0} → ℂ,
        (∀ i j : {x : ZMod p // x ≠ 0}, ((i : ZMod p), (j : ZMod p)) ∈ Ω → r i + s j = 0) →
          ∀ i j, r i + s j = 0) ∧
        ∃ σ : Equiv.Perm {x : ZMod p // x ≠ 0}, ∀ i : {x : ZMod p // x ≠ 0},
          ((i : ZMod p), (σ i : ZMod p)) ∈ Ω := by
  have hχ : (fun x => χ x ≠ 0) = fun x : ZMod p => x ≠ 0 :=
    funext fun x => propext (mulChar_ne_zero_iff χ x)
  have hn (x : ZMod p) (hx : χ x ≠ 0) : ‖χ x‖ = 1 :=
    norm_mulChar_eq_one χ ((mulChar_ne_zero_iff χ x).mp hx)
  have := forall_isMinOn_vecMulVec_iff_equiv_support (a := ⇑χ) (b := ⇑χ) hn hn rfl Ω
  rw [hχ] at this
  rwa [taskKernel_mul_mulChar]

/-- The two nonzero residues of `ZMod 3`. -/
lemma nonzero_zmod_three (x : {x : ZMod 3 // x ≠ 0}) : x = ⟨1, by decide⟩ ∨ x = ⟨2, by decide⟩ := by
  revert x
  decide

/-- For `p = 3` three of the nine cells recover `[χ(a b)]` for the principal character. -/
example := (forall_isMinOn_taskKernel_mul_mulChar_iff (p := 3) 1 {(1, 1), (1, 2), (2, 2)}).mpr
  ⟨fun r s h i j => by
    have h11 := h ⟨1, by decide⟩ ⟨1, by decide⟩ (by simp)
    have h12 := h ⟨1, by decide⟩ ⟨2, by decide⟩ (by simp)
    have h22 := h ⟨2, by decide⟩ ⟨2, by decide⟩ (by simp)
    rcases nonzero_zmod_three i with rfl | rfl <;> rcases nonzero_zmod_three j with rfl | rfl
    · exact h11
    · exact h12
    · linear_combination h22 - h12 + h11
    · exact h22,
   Equiv.refl _, fun i => by rcases nonzero_zmod_three i with rfl | rfl <;> simp⟩

/-- For `p = 3` the cells `(1, 1)` and `(2, 1)` do not recover `[χ(a b)]`: both nonzero rows
meet only column 1. -/
example := mt (forall_isMinOn_taskKernel_mul_mulChar_iff (p := 3) 1 {(1, 1), (2, 1)}).mp
  fun ⟨_, σ, hσ⟩ => by
    have key (i) : (σ i : ZMod 3) = 1 := by
      have := hσ i
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff, Prod.mk.injEq] at this
      tauto
    exact absurd (congrArg Subtype.val (σ.injective
      (Subtype.ext ((key ⟨1, by decide⟩).trans (key ⟨2, by decide⟩).symm)))) (by decide)

/-- `χ(a + a b) = χ(a) χ(1 + b)`: the kernel `[χ(a + a b)]` is `χ ⊗ χ(1 + ·)`. -/
lemma taskKernel_add_mul_mulChar (χ : MulChar (ZMod p) ℂ) :
    taskKernel (fun a b => a + a * b) χ = vecMulVec χ fun b => χ (1 + b) := by
  ext a b
  rw [taskKernel, of_apply, vecMulVec_apply, ← map_mul, mul_one_add]

example := @taskKernel_add_mul_mulChar 3 _ 1

/-- **`χ(a + a b)` from Theorem G** (`review/flat_spectrum.md`): for a multiplicative character
`χ` of `ZMod p`, M recovers `[χ(a + a b)]` from `Ω` exactly when the cells of `Ω` with `a ≠ 0`
and `b ≠ −1` are rigid on these rows and columns and contain the graph of a bijection between
them. Row `0` and column `−1` need no cell. -/
theorem forall_isMinOn_taskKernel_add_mul_mulChar_iff (χ : MulChar (ZMod p) ℂ)
    (Ω : Set (ZMod p × ZMod p)) :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (fun a b => a + a * b) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (fun a b => a + a * b) χ ij.1 ij.2} X ↔
          X = taskKernel (fun a b => a + a * b) χ)) ↔
      (∀ (r : {x : ZMod p // x ≠ 0} → ℂ) (s : {x : ZMod p // x ≠ -1} → ℂ),
        (∀ (i : {x : ZMod p // x ≠ 0}) (j : {x : ZMod p // x ≠ -1}),
          ((i : ZMod p), (j : ZMod p)) ∈ Ω → r i + s j = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ σ : {x : ZMod p // x ≠ 0} ≃ {x : ZMod p // x ≠ -1}, ∀ i : {x : ZMod p // x ≠ 0},
          ((i : ZMod p), (σ i : ZMod p)) ∈ Ω := by
  have hχ : (fun x => χ x ≠ 0) = fun x : ZMod p => x ≠ 0 :=
    funext fun x => propext (mulChar_ne_zero_iff χ x)
  have hχ' : (fun x => χ (1 + x) ≠ 0) = fun x : ZMod p => x ≠ -1 :=
    funext fun x => propext ((mulChar_ne_zero_iff χ _).trans (not_congr add_eq_zero_iff_eq_neg'))
  have hn (x : ZMod p) (hx : χ x ≠ 0) : ‖χ x‖ = 1 :=
    norm_mulChar_eq_one χ ((mulChar_ne_zero_iff χ x).mp hx)
  have hN : Nat.card {x // χ x ≠ 0} = Nat.card {x // χ (1 + x) ≠ 0} :=
    Nat.card_congr ((Equiv.subRight 1).subtypeEquiv fun x => by simp)
  have := forall_isMinOn_vecMulVec_iff_equiv_support (a := ⇑χ) (b := fun x => χ (1 + x)) hn
    (fun x => hn (1 + x)) hN Ω
  rw [hχ, hχ'] at this
  rwa [taskKernel_add_mul_mulChar]

/-- For `p = 3` the cells off row `0` and column `−1` recover `[χ(a + a b)]` for the principal
character. -/
example :=
  let σ : {x : ZMod 3 // x ≠ 0} ≃ {x : ZMod 3 // x ≠ -1} :=
    (Equiv.subRight 1).subtypeEquiv fun x => by rw [Equiv.subRight_apply, Ne, Ne, sub_eq_neg_self]
  (forall_isMinOn_taskKernel_add_mul_mulChar_iff (p := 3) 1 {ij | ij.1 ≠ 0 ∧ ij.2 ≠ -1}).mpr
    ⟨fun r s h i j => h i j ⟨i.2, j.2⟩, σ, fun i => ⟨i.2, (σ i).2⟩⟩

end Transformer.Grokking.FlatSpectrum
