module

public import Mathlib.Combinatorics.Hall.Basic
public import Transformer.Grokking.FlatSpectrum.WeightedCriterion

/-!
# Theorem G for vectors of modulus one on their supports

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G,
corollary (added). Let `|a_i| = 1` on the support `S_a = {i | a_i ≠ 0}`, `|b_j| = 1` on
`S_b`, and `#S_a = #S_b`. M recovers `a bᵀ` from `Ω` exactly when the cells of `Ω` in
`S_a × S_b` are rigid on `S_a × S_b` and contain the graph of a bijection `S_a → S_b`
(`forall_isMinOn_vecMulVec_iff_equiv_support`). Rows outside `S_a` and columns outside `S_b`
need no cell.

* Rigidity on the supports is rigidity of the pattern cut down to `S_a × S_b`
  (`rigid_support_iff_subtype`).
* Here `∑_R |a_i|² = #(R ∩ S_a)` (`sum_norm_sq_eq_card`), so Theorem G's inequality reads
  `#(R ∩ S_a) ≤ #(N(R) ∩ S_b)`. Hall's theorem
  (`Finset.all_card_le_biUnion_card_iff_exists_injective`) turns it into an injection
  `S_a → S_b` inside `Ω`, a bijection since `#S_a = #S_b` (`forall_sum_le_iff_exists_equiv`).

Theorem E is the case without zero entries; `CharacterRecovery` applies the corollary to
multiplicative characters.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- Rigidity on the supports of `a` and `b` is rigidity of the pattern cut down to them. -/
lemma rigid_support_iff_subtype (a : m → ℂ) (b : n → ℂ) (Ω : Set (m × n)) :
    (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
      ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0) ↔
      ∀ (r : {i // a i ≠ 0} → ℂ) (s : {j // b j ≠ 0} → ℂ),
        (∀ (i : {i // a i ≠ 0}) (j : {j // b j ≠ 0}), ((i : m), (j : n)) ∈ Ω → r i + s j = 0) →
          ∀ i j, r i + s j = 0 := by
  refine ⟨fun h r s hrs i j => ?_, fun h r s hrs i j hi hj =>
    h (fun i => r i) (fun j => s j) (fun i j hij => hrs _ hij i.2 j.2) ⟨i, hi⟩ ⟨j, hj⟩⟩
  have := h (fun i => if hi : a i ≠ 0 then r ⟨i, hi⟩ else 0)
    (fun j => if hj : b j ≠ 0 then s ⟨j, hj⟩ else 0) (fun ij hij hi hj => by
      rw [dite_eq_left hi, dite_eq_left hj]
      exact hrs ⟨ij.1, hi⟩ ⟨ij.2, hj⟩ hij) i j i.2 j.2
  rwa [dite_eq_left i.2, dite_eq_left j.2] at this

omit [Fintype m] [DecidableEq m] in
/-- With `|a_i| = 1` on the support of `a`, `∑_R |a_i|²` counts the support in `R`. -/
lemma sum_norm_sq_eq_card {a : m → ℂ} (ha : ∀ i, a i ≠ 0 → ‖a i‖ = 1) (R : Finset m) :
    ∑ i ∈ R, ‖a i‖ ^ 2 = #{i ∈ R | a i ≠ 0} := by
  rw [← sum_boole]
  refine sum_congr rfl fun i _ => ?_
  by_cases hi : a i = 0
  · simp [hi]
  · simp [hi, ha i hi]

example := sum_norm_sq_eq_card (a := ![1, 0]) (by simp [Fin.forall_fin_two]) univ

omit [DecidableEq m] in
/-- **Hall's theorem for Theorem G**: let `|a_i| = 1` on the support of `a`, `|b_j| = 1` on the
support of `b`, and let the supports have the same size. Every set of rows `R` has
`‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²` exactly when `Ω` contains the graph of a bijection
from the support of `a` to that of `b`. -/
lemma forall_sum_le_iff_exists_equiv {a : m → ℂ} {b : n → ℂ}
    (ha : ∀ i, a i ≠ 0 → ‖a i‖ = 1) (hb : ∀ j, b j ≠ 0 → ‖b j‖ = 1)
    (hN : Nat.card {i // a i ≠ 0} = Nat.card {j // b j ≠ 0}) (Ω : Set (m × n)) :
    (∀ R : Finset m, (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 ≤
        (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2) ↔
      ∃ σ : {i // a i ≠ 0} ≃ {j // b j ≠ 0}, ∀ i : {i // a i ≠ 0}, ((i : m), (σ i : n)) ∈ Ω := by
  have hA : #{i | a i ≠ 0} = Nat.card {i // a i ≠ 0} := by
    rw [Nat.card_eq_fintype_card, Fintype.card_subtype]
  have hB : #{j | b j ≠ 0} = Nat.card {i // a i ≠ 0} := by
    rw [hN, Nat.card_eq_fintype_card, Fintype.card_subtype]
  simp only [sum_norm_sq_eq_card ha, sum_norm_sq_eq_card hb, hA, hB]
  obtain h0 | hpos := (Nat.card {i // a i ≠ 0}).eq_zero_or_pos
  · have : IsEmpty {i // a i ≠ 0} := Finite.card_eq_zero_iff.mp h0
    have : IsEmpty {j // b j ≠ 0} := Finite.card_eq_zero_iff.mp (hN.symm.trans h0)
    simp only [h0, Nat.cast_zero, mul_zero, zero_mul, le_refl, implies_true, true_iff]
    exact ⟨Equiv.equivOfIsEmpty _ _, fun i => isEmptyElim i⟩
  have key (R : Finset m) : ((#{i ∈ R | a i ≠ 0} : ℝ) * Nat.card {i // a i ≠ 0} ≤
      Nat.card {i // a i ≠ 0} * #{j ∈ nbhd Ω R | b j ≠ 0}) ↔
      #{i ∈ R | a i ≠ 0} ≤ #{j ∈ nbhd Ω R | b j ≠ 0} := by
    rw [mul_comm, mul_le_mul_iff_of_pos_left (Nat.cast_pos.mpr hpos), Nat.cast_le]
  simp only [key]
  refine ⟨fun h => ?_, fun ⟨σ, hσ⟩ R => ?_⟩
  · classical
    obtain ⟨f, hf, hft⟩ := (all_card_le_biUnion_card_iff_exists_injective
      fun i : {i // a i ≠ 0} => {j : {j // b j ≠ 0} | ((i : m), (j : n)) ∈ Ω}).mp fun s => by
        have hs : ∀ i ∈ s.map (Function.Embedding.subtype _), a i ≠ 0 := fun i hi => by
          obtain ⟨i, -, rfl⟩ := mem_map.mp hi
          exact i.2
        have hR := h (s.map (Function.Embedding.subtype _))
        rw [filter_true_of_mem hs, card_map, ← card_subtype] at hR
        refine hR.trans (card_le_card fun j hj => ?_)
        obtain ⟨_, hi, hij⟩ := mem_nbhd.mp (mem_subtype.mp hj)
        obtain ⟨i, his, rfl⟩ := mem_map.mp hi
        exact mem_biUnion.mpr ⟨i, his, by simpa using hij⟩
    exact ⟨Equiv.ofBijective f ((Nat.bijective_iff_injective_and_card f).mpr ⟨hf, hN⟩),
      fun i => by simpa using hft i⟩
  · rw [← card_subtype, ← card_subtype]
    refine card_le_card_of_injOn σ (fun i hi => ?_) σ.injective.injOn
    rw [mem_coe, mem_subtype] at hi ⊢
    exact mem_nbhd.mpr ⟨i, hi, hσ i⟩

/-- **Theorem G for vectors of modulus one on their supports** (`review/flat_spectrum.md`,
corollary of Theorem G): let `|a_i| = 1` on the support of `a`, `|b_j| = 1` on the support of
`b`, and let the supports have the same size. M recovers `a bᵀ` from `Ω` exactly when the cells
of `Ω` in the product of the supports are rigid on it and contain the graph of a bijection
between the supports. -/
theorem forall_isMinOn_vecMulVec_iff_equiv_support {a : m → ℂ} {b : n → ℂ}
    (ha : ∀ i, a i ≠ 0 → ‖a i‖ = 1) (hb : ∀ j, b j ≠ 0 → ‖b j‖ = 1)
    (hN : Nat.card {i // a i ≠ 0} = Nat.card {j // b j ≠ 0}) (Ω : Set (m × n)) :
    (∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ (r : {i // a i ≠ 0} → ℂ) (s : {j // b j ≠ 0} → ℂ),
        (∀ (i : {i // a i ≠ 0}) (j : {j // b j ≠ 0}), ((i : m), (j : n)) ∈ Ω → r i + s j = 0) →
          ∀ i j, r i + s j = 0) ∧
        ∃ σ : {i // a i ≠ 0} ≃ {j // b j ≠ 0}, ∀ i : {i // a i ≠ 0}, ((i : m), (σ i : n)) ∈ Ω := by
  rw [forall_isMinOn_vecMulVec_iff_support, rigid_support_iff_subtype,
    forall_sum_le_iff_exists_equiv ha hb hN]

/-- The single cell `(0, 0)` recovers `(1, 0)(1, 0)ᵀ`: row 1 and column 1 need no cell. -/
example := (forall_isMinOn_vecMulVec_iff_equiv_support (a := ![1, 0]) (b := ![1, 0])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) rfl
  {((0 : Fin 2), (0 : Fin 2))}).mpr
  (have h0 : ∀ k : Fin 2, ![(1 : ℂ), 0] k ≠ 0 → k = 0 :=
    Fin.forall_fin_two.mpr ⟨fun _ => rfl, fun hk => absurd (by simp) hk⟩
  ⟨fun r s h i j => h i j (by rw [Set.mem_singleton_iff, h0 i i.2, h0 j j.2]), Equiv.refl _,
    fun i => by rw [Set.mem_singleton_iff, Equiv.refl_apply, h0 i i.2]⟩)

end Transformer.Grokking.FlatSpectrum
