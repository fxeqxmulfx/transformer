module

public import Mathlib.LinearAlgebra.Dimension.Constructions
public import Mathlib.Data.Set.Card
public import Transformer.Grokking.FlatSpectrum.TablePart.Defs

/-!
# PT1. How many cells determine a class

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT1. For a class `H` with
two different members on `N` cells,

  `every part of m cells determines H  ⟺  m > N − δ(H)` (`forall_determines_iff`).

Two different members agree on at most `N − δ` cells, so a larger part tells them apart, and the
`N − δ` cells where an extremal pair agrees do not. A part that determines `H` must name a member:
`|H| ≤ |A|^{|Ω|}` (`ncard_le_pow_of_determines`). For a linear space `H` over a field this reads
`dim H ≤ |Ω|` (`finrank_le_card_of_determines`), a part determines `H` exactly when no nonzero
member vanishes on it (`determines_submodule_iff`), and `δ(H)` is the least number of nonzero
cells of a nonzero member (`minDistance_submodule`). With no restriction on the class, `δ = 1` and
only the whole table determines it (`minDistance_univ`, `determines_univ_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset

variable {B A : Type*} [Fintype B] [DecidableEq A] {H : Set (B → A)}

/-- **PT1** (`review/flat_spectrum.md`): for a class with two different members, every part of `m`
cells determines it exactly when `m > N − δ(H)`. -/
theorem forall_determines_iff (hH : H.Nontrivial) (m : ℕ) :
    (∀ Ω : Finset B, #Ω = m → Determines H Ω) ↔ Fintype.card B - minDistance H < m := by
  classical
  constructor
  · intro h
    by_contra hm
    obtain ⟨f, hf, f', hf', hne, hd⟩ := exists_hammingDist_eq_minDistance hH
    have hagree : #{x | f x = f' x} = Fintype.card B - minDistance H := by
      have hsplit := card_filter_add_card_filter_not (s := (univ : Finset B))
        (fun x => f x = f' x)
      rw [card_univ] at hsplit
      have hdist : #{x | ¬ f x = f' x} = minDistance H := hd
      omega
    obtain ⟨Ω, hΩ, hcard⟩ := exists_subset_card_eq (s := {x | f x = f' x})
      (n := m) (by omega)
    exact hne (h Ω hcard f hf f' hf' fun x hx => (mem_filter.1 (hΩ (by exact_mod_cast hx))).2)
  · intro hm Ω hΩ g hg g' hg' hagree
    by_contra hne
    have hle := minDistance_le hg hg' hne
    have hout : hammingDist g g' ≤ #Ωᶜ := card_le_card fun x hx => by
      have hx' : g x ≠ g' x := (mem_filter.1 hx).2
      exact mem_compl.2 fun hxΩ => hx' (hagree x (by exact_mod_cast hxΩ))
    rw [card_compl, hΩ] at hout
    have hpos : 0 < hammingDist g g' := hammingDist_pos.2 hne
    omega

example := forall_determines_iff (B := Fin 2) (A := Bool) (H := Set.univ) Set.nontrivial_univ 1

omit [DecidableEq A] in
/-- A part that determines `H` can name each member: `|H| ≤ |A|^{|Ω|}`
(`review/flat_spectrum.md`, PT1). -/
theorem ncard_le_pow_of_determines [Fintype A] {Ω : Finset B} (h : Determines H Ω) :
    H.ncard ≤ Fintype.card A ^ #Ω := by
  have hinj : Function.Injective fun (f : H) (x : Ω) => (f : B → A) x := fun f f' hff =>
    Subtype.ext (h f f.2 f' f'.2 fun x hx => congrFun hff ⟨x, by exact_mod_cast hx⟩)
  calc H.ncard = Nat.card H := (Nat.card_coe_set_eq H).symm
    _ ≤ Nat.card (Ω → A) := Nat.card_le_card_of_injective _ hinj
    _ = Fintype.card A ^ #Ω := by
      rw [Nat.card_fun, Nat.card_eq_fintype_card, Nat.card_eq_fintype_card, Fintype.card_coe]

example : (Set.univ : Set (Fin 2 → Bool)).ncard ≤ Fintype.card Bool ^ #(univ : Finset (Fin 2)) :=
  ncard_le_pow_of_determines (Ω := univ) fun _ _ _ _ h => funext fun x =>
    h x (mem_coe.2 (mem_univ x))

omit [Fintype B] [DecidableEq A] in
/-- Without a restriction on the class, only the whole table determines it
(`review/flat_spectrum.md`, PT1). -/
theorem determines_univ_iff [Nontrivial A] (Ω : Set B) :
    Determines (Set.univ : Set (B → A)) Ω ↔ Ω = Set.univ := by
  classical
  constructor
  · intro h
    by_contra hΩ
    obtain ⟨x, hx⟩ := (Set.ne_univ_iff_exists_notMem Ω).1 hΩ
    obtain ⟨a, b, hab⟩ := exists_pair_ne A
    have heq := h (Function.const B a) (Set.mem_univ _) (Function.update (Function.const B a) x b)
      (Set.mem_univ _) fun y hy => by
        have hyx : y ≠ x := fun e => hx (e ▸ hy)
        simp [Function.update_of_ne hyx]
    have hval := congrFun heq x
    simp only [Function.const_apply, Function.update_self] at hval
    exact hab hval
  · rintro rfl f _ f' _ h
    exact funext fun x => h x (Set.mem_univ x)

example := (determines_univ_iff (B := Fin 2) (A := Bool) Set.univ).2 rfl

/-- Without a restriction on the class, two tables can differ in one cell: `δ = 1`
(`review/flat_spectrum.md`, PT1). -/
theorem minDistance_univ [Nonempty B] [Nontrivial A] :
    minDistance (Set.univ : Set (B → A)) = 1 := by
  classical
  obtain ⟨x⟩ := ‹Nonempty B›
  obtain ⟨a, b, hab⟩ := exists_pair_ne A
  have hne : Function.const B a ≠ Function.update (Function.const B a) x b := fun h => by
    have := congrFun h x
    simp only [Function.const_apply, Function.update_self] at this
    exact hab this
  have hdist : hammingDist (Function.const B a) (Function.update (Function.const B a) x b) = 1 := by
    refine card_eq_one.2 ⟨x, ext fun y => ?_⟩
    simp only [mem_filter, mem_univ, true_and, mem_singleton]
    by_cases hy : y = x
    · subst hy
      simpa using hab
    · simp [hy]
  have hle := minDistance_le (Set.mem_univ _) (Set.mem_univ _) hne
  rw [hdist] at hle
  obtain ⟨f, _, f', _, hff, hd⟩ :=
    exists_hammingDist_eq_minDistance (H := (Set.univ : Set (B → A))) Set.nontrivial_univ
  have hpos : 0 < hammingDist f f' := hammingDist_pos.2 hff
  omega

example : minDistance (Set.univ : Set (Fin 3 → Bool)) = 1 := minDistance_univ

section Linear

variable {F : Type*} [Field F]

omit [Fintype B] [DecidableEq A] in
/-- For a linear class, a part determines it exactly when no nonzero member vanishes on it
(`review/flat_spectrum.md`, PT1). -/
theorem determines_submodule_iff (H : Submodule F (B → F)) (Ω : Set B) :
    Determines (H : Set (B → F)) Ω ↔ ∀ f ∈ H, (∀ x ∈ Ω, f x = 0) → f = 0 :=
  ⟨fun h f hf h0 => h f hf 0 H.zero_mem (by simpa using h0), fun h f hf f' hf' hag =>
    sub_eq_zero.1 (h (f - f') (H.sub_mem hf hf') fun x hx => by simp [hag x hx])⟩

example := (determines_submodule_iff (⊤ : Submodule ℚ (Fin 2 → ℚ)) Set.univ).2
  fun _ _ h => funext fun x => h x (Set.mem_univ x)

omit [DecidableEq A] in
/-- A part that determines a linear class has at least `dim H` cells
(`review/flat_spectrum.md`, PT1). -/
theorem finrank_le_card_of_determines (H : Submodule F (B → F)) {Ω : Finset B}
    (h : Determines (H : Set (B → F)) Ω) : Module.finrank F H ≤ #Ω := by
  let r : H →ₗ[F] (Ω → F) := LinearMap.funLeft F F ((↑) : Ω → B) ∘ₗ H.subtype
  have hr : Function.Injective r := (injective_iff_map_eq_zero r).2 fun f hf =>
    Subtype.ext ((determines_submodule_iff H _).1 h f f.2 fun x hx =>
      congrFun hf ⟨x, by exact_mod_cast hx⟩)
  simpa using LinearMap.finrank_le_finrank_of_injective hr

example : Module.finrank ℚ (⊤ : Submodule ℚ (Fin 2 → ℚ)) ≤ #(univ : Finset (Fin 2)) :=
  finrank_le_card_of_determines (Ω := univ) ⊤ fun _ _ _ _ h => funext fun x =>
    h x (mem_coe.2 (mem_univ x))

omit [DecidableEq A] in
/-- For a linear class, `δ(H)` is the least number of nonzero cells of a nonzero member
(`review/flat_spectrum.md`, PT1). -/
theorem minDistance_submodule [DecidableEq F] (H : Submodule F (B → F)) :
    minDistance (H : Set (B → F)) = sInf {d | ∃ f ∈ H, f ≠ 0 ∧ hammingNorm f = d} := by
  unfold minDistance
  congr 1
  ext d
  constructor
  · rintro ⟨f, hf, f', hf', hne, rfl⟩
    refine ⟨-f + f', H.add_mem (H.neg_mem hf) hf', fun h0 => hne ?_,
      (hammingDist_eq_hammingNorm f f').symm⟩
    exact neg_add_eq_zero.1 h0
  · rintro ⟨f, hf, hne, rfl⟩
    exact ⟨0, H.zero_mem, f, hf, hne.symm, by rw [hammingDist_zero_left]⟩

example : minDistance ((⊤ : Submodule ℚ (Fin 1 → ℚ)) : Set (Fin 1 → ℚ)) =
    sInf {d | ∃ f ∈ (⊤ : Submodule ℚ (Fin 1 → ℚ)), f ≠ 0 ∧ hammingNorm f = d} :=
  minDistance_submodule ⊤

end Linear

end Transformer.Grokking.FlatSpectrum.TablePart
