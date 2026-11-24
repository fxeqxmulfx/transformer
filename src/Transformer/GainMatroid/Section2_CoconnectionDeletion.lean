module

public import Transformer.GainMatroid.Section2_CoconnectionMaps
public import Transformer.GainMatroid.Section2_DualFlats

/-!
# Deleting the triad from the dual of `coconnectionMatrix`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. `Δ_Y M*` deletes `Y` from
the generalized parallel connection `P` of `M*` and `M(K₄)` across `Y = {a, b, c}`, and names the
new elements `a', b', c'` as `a, b, c` (`deltaRelabel`). When `P` is the dual of the matroid of
`Q = coconnectionMatrix A a b c x y z`, this leaves the dual of `M(∇_Y A)`
(`IsTriadForm.dual_comap_coconnection`). Independence in a dual is read off the spans of the columns
outside a set (`IsMatrixRepresentation.dual_indep_iff`). Adding the new rows to the rows `x, y, z`
(`contractNew`) kills the columns of `Y` and turns the other columns of `Q` into those of `∇_Y A`,
and changes a vector only by a combination of the columns of `Y`, which lie outside every image of
`deltaRelabel`.

* `deltaRelabel` misses the elements of `Y` in `E` (`deltaRelabel_ne_inl_triad`), and hits every
  other element of `E ⊕ Fin 3` (`eq_deltaRelabel_or_eq_inl_triad`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [DecidableEq E] [DecidableEq R] [Field F] {A : Matrix R E F} {v : R}
  {a b c : E} {x y z : R}

/-- `deltaRelabel` misses the elements of `Y` in `E`. -/
theorem deltaRelabel_ne_inl_triad (e : E) (i : Fin 3) :
    deltaRelabel a b c e ≠ .inl (![a, b, c] i) := by
  unfold deltaRelabel
  split_ifs with ha hb hc
  · exact Sum.inr_ne_inl
  · exact Sum.inr_ne_inl
  · exact Sum.inr_ne_inl
  · intro h
    fin_cases i
    · exact ha (Sum.inl_injective h)
    · exact hb (Sum.inl_injective h)
    · exact hc (Sum.inl_injective h)

/-- Every element of `E ⊕ Fin 3` outside the elements of `Y` in `E` is hit by `deltaRelabel`. -/
theorem eq_deltaRelabel_or_eq_inl_triad (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (q : E ⊕ Fin 3) : (∃ e, deltaRelabel a b c e = q) ∨ ∃ i : Fin 3, q = .inl (![a, b, c] i) := by
  rcases q with e | k
  · rcases eq_or_eq_triad (a := a) (b := b) (c := c) e with ⟨ha, hb, hc⟩ | ⟨i, rfl⟩
    · exact .inl ⟨e, by simp [deltaRelabel, ha, hb, hc]⟩
    · exact .inr ⟨i, rfl⟩
  · refine .inl ⟨![a, b, c] k, ?_⟩
    fin_cases k <;> simp [deltaRelabel, hab.symm, hac.symm, hbc.symm]

/-- **Deleting `Y` from the dual of `M(Q)`** and naming `a', b', c'` as `a, b, c` leaves the dual of
`M(∇_Y A)`. -/
theorem IsTriadForm.dual_comap_coconnection [Finite E] {M' : Matroid E}
    (hform : IsTriadForm A v a b c x y z)
    (hA' : IsMatrixRepresentation M' (nablaMatrix A a b c x y z)) :
    (matrixMatroid (coconnectionMatrix A a b c x y z))✶.comap (deltaRelabel a b c) = M'✶ := by
  have hab := hform.a_ne_b
  have hac := hform.a_ne_c
  have hbc := hform.b_ne_c
  have hQ := matrixMatroid_isMatrixRepresentation (coconnectionMatrix A a b c x y z)
  refine Matroid.ext_indep ?_ fun I _ => ?_
  · rw [Matroid.comap_ground_eq, Matroid.dual_ground, Matroid.dual_ground, hQ.1, hA'.1,
      Set.preimage_univ]
  rw [Matroid.comap_indep_iff, hQ.dual_indep_iff, hA'.dual_indep_iff]
  simp only [coconnectionMatrix, Matrix.of_apply, (deltaRelabel_injective a b c).injOn, and_true]
  have hsub (u : R ⊕ Fin 3 → F) : u - extendNew (contractNew x y z u) ∈
      span F (coconnectionCol A a b c x y z '' (deltaRelabel a b c '' I)ᶜ) := by
    refine span_mono (Set.image_mono ?_) (sub_extendNew_contractNew_mem_span hab hac hbc u)
    rintro _ ⟨i, rfl⟩ ⟨e, -, he⟩
    exact deltaRelabel_ne_inl_triad e i he
  have hiff (u : R ⊕ Fin 3 → F) :
      u ∈ span F (coconnectionCol A a b c x y z '' (deltaRelabel a b c '' I)ᶜ) ↔
        extendNew (contractNew x y z u) ∈
          span F (coconnectionCol A a b c x y z '' (deltaRelabel a b c '' I)ᶜ) :=
    ⟨fun hu => by simpa using sub_mem hu (hsub u), fun hu => by simpa using add_mem (hsub u) hu⟩
  constructor
  · intro h e
    rw [← contractNew_deltaRelabel hab hac hbc e]
    refine (map_span_le _ _ _).2 ?_ (mem_map_of_mem (h (deltaRelabel a b c e)))
    rintro _ ⟨q, hq, rfl⟩
    rcases eq_deltaRelabel_or_eq_inl_triad hab hac hbc q with ⟨f, rfl⟩ | ⟨i, rfl⟩
    · rw [contractNew_deltaRelabel hab hac hbc]
      exact subset_span ⟨f, fun hf => hq ⟨f, hf, rfl⟩, rfl⟩
    · rw [contractNew_triad hab hac hbc]
      exact zero_mem _
  · intro h q
    rw [hiff]
    rcases eq_deltaRelabel_or_eq_inl_triad hab hac hbc q with ⟨e, rfl⟩ | ⟨i, rfl⟩
    · rw [contractNew_deltaRelabel hab hac hbc]
      refine (map_span_le _ _ _).2 ?_ (mem_map_of_mem (h e))
      rintro _ ⟨f, hf, rfl⟩
      dsimp only
      rw [← contractNew_deltaRelabel hab hac hbc, ← hiff]
      exact subset_span ⟨_, fun ⟨f', hf', heq⟩ => hf (deltaRelabel_injective a b c heq ▸ hf'), rfl⟩
    · rw [contractNew_triad hab hac hbc, map_zero]
      exact zero_mem _

example := deltaRelabel_ne_inl_triad (a := (0 : Fin 6)) (b := 1) (c := 2) 3 0

example := eq_deltaRelabel_or_eq_inl_triad (a := (0 : Fin 6)) (b := 1) (c := 2) (by decide)
  (by decide) (by decide) (.inr 0)

example := (iK4_isTriadForm (F := ℚ)).dual_comap_coconnection
  (matrixMatroid_isMatrixRepresentation _)

end Transformer.GainMatroid
