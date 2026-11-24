module

public import Transformer.GainMatroid.Section2_ExchangeMatrices
public import Transformer.GainMatroid.Section2_K4NewColumns

/-!
# The matrix of the dual of the parallel connection across a triad

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski: `M(∇_Y A) = ∇_Y M(A)` for a
matrix `A` whose columns on the triad `Y = {a, b, c}` are the first three columns of `I(K₄)` on the
rows `x, y, z, v` (`IsTriadForm`). Since `∇_Y M = (Δ_Y M*)*`, this needs the generalized parallel
connection of `M*` and `M(K₄)` across `Y`. It is the dual of the matroid of `coconnectionMatrix`:
three new rows `x', y', z'` (`inr 0, 1, 2`), the columns of `Y` replaced by `x − x'`, `y − y'` and
`z − z'`, and three new columns `a', b', c'` forming the triangle `y' − z'`, `x' − z'`, `x' − y'`.
For the graph of `A`, the vertex `v` of the triad is split into the triangle `x'y'z'`.

* On the new rows, the columns of the edges of `K₄` are `k4NewCol` (`Section2_K4NewColumns`); the
  other columns vanish there.
* `a, b, c` are distinct (`IsTriadForm.a_ne_b`, `IsTriadForm.a_ne_c`, `IsTriadForm.b_ne_c`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [Field F] [DecidableEq R]

/-- The columns of `coconnectionMatrix`: `x − x'`, `y − y'`, `z − z'` on `a, b, c`, the columns of
`A` with zeros on the new rows elsewhere in `E`, and the triangle `y' − z'`, `x' − z'`, `x' − y'`
on the new elements `a', b', c'`. -/
def coconnectionCol [DecidableEq E] (A : Matrix R E F) (a b c : E) (x y z : R) :
    E ⊕ Fin 3 → R ⊕ Fin 3 → F
  | .inl e =>
    if e = a then Sum.elim (Pi.single x 1) (k4NewCol (.inl 0))
    else if e = b then Sum.elim (Pi.single y 1) (k4NewCol (.inl 1))
    else if e = c then Sum.elim (Pi.single z 1) (k4NewCol (.inl 2))
    else Sum.elim (fun r => A r e) 0
  | .inr k => Sum.elim 0 (k4NewCol (.inr k))

/-- The matrix whose matroid is dual to the generalized parallel connection of `M(A)*` and `M(K₄)`
across the triad `a, b, c` on the rows `x, y, z`. -/
def coconnectionMatrix [DecidableEq E] (A : Matrix R E F) (a b c : E) (x y z : R) :
    Matrix (R ⊕ Fin 3) (E ⊕ Fin 3) F :=
  Matrix.of fun r q => coconnectionCol A a b c x y z q r

variable {A : Matrix R E F} {v : R} {a b c : E} {x y z : R}

namespace IsTriadForm

theorem a_ne_b (hform : IsTriadForm A v a b c x y z) : a ≠ b := fun h => by
  have hx := hform.col_a x
  rw [h, hform.col_b x] at hx
  simp [hform.ne_xy, hform.ne_vx.symm] at hx

theorem a_ne_c (hform : IsTriadForm A v a b c x y z) : a ≠ c := fun h => by
  have hx := hform.col_a x
  rw [h, hform.col_c x] at hx
  simp [hform.ne_xz, hform.ne_vx.symm] at hx

theorem b_ne_c (hform : IsTriadForm A v a b c x y z) : b ≠ c := fun h => by
  have hy := hform.col_b y
  rw [h, hform.col_c y] at hy
  simp [hform.ne_yz, hform.ne_vy.symm] at hy

/-- The column of `A` at the `i`-th element of `Y` is `x_i − v`. -/
theorem col_triad (hform : IsTriadForm A v a b c x y z) (i : Fin 3) (r : R) :
    A r (![a, b, c] i) = (Pi.single (![x, y, z] i) 1 : R → F) r - (Pi.single v 1 : R → F) r := by
  fin_cases i
  · simp [hform.col_a r, Pi.single_apply]
  · simp [hform.col_b r, Pi.single_apply]
  · simp [hform.col_c r, Pi.single_apply]

theorem triad_v (hform : IsTriadForm A v a b c x y z) (i : Fin 3) : A v (![a, b, c] i) = -1 := by
  have hv : v ≠ ![x, y, z] i := by
    fin_cases i
    exacts [hform.ne_vx, hform.ne_vy, hform.ne_vz]
  rw [hform.col_triad, Pi.single_eq_of_ne hv, Pi.single_eq_same, zero_sub]

end IsTriadForm

/-- An element of `E` is outside `Y` or one of `a, b, c`. -/
theorem eq_or_eq_triad (e : E) : (e ≠ a ∧ e ≠ b ∧ e ≠ c) ∨ ∃ i : Fin 3, e = ![a, b, c] i := by
  by_cases ha : e = a
  · exact .inr ⟨0, ha⟩
  by_cases hb : e = b
  · exact .inr ⟨1, hb⟩
  by_cases hc : e = c
  · exact .inr ⟨2, hc⟩
  exact .inl ⟨ha, hb, hc⟩

/-- An element of `E ⊕ Fin 3` is an element of `E` outside `Y` or an edge of `K₄`. -/
theorem eq_inl_or_eq_deltaEmbed (q : E ⊕ Fin 3) :
    (∃ f, f ≠ a ∧ f ≠ b ∧ f ≠ c ∧ q = .inl f) ∨ ∃ g, q = deltaEmbed a b c g := by
  rcases q with e | k
  · rcases eq_or_eq_triad (a := a) (b := b) (c := c) e with ⟨ha, hb, hc⟩ | ⟨i, rfl⟩
    · exact .inl ⟨e, ha, hb, hc, rfl⟩
    · exact .inr ⟨.inl i, rfl⟩
  · exact .inr ⟨.inr k, rfl⟩

section Columns

variable [DecidableEq E]

theorem coconnectionCol_inl {e : E} (ha : e ≠ a) (hb : e ≠ b) (hc : e ≠ c) :
    coconnectionCol A a b c x y z (.inl e) = Sum.elim (fun r => A r e) 0 := by
  simp [coconnectionCol, ha, hb, hc]

theorem coconnectionCol_triad (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (i : Fin 3) :
    coconnectionCol A a b c x y z (.inl (![a, b, c] i)) =
      Sum.elim (Pi.single (![x, y, z] i) 1) (k4NewCol (.inl i)) := by
  fin_cases i <;> simp [coconnectionCol, hab.symm, hac.symm, hbc.symm]

theorem coconnectionCol_inr (k : Fin 3) :
    coconnectionCol A a b c x y z (.inr k) = Sum.elim (0 : R → F) (k4NewCol (.inr k)) := rfl

/-- Each new column is the sum or difference of the other two. -/
theorem coconnectionCol_inr_mem_span (k : Fin 3) :
    coconnectionCol A a b c x y z (.inr k) ∈
      span F (coconnectionCol A a b c x y z '' {q | ∃ j, j ≠ k ∧ q = .inr j}) := by
  have h (j : Fin 3) (hj : j ≠ k) : coconnectionCol A a b c x y z (.inr j) ∈
      span F (coconnectionCol A a b c x y z '' {q | ∃ j, j ≠ k ∧ q = .inr j}) :=
    subset_span ⟨.inr j, ⟨j, hj, rfl⟩, rfl⟩
  have h1 : coconnectionCol A a b c x y z (.inr 1) =
      coconnectionCol A a b c x y z (.inr 0) + coconnectionCol A a b c x y z (.inr 2) := by
    rw [coconnectionCol_inr, coconnectionCol_inr, coconnectionCol_inr, k4NewCol_inr_one,
      ← Sum.elim_add_add, add_zero]
  fin_cases k
  · simpa [h1] using sub_mem (h 1 (by decide)) (h 2 (by decide))
  · simpa [h1] using add_mem (h 0 (by decide)) (h 2 (by decide))
  · simpa [h1] using sub_mem (h 1 (by decide)) (h 0 (by decide))

/-- The difference of two new rows lies in the span of a set holding the new column of the third
index. -/
theorem newDiff_mem_span (i j : Fin 3) {T : Set (E ⊕ Fin 3)}
    (h : i ≠ j → .inr (thirdIndex i j) ∈ T) :
    Sum.elim (0 : R → F) (Pi.single i 1 - Pi.single j 1) ∈
      span F (coconnectionCol A a b c x y z '' T) := by
  by_cases hij : i = j
  · simp [hij]
  have hmem : coconnectionCol A a b c x y z (.inr (thirdIndex i j)) ∈
      span F (coconnectionCol A a b c x y z '' T) := subset_span ⟨_, h hij, rfl⟩
  rw [coconnectionCol_inr] at hmem
  rcases k4NewCol_thirdIndex (F := F) i j hij with h' | h' <;> rw [h'] at hmem
  · exact hmem
  · simpa [← Sum.elim_neg_neg] using neg_mem hmem

/-- A submodule holding the difference of two new rows holds the new column of the third index. -/
theorem coconnectionCol_thirdIndex_mem (i j : Fin 3) (hij : i ≠ j)
    {N : Submodule F (R ⊕ Fin 3 → F)}
    (h : Sum.elim (0 : R → F) (Pi.single i 1 - Pi.single j 1) ∈ N) :
    coconnectionCol A a b c x y z (.inr (thirdIndex i j)) ∈ N := by
  rw [coconnectionCol_inr]
  rcases k4NewCol_thirdIndex (F := F) i j hij with h' | h' <;> rw [h']
  · exact h
  · simpa [← Sum.elim_neg_neg] using neg_mem h

end Columns

example := (iK4_isTriadForm (F := ℚ)).a_ne_b

example := (iK4_isTriadForm (F := ℚ)).a_ne_c

example := (iK4_isTriadForm (F := ℚ)).b_ne_c

example := (iK4_isTriadForm (F := ℚ)).col_triad 0 3

example := (iK4_isTriadForm (F := ℚ)).triad_v 0

example := coconnectionCol_inl (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1) (c := 2)
  (x := 0) (y := 1) (z := 2) (e := 3) (by decide) (by decide) (by decide)

example := coconnectionCol_triad (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (x := 0) (y := 1) (z := 2)
  (a := 0) (b := 1) (c := 2) (by decide) (by decide) (by decide)

example := newDiff_mem_span (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1) (c := 2)
  (x := 0) (y := 1) (z := 2) 0 1 (T := Set.univ) fun _ => trivial

example := coconnectionCol_thirdIndex_mem (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1)
  (c := 2) (x := 0) (y := 1) (z := 2) 0 1 (by decide) (N := ⊤) trivial

end Transformer.GainMatroid
