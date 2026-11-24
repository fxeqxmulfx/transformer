module

public import Transformer.GainMatroid.Section2_ExchangeMatrices
public import Transformer.GainMatroid.Section2_K4Flats

/-!
# The parallel connection of a matrix and `I(K₄)` across a triangle

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski: `M(Δ_X A) = Δ_X M(A)` for
a matrix `A` whose columns on the triangle `X = {a, b, c}` are the last three columns of `I(K₄)` on
the rows `x, y, z` (`IsTriangleForm`). `connectionMatrix` adds to `A` a zero row `none` and three
new columns `a′, b′, c′`: the first three columns of `I(K₄)`, `x − none`, `y − none` and
`z − none`. Its matroid is the generalized parallel connection of `M(A)` and `M(K₄)` across `X`
(`IsTriangleForm.isFlat_connection_iff`).

* Deleting `X` and naming the new columns `a, b, c` gives `Δ_X A`
  (`deltaMatrix_eq_connectionMatrix`).
* The row `none` vanishes on the columns of `A` and is `-1` on the new columns, and two new columns
  differ by the column of the third edge of their triangle of `M(K₄)`
  (`IsTriangleForm.exists_triangle`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E R F : Type*} [DecidableEq R] [Field F]

/-- The matrix of the parallel connection of `A` and `I(K₄)` across a triangle on the rows
`x, y, z`: the columns of `A` with a zero row `none`, and the new columns `x − none`, `y − none` and
`z − none`. -/
def connectionMatrix (A : Matrix R E F) (x y z : R) : Matrix (Option R) (E ⊕ Fin 3) F :=
  Matrix.of fun r => Sum.elim (fun e => r.elim 0 fun w => A w e)
    fun i => (if r = some (![x, y, z] i) then 1 else 0) - if r = none then 1 else 0

/-- The column `q` of `connectionMatrix`. -/
def connectionCol (A : Matrix R E F) (x y z : R) (q : E ⊕ Fin 3) : Option R → F :=
  fun r => connectionMatrix A x y z r q

omit [DecidableEq R] in
/-- Extending a vector by a zero entry at `none`. -/
def extendZero : (R → F) →ₗ[F] Option R → F where
  toFun g r := r.elim 0 g
  map_add' g h := by
    funext r
    cases r <;> simp
  map_smul' s g := by
    funext r
    cases r <;> simp

variable {A : Matrix R E F} {a b c : E} {x y z : R}

theorem connectionCol_inl (e : E) :
    connectionCol A x y z (.inl e) = extendZero (fun r => A r e) := by
  funext r
  cases r <;> rfl

theorem connectionCol_inl_none (e : E) : connectionCol A x y z (.inl e) none = 0 := rfl

theorem connectionCol_inr_none (i : Fin 3) : connectionCol A x y z (.inr i) none = -1 := by
  simp [connectionCol, connectionMatrix]

theorem connectionCol_inr_some (i : Fin 3) (w : R) :
    connectionCol A x y z (.inr i) (some w) = if w = ![x, y, z] i then 1 else 0 := by
  change (if some w = some (![x, y, z] i) then (1 : F) else 0) -
    (if some w = none then 1 else 0) = _
  rw [ite_eq_right (Option.some_ne_none w), sub_zero]
  by_cases h : w = ![x, y, z] i
  · rw [ite_eq_left (congrArg some h), ite_eq_left h]
  · rw [ite_eq_right fun h' => h (Option.some.inj h'), ite_eq_right h]

/-- `Δ_X A` is `connectionMatrix` with `X` deleted and the new columns named `a, b, c`. -/
theorem deltaMatrix_eq_connectionMatrix [DecidableEq E] (r : Option R) (e : E) :
    deltaMatrix A a b c x y z r e = connectionMatrix A x y z r (deltaRelabel a b c e) := by
  simp only [deltaMatrix, deltaRelabel, Matrix.of_apply]
  by_cases ha : e = a
  · rw [ite_eq_left ha, ite_eq_left ha]
    rfl
  by_cases hb : e = b
  · rw [ite_eq_right ha, ite_eq_right ha, ite_eq_left hb, ite_eq_left hb]
    rfl
  by_cases hc : e = c
  · rw [ite_eq_right ha, ite_eq_right ha, ite_eq_right hb, ite_eq_right hb, ite_eq_left hc,
      ite_eq_left hc]
    rfl
  rw [ite_eq_right ha, ite_eq_right ha, ite_eq_right hb, ite_eq_right hb, ite_eq_right hc,
    ite_eq_right hc]
  rfl

namespace IsTriangleForm

theorem connectionCol_a (hform : IsTriangleForm A a b c x y z) :
    connectionCol A x y z (.inl a) =
      connectionCol A x y z (.inr 1) - connectionCol A x y z (.inr 2) := by
  funext r
  cases r with
  | none =>
    rw [Pi.sub_apply, connectionCol_inl_none, connectionCol_inr_none, connectionCol_inr_none,
      sub_self]
  | some w =>
    rw [Pi.sub_apply, connectionCol_inr_some, connectionCol_inr_some]
    exact hform.col_a w

theorem connectionCol_b (hform : IsTriangleForm A a b c x y z) :
    connectionCol A x y z (.inl b) =
      connectionCol A x y z (.inr 0) - connectionCol A x y z (.inr 2) := by
  funext r
  cases r with
  | none =>
    rw [Pi.sub_apply, connectionCol_inl_none, connectionCol_inr_none, connectionCol_inr_none,
      sub_self]
  | some w =>
    rw [Pi.sub_apply, connectionCol_inr_some, connectionCol_inr_some]
    exact hform.col_b w

theorem connectionCol_c (hform : IsTriangleForm A a b c x y z) :
    connectionCol A x y z (.inl c) =
      connectionCol A x y z (.inr 0) - connectionCol A x y z (.inr 1) := by
  funext r
  cases r with
  | none =>
    rw [Pi.sub_apply, connectionCol_inl_none, connectionCol_inr_none, connectionCol_inr_none,
      sub_self]
  | some w =>
    rw [Pi.sub_apply, connectionCol_inr_some, connectionCol_inr_some]
    exact hform.col_c w

/-- Two new columns differ by the column of the third edge of their triangle of `M(K₄)`. -/
theorem exists_triangle (hform : IsTriangleForm A a b c x y z) (i j : Fin 3) (hij : i ≠ j) :
    ∃ k : Fin 3, ({.inr i, .inr j, .inl k} : Finset (Fin 3 ⊕ Fin 3)) ∈ k4Triangles ∧
      (connectionCol A x y z (.inr i) - connectionCol A x y z (.inr j) =
          connectionCol A x y z (.inl (![a, b, c] k)) ∨
        connectionCol A x y z (.inr j) - connectionCol A x y z (.inr i) =
          connectionCol A x y z (.inl (![a, b, c] k))) := by
  fin_cases i <;> fin_cases j
  all_goals first
    | exact absurd rfl hij
    | exact ⟨2, by decide, Or.inl hform.connectionCol_c.symm⟩
    | exact ⟨2, by decide, Or.inr hform.connectionCol_c.symm⟩
    | exact ⟨1, by decide, Or.inl hform.connectionCol_b.symm⟩
    | exact ⟨1, by decide, Or.inr hform.connectionCol_b.symm⟩
    | exact ⟨0, by decide, Or.inl hform.connectionCol_a.symm⟩
    | exact ⟨0, by decide, Or.inr hform.connectionCol_a.symm⟩

end IsTriangleForm

example := (iK4_isTriangleForm (F := ℚ)).connectionCol_a

example := (iK4_isTriangleForm (F := ℚ)).connectionCol_b

example := (iK4_isTriangleForm (F := ℚ)).connectionCol_c

example := (iK4_isTriangleForm (F := ℚ)).exists_triangle 0 1 (by decide)

end Transformer.GainMatroid
